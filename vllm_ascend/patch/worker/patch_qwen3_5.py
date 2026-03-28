#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from collections.abc import Iterable
# mypy: ignore-errors


import torch
from einops import rearrange
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.ops.triton.fla.sigmoid_gating import fused_sigmoid_gating_delta_rule_update
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.utils import enable_sp, vllm_version_is


def to_int64_tuple(t):
    t = t.to(torch.int64)
    if t.dim() == 0:
        return (t.item(),)
    return tuple(t.tolist())


class AscendQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        num_tokens = mixed_qkvz.size(0)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        ba, _ = self.in_proj_ba(hidden_states)
        b, a = ba.chunk(2, dim=-1)

        b = b.contiguous()
        a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )
        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        o_out, _ = self.out_proj(core_attn_out)
        actual_num_tokens = o_out.shape[0]
        output[:actual_num_tokens] = o_out

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        # Core attention computation (called by custom op).

        # NOTE: The processing logic of Qwen3_5GatedDeltaNet is the same as Qwen3NextGatedDeltaNet.
        # However, because the ops `torch_npu.npu_recurrent_gated_delta_rule`
        # currently does not support `ssm_state` inputs in float32 format,
        # we temporarily retain the current _forward_core implementation.
        # Once the ops supports float32 `ssm_state`, this patch should be removed.

        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        if vllm_version_is("0.18.0"):
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        else:
            self_kv_cache = self.kv_cache
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        if not enable_sp():
            mixed_qkv = mixed_qkv[:num_actual_tokens]
            b = b[:num_actual_tokens]
            a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][: attn_metadata.num_spec_decodes],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                conv_weights_T = conv_weights.transpose(0, 1)
                activation_num = 1 if self.activation else 0
                mixed_qkv_non_spec = torch.ops._C_ascend.npu_causal_conv1d_custom(
                    mixed_qkv_non_spec,
                    conv_weights_T,
                    conv_state=self_kv_cache[0],
                    bias_opt=self.conv1d.bias,
                    query_start_loc_opt=to_int64_tuple(non_spec_query_start_loc),
                    cache_indices_opt=to_int64_tuple(non_spec_state_indices_tensor),
                    initial_state_mode_opt=to_int64_tuple(has_initial_state),
                    num_accepted_tokens_opt=[],
                    activation_mode=activation_num,
                    pad_slot_id=PAD_SLOT_ID,
                    run_mode=0,
                )
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[: attn_metadata.num_actual_tokens],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

        if attn_metadata.num_prefills > 0 or spec_sequence_masks is not None:
            g, beta = fused_gdn_gating_patch(self.A_log, a, b, self.dt_bias)
            if spec_sequence_masks is not None:
                if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                    g_spec = g
                    beta_spec = beta
                    g_non_spec = None
                    beta_non_spec = None
                else:
                    g_spec = g.index_select(1, spec_token_indx)
                    beta_spec = beta.index_select(1, spec_token_indx)
                    g_non_spec = g.index_select(1, non_spec_token_indx)
                    beta_non_spec = beta.index_select(1, non_spec_token_indx)
            else:
                g_spec = None
                beta_spec = None
                g_non_spec = g
                beta_non_spec = beta

            # 2. Recurrent attention

            # 2.1: Process the multi-query part
            if spec_sequence_masks is not None:
                core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    g=g_spec,
                    beta=beta_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out_spec, last_recurrent_state = None, None

            # 2.2: Process the remaining part
            if attn_metadata.num_prefills > 0:
                initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
                initial_state[~has_initial_state, ...] = 0
                non_spec_chunked_prefill_meta = getattr(
                    attn_metadata,
                    "non_spec_chunked_prefill_meta",
                    None,
                )
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=non_spec_query_start_loc,
                    prebuilt_meta=non_spec_chunked_prefill_meta,
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                )
                # Init cache
                ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)
            elif attn_metadata.num_decodes > 0:
                core_attn_out_non_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out_non_spec, last_recurrent_state = None, None

        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log.contiguous(),
                dt_bias=self.dt_bias.contiguous(),
                q=query_non_spec.contiguous(),
                k=key_non_spec.contiguous(),
                v=value_non_spec.contiguous(),
                a=a.contiguous(),
                b=b.contiguous(),
                initial_state_source=ssm_state,
                initial_state_indices=non_spec_state_indices_tensor,
                cu_seqlens=non_spec_query_start_loc,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
            )

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)[:num_actual_tokens]
        elif spec_sequence_masks is not None:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)[:num_actual_tokens]
        else:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)[:num_actual_tokens]
        maybe_save_kv_layer_to_connector("", [])


class AscendQwen3NextAttention(Qwen3NextAttention):
    def forward(self, positions: torch.Tensor, output: torch.Tensor, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)
        if "qwen3_5" in self.config.model_type:
            cos_sin = self.rotary_emb.cos_sin_cache[positions]
            if cos_sin.device != qkv.device:
                cos_sin = cos_sin.to(qkv.device)
            if cos_sin.dtype != qkv.dtype:
                cos_sin = cos_sin.to(qkv.dtype)

            q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
                qkv=qkv,
                q_weight=1.0 + self.q_norm.weight,
                k_weight=1.0 + self.k_norm.weight,
                cos_sin=cos_sin,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                eps=self.config.rms_norm_eps,
                mrope_section=self.rotary_emb.mrope_section,
                is_interleaved=self.rotary_emb.mrope_interleaved,
                rope_dim=self.rotary_emb.rotary_dim,
                has_gate=self.attn_output_gate,
            )
        else:
            if self.attn_output_gate:
                q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
                orig_shape = q_gate.shape[:-1]
                q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
                q, gate = torch.chunk(q_gate, 2, dim=-1)
                q = q.reshape(*orig_shape, -1)
                gate = gate.reshape(*orig_shape, -1)
            else:
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)

            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)


class AscendQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_idx == 0 and _EXTRA_CTX.flash_comm_v1_enabled:
            tp_size = get_tensor_model_parallel_world_size()
            n_out = (hidden_states.shape[0] + tp_size - 1) // tp_size
            hidden_dim = hidden_states.shape[-1]
            self_attention_output = torch.empty(
                (n_out, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
        else:
            self_attention_output = torch.empty_like(hidden_states)

        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype) + 1)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, {len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype) + 1)

        return hidden_states, residual


Qwen3_5GatedDeltaNet.forward = AscendQwen3_5GatedDeltaNet.forward
Qwen3_5GatedDeltaNet._forward_core = AscendQwen3_5GatedDeltaNet._forward_core
Qwen3NextAttention.forward = AscendQwen3NextAttention.forward
Qwen3_5DecoderLayer.forward = AscendQwen3_5DecoderLayer.forward
