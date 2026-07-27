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
#

import torch
from einops import rearrange
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.triton_utils import triton
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import fused_qkvzba_split_reshape_cat
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.ops.triton.mamba.causal_conv1d import extract_last_width
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.25.1"):
    from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd  # type: ignore[import-not-found]
else:
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd


class AscendGatedDeltaNetAttention(GatedDeltaNetAttention):
    def _split_ba_for_tp(self, ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "split_ba"):
            return self.split_ba(ba)
        return ba.chunk(2, dim=-1)

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
        return

    def _warmup_prefill_kernels_v0202(self, mixed_qkv: torch.Tensor) -> None:
        return

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendGDNAttentionBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor = None,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        if hasattr(self, "in_proj_qkv"):
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = self._split_ba_for_tp(ba)
            b = b.contiguous()
            a = a.contiguous()
        else:
            if not self.gqa_interleaved_layout:
                mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
                num_tokens = mixed_qkvz.size(0)
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                ba, _ = self.in_proj_ba(hidden_states)
                b, a = self._split_ba_for_tp(ba)

                b = b.contiguous()
                a = a.contiguous()
            else:
                projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
                projected_states_ba, _ = self.in_proj_ba(hidden_states)
                num_tokens = projected_states_qkvz.size(0)

                mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                    projected_states_qkvz,
                    projected_states_ba,
                    triton.cdiv(self.num_k_heads, self.tp_size),
                    triton.cdiv(self.num_v_heads, self.tp_size),
                    self.head_k_dim,
                    self.head_v_dim,
                )

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

        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
            False,
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
        out, _ = self.out_proj(core_attn_out)
        if output is not None:
            output[:num_tokens] = out
        return out

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation (called by custom op).
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

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
            conv_weights_T = conv_weights.transpose(0, 1)
            activation_num = 1 if self.activation else 0
            spec_causal_conv1d_meta = attn_metadata.spec_decode_metadata.spec_causal_conv1d
            spec_query_start_loc_device = spec_causal_conv1d_meta.query_start_loc
            output_spec = torch.empty_like(mixed_qkv_spec)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                output_spec,
                mixed_qkv_spec,
                conv_weights_T,
                conv_state=self_kv_cache[0],
                bias_opt=self.conv1d.bias,
                query_start_loc_opt=spec_query_start_loc_device,
                cache_indices_opt=spec_causal_conv1d_meta.cache_indices,
                initial_state_mode_opt=None,
                num_accepted_tokens_opt=spec_causal_conv1d_meta.num_accepted_tokens,
                activation_mode=activation_num,
                pad_slot_id=PAD_SLOT_ID,
                run_mode=1,
            )
            mixed_qkv_spec = output_spec

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                non_spec_causal_conv1d_meta = attn_metadata.non_spec_prefill_metadata.causal_conv1d
                query_start_loc_opt = non_spec_causal_conv1d_meta.query_start_loc
                cache_indices_opt = non_spec_causal_conv1d_meta.cache_indices
                initial_state_mode_opt = non_spec_causal_conv1d_meta.initial_state_mode
                if get_pcp_group().world_size > 1:
                    conv_weights_T = conv_weights.transpose(0, 1)
                    activation_num = 1 if self.activation else 0
                    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
                    assert non_spec_query_start_loc is not None
                    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
                    width = conv_weights.shape[1]
                    state_len = width - 1
                    num_seqs = non_spec_query_start_loc.shape[0] - 1
                    prefill_seq_offset = max(0, num_seqs - attn_metadata.num_prefills)
                    prefill_cache_indices = non_spec_state_indices_tensor[prefill_seq_offset:]
                    mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
                    last_width_prefill_x = extract_last_width(
                        mixed_qkv_non_spec_T, non_spec_query_start_loc[prefill_seq_offset:], state_len
                    )
                    pcp_rank = get_pcp_group().rank_in_group
                    all_last_width_prefill_x = get_pcp_group().all_gather(
                        last_width_prefill_x.unsqueeze(0).contiguous(), 0
                    )
                    if pcp_rank > 0 and prefill_cache_indices.shape[0] > 0:
                        self_kv_cache[0][prefill_cache_indices, :state_len, :] = all_last_width_prefill_x[
                            pcp_rank - 1, ...
                        ].transpose(-1, -2)
                    mixed_qkv_non_spec_output = torch.empty_like(mixed_qkv_non_spec)
                    torch.ops._C_ascend.npu_causal_conv1d_custom(
                        mixed_qkv_non_spec_output,
                        mixed_qkv_non_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=query_start_loc_opt,
                        cache_indices_opt=cache_indices_opt,
                        initial_state_mode_opt=initial_state_mode_opt,
                        num_accepted_tokens_opt=None,
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=0,
                    )
                    mixed_qkv_non_spec = mixed_qkv_non_spec_output
                    if prefill_cache_indices.shape[0] > 0:
                        self_kv_cache[0][prefill_cache_indices, :state_len, :] = all_last_width_prefill_x[
                            -1, ...
                        ].transpose(-1, -2)
                else:
                    conv_weights_T = conv_weights.transpose(0, 1)
                    activation_num = 1 if self.activation else 0
                    mixed_qkv_non_spec_output = torch.empty_like(mixed_qkv_non_spec)
                    torch.ops._C_ascend.npu_causal_conv1d_custom(
                        mixed_qkv_non_spec_output,
                        mixed_qkv_non_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=query_start_loc_opt,
                        cache_indices_opt=cache_indices_opt,
                        initial_state_mode_opt=initial_state_mode_opt,
                        num_accepted_tokens_opt=None,
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=0,
                    )
                    mixed_qkv_non_spec = mixed_qkv_non_spec_output
        elif attn_metadata.num_decodes > 0:
            conv_weights_T = conv_weights.transpose(0, 1)
            activation_num = 1 if self.activation else 0
            non_spec_causal_conv1d_meta = attn_metadata.non_spec_decode_metadata.causal_conv1d
            non_spec_query_start_loc_device = non_spec_causal_conv1d_meta.query_start_loc
            output_non_spec = torch.empty_like(mixed_qkv_non_spec)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                output_non_spec,
                mixed_qkv_non_spec,
                conv_weights_T,
                conv_state=self_kv_cache[0],
                bias_opt=self.conv1d.bias,
                query_start_loc_opt=non_spec_query_start_loc_device,
                cache_indices_opt=non_spec_causal_conv1d_meta.cache_indices,
                initial_state_mode_opt=None,
                num_accepted_tokens_opt=None,
                activation_mode=activation_num,
                pad_slot_id=PAD_SLOT_ID,
                run_mode=1,
            )
            mixed_qkv_non_spec = output_non_spec
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

        # 2. Recurrent attention
        g, beta = DeviceOperator.fused_gdn_gating(self.A_log, a, b, self.dt_bias)
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

        split_non_spec = (
            spec_sequence_masks is None and attn_metadata.num_prefills > 0 and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            actual_seq_lengths = attn_metadata.spec_decode_metadata.actual_seq_lengths
            query_spec = l2norm_fwd(query_spec)
            key_spec = l2norm_fwd(key_spec)
            # Dispatches to the vllm-ascend AscendC custom operator
            # (csrc/recurrent_gated_delta_rule), NOT the built-in CANN operator.
            # The custom op extends dtype support (e.g. float32 state) and is
            # loaded at runtime via ASCEND_CUSTOM_OPP_PATH.
            core_attn_out_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=query_spec.squeeze(0),
                key=key_spec.squeeze(0),
                value=value_spec.squeeze(0),
                g=g_spec.squeeze(0),
                beta=beta_spec.squeeze(0),
                state=ssm_state,
                scale=key_spec.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=spec_state_indices_tensor.flatten(),
                num_accepted_tokens=spec_causal_conv1d_meta.num_accepted_tokens.to(torch.int32),
            ).unsqueeze(0)
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process non-spec-decode part in mixed non-spec batches
        if split_non_spec:
            assert mixed_qkv_non_spec is not None
            assert g_non_spec is not None
            assert beta_non_spec is not None
            query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(mixed_qkv_non_spec[:num_decode_tokens])
            actual_seq_lengths = attn_metadata.non_spec_decode_metadata.actual_seq_lengths
            query_decode = l2norm_fwd(query_decode)
            key_decode = l2norm_fwd(key_decode)
            core_attn_out_decode = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=query_decode.squeeze(0),
                key=key_decode.squeeze(0),
                value=value_decode.squeeze(0),
                g=g_non_spec[:, :num_decode_tokens].squeeze(0),
                beta=beta_non_spec[:, :num_decode_tokens].squeeze(0),
                state=ssm_state,
                scale=key_decode.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=non_spec_state_indices_tensor[: attn_metadata.num_decodes],
            ).unsqueeze(0)
        else:
            core_attn_out_decode = None

        # 2.3: Process the remaining part
        if attn_metadata.num_prefills > 0:
            prefill_query_start_loc = attn_metadata.prefill_query_start_loc
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_query_start_loc is not None
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            assert g_non_spec is not None
            assert beta_non_spec is not None
            if split_non_spec:
                query_non_spec = query_non_spec[:, num_decode_tokens:]
                key_non_spec = key_non_spec[:, num_decode_tokens:]
                value_non_spec = value_non_spec[:, num_decode_tokens:]
                g_non_spec = g_non_spec[:, num_decode_tokens:]
                beta_non_spec = beta_non_spec[:, num_decode_tokens:]

            initial_state = ssm_state[prefill_state_indices].transpose(-1, -2).contiguous()
            clear_ssm_states(initial_state, prefill_has_initial_state)
            (core_attn_out_non_spec, last_recurrent_state) = chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=prefill_query_start_loc,
                prebuilt_meta=attn_metadata.non_spec_prefill_metadata.chunk,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            ssm_state[prefill_state_indices] = last_recurrent_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
            if split_non_spec:
                core_attn_out_non_spec = torch.cat(
                    [core_attn_out_decode, core_attn_out_non_spec],
                    dim=1,
                )
        elif attn_metadata.num_decodes > 0:
            actual_seq_lengths = attn_metadata.non_spec_decode_metadata.actual_seq_lengths
            query_non_spec = l2norm_fwd(query_non_spec)
            key_non_spec = l2norm_fwd(key_non_spec)
            # Dispatches to the vllm-ascend AscendC custom operator
            # (csrc/recurrent_gated_delta_rule), NOT the built-in CANN operator.
            core_attn_out_non_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=query_non_spec.squeeze(0),
                key=key_non_spec.squeeze(0),
                value=value_non_spec.squeeze(0),
                g=g_non_spec.squeeze(0) if g_non_spec is not None else g_non_spec,
                beta=beta_non_spec.squeeze(0) if beta_non_spec is not None else beta_non_spec,
                state=ssm_state,
                scale=key_non_spec.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=non_spec_state_indices_tensor,
            ).unsqueeze(0)
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
        maybe_save_kv_layer_to_connector("", [])
