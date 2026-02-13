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
#from collections.abc import Iterable

import torch
import torch_npu
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from einops import rearrange
from torch import nn
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet
from vllm.triton_utils import triton
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import \
    fused_qkvzba_split_reshape_cat
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch

class AscendQwen3Next_GatedDeltaNet(Qwen3NextGatedDeltaNet):

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
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)
        forward_context = get_forward_context()
        is_cuda_graph = forward_context.cudagraph_runtime_mode != CUDAGraphMode.NONE
        # triton grid should be less than 66536
        divide_grid = projected_states_qkvz.shape[0] * triton.cdiv(
            self.num_k_heads, self.tp_size)
        if self.num_v_heads // self.num_k_heads in [1, 2, 4] and \
            is_cuda_graph and divide_grid < 65536:
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                projected_states_qkvz,
                projected_states_ba,
                triton.cdiv(self.num_k_heads, self.tp_size),
                triton.cdiv(self.num_v_heads, self.tp_size),
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                projected_states_qkvz, projected_states_ba)
            query, key, value = map(lambda x: rearrange(x, 'l p d -> l (p d)'),
                                    (query, key, value))
            mixed_qkv = torch.cat((query, key, value), dim=-1)

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
        output[:num_tokens], _ = self.out_proj(core_attn_out)

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
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(
                    0, non_spec_token_indx)
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
                conv_state_indices=spec_state_indices_tensor[:, 0]
                [:attn_metadata.num_spec_decodes],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                conv_weights_T = conv_weights.transpose(0, 1)
                mixed_qkv_non_spec = torch.ops._C_ascend.causal_conv1d_fn(
                    mixed_qkv_non_spec,
                    conv_weights_T,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_state=self_kv_cache[0],
                    has_initial_state=has_initial_state,
                    non_spec_state_indices_tensor=non_spec_state_indices_tensor,
                    non_spec_query_start_loc=non_spec_query_start_loc,
                    pad_slot_id=PAD_SLOT_ID,
                )
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=
                non_spec_state_indices_tensor[:attn_metadata.
                                              num_actual_tokens],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(
            mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec)
        g, beta = fused_gdn_gating_patch(self.A_log, a, b,
                                                 self.dt_bias)
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
            cu_seqlens=spec_query_start_loc[
                    : attn_metadata.num_spec_decodes + 1
                ]
            actual_seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            query_spec = l2norm_fwd(query_spec)
            key_spec = l2norm_fwd(key_spec) 
            core_attn_out_spec = torch_npu.npu_recurrent_gated_delta_rule(
                query=query_spec.squeeze(0),
                key=key_spec.squeeze(0),
                value=value_spec.squeeze(0),
                g=g_spec.squeeze(0),
                beta=beta_spec.squeeze(0),
                state=ssm_state,
                scale=key_spec.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=spec_state_indices_tensor.flatten(),
                num_accepted_tokens=num_accepted_tokens.to(torch.int32)
            ).unsqueeze(0)
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            initial_state = ssm_state[
                non_spec_state_indices_tensor].transpose(-1, -2).contiguous()

            initial_state[~has_initial_state, ...] = 0
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
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            ssm_state[
            non_spec_state_indices_tensor] = last_recurrent_state.transpose(-1,-2).contiguous().to(
                ssm_state.dtype)

        elif attn_metadata.num_decodes > 0:
            cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ]
            actual_seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            query_non_spec = l2norm_fwd(query_non_spec)
            key_non_spec = l2norm_fwd(key_non_spec)
            core_attn_out_non_spec = torch_npu.npu_recurrent_gated_delta_rule(
                query=query_non_spec.squeeze(0),
                key=key_non_spec.squeeze(0),
                value=value_non_spec.squeeze(0),
                g=g_non_spec.squeeze(0),
                beta=beta_non_spec.squeeze(0),
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
            merged_out.index_copy_(1, non_spec_token_indx,
                                core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(
                0)


class AscendQwen3NextAttention(nn.Module):
    def forward(self, positions: torch.Tensor, output: torch.Tensor, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)
        if "qwen3_5" in self.config.model_type:
            if self.attn_output_gate:
                q_gate, kv = qkv.split([self.q_size * 2, self.kv_size * 2], dim=-1)
                orig_shape = q_gate.shape[:-1]
                q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
                q, gate = torch.chunk(q_gate, chunks=2, dim=-1)
                gate = gate.reshape(*orig_shape, -1)
                q = q.reshape(*orig_shape, -1)
                qkv = torch.cat([q, kv], dim=-1)

            cos_sin = self.rotary_emb.cos_sin_cache[positions]
            if cos_sin.device != qkv.device:
                cos_sin = cos_sin.to(qkv.device)
            if cos_sin.dtype != qkv.dtype:
                cos_sin = cos_sin.to(qkv.dtype)

            q, k, v = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
                qkv=qkv,
                q_weight=self.q_norm.weight,
                k_weight=self.k_norm.weight,
                cos_sin=cos_sin,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                eps=self.config.rms_norm_eps,
                mrope_section=self.rotary_emb.mrope_section,
                is_interleaved=self.rotary_emb.mrope_interleaved,
                rope_dim=self.rotary_emb.rotary_dim, )
        else:
            if self.attn_output_gate:
                q_gate, k, v = qkv.split(
                    [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
                )
                orig_shape = q_gate.shape[:-1]
                q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
                q, gate = torch.chunk(q_gate, 2, dim=-1)
                q = q.reshape(*orig_shape, -1)
                gate = gate.reshape(*orig_shape, -1)
            else:
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
                -1, self.num_heads * self.head_dim
            )
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
                -1, self.num_kv_heads * self.head_dim
            )

            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)


Qwen3NextGatedDeltaNet.forward = AscendQwen3Next_GatedDeltaNet.forward
Qwen3NextGatedDeltaNet._forward_core = AscendQwen3Next_GatedDeltaNet._forward_core
Qwen3NextAttention.forward = AscendQwen3NextAttention.forward
