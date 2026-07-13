#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import os
from einops import rearrange
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3KimiDeltaAttention
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_update_npu
if os.environ.get("ENABLE_PYPTO") == "1":
    from vllm_ascend.ops.pypto.kda.chunk_kda_impl import chunk_kda_wrapper as chunk_kda
    from vllm_ascend.ops.pypto.kda.fused_recurrent_kda_impl import fused_recurrent_kda
else:
    from vllm_ascend.ops.triton.kda.kda import chunk_kda, fused_recurrent_kda


class AscendBailingMoeV3KimiDeltaAttention(BailingMoeV3KimiDeltaAttention):

    def _forward(
        self,
        q_proj_states,
        k_proj_states,
        v_proj_states,
        g1,
        beta,
        core_attn_out,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        if attn_metadata is None:
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
        spec_state_indices = attn_metadata.spec_state_indices_tensor
        non_spec_state_indices = attn_metadata.non_spec_state_indices_tensor
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        num_actual_tokens = attn_metadata.num_actual_tokens
        conv_state_q, conv_state_k, conv_state_v, recurrent_state = (
            self.kv_cache
        )
        recurrent_state_active = recurrent_state[..., : self.head_dim]
        conv_state_q = conv_state_q.transpose(-1, -2)
        conv_state_k = conv_state_k.transpose(-1, -2)
        conv_state_v = conv_state_v.transpose(-1, -2)

        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        q_conv_weights = self.q_conv1d.weight.view(
            self.q_conv1d.weight.size(0), self.q_conv1d.weight.size(2)
        )
        k_conv_weights = self.k_conv1d.weight.view(
            self.k_conv1d.weight.size(0), self.k_conv1d.weight.size(2)
        )
        v_conv_weights = self.v_conv1d.weight.view(
            self.v_conv1d.weight.size(0), self.v_conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            assert spec_query_start_loc is not None
            assert spec_state_indices is not None
            assert num_accepted_tokens is not None
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                q_proj_states_spec = q_proj_states
                k_proj_states_spec = k_proj_states
                v_proj_states_spec = v_proj_states
                g1_spec = g1
                beta_spec = beta
                q_proj_states_non_spec = None
                k_proj_states_non_spec = None
                v_proj_states_non_spec = None
                g1_non_spec = None
                beta_non_spec = None
            else:
                assert spec_token_indx is not None
                assert non_spec_token_indx is not None
                q_proj_states_spec = q_proj_states.index_select(0, spec_token_indx)
                k_proj_states_spec = k_proj_states.index_select(0, spec_token_indx)
                v_proj_states_spec = v_proj_states.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                q_proj_states_non_spec = q_proj_states.index_select(
                    0, non_spec_token_indx
                )
                k_proj_states_non_spec = k_proj_states.index_select(
                    0, non_spec_token_indx
                )
                v_proj_states_non_spec = v_proj_states.index_select(
                    0, non_spec_token_indx
                )
                g1_non_spec = g1.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            q_proj_states_spec = None
            k_proj_states_spec = None
            v_proj_states_spec = None
            g1_spec = None
            beta_spec = None
            q_proj_states_non_spec = q_proj_states
            k_proj_states_non_spec = k_proj_states
            v_proj_states_non_spec = v_proj_states
            g1_non_spec = g1
            beta_non_spec = beta

        def _causal_conv1d_spec(
            x: torch.Tensor,
            conv_state: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None,
        ) -> torch.Tensor:
            assert spec_query_start_loc is not None
            assert spec_state_indices is not None
            assert num_accepted_tokens is not None
            return causal_conv1d_update_npu(
                x,
                conv_state,
                weight,
                bias,
                activation="silu",
                conv_state_indices=spec_state_indices[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices.size(-1),
                validate_data=False,
            )

        if spec_sequence_masks is not None:
            assert q_proj_states_spec is not None
            assert k_proj_states_spec is not None
            assert v_proj_states_spec is not None
            q_spec = _causal_conv1d_spec(
                q_proj_states_spec,
                conv_state_q,
                q_conv_weights,
                self.q_conv1d.bias,
            )
            k_spec = _causal_conv1d_spec(
                k_proj_states_spec,
                conv_state_k,
                k_conv_weights,
                self.k_conv1d.bias,
            )
            v_spec = _causal_conv1d_spec(
                v_proj_states_spec,
                conv_state_v,
                v_conv_weights,
                self.v_conv1d.bias,
            )
        else:
            q_spec = None
            k_spec = None
            v_spec = None

        if attn_metadata.num_prefills > 0:
            assert q_proj_states_non_spec is not None
            assert k_proj_states_non_spec is not None
            assert v_proj_states_non_spec is not None
            q = causal_conv1d_fn(
                q_proj_states_non_spec.transpose(0, 1),
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_states=conv_state_q,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices,
                query_start_loc=non_spec_query_start_loc,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k_proj_states_non_spec.transpose(0, 1),
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_states=conv_state_k,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices,
                query_start_loc=non_spec_query_start_loc,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v_proj_states_non_spec.transpose(0, 1),
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_states=conv_state_v,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices,
                query_start_loc=non_spec_query_start_loc,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert q_proj_states_non_spec is not None
            assert k_proj_states_non_spec is not None
            assert v_proj_states_non_spec is not None
            assert non_spec_state_indices is not None
            decode_indices = non_spec_state_indices[:num_actual_tokens]
            q = causal_conv1d_update(
                q_proj_states_non_spec,
                conv_state_q,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_indices,
                validate_data=True,
            )
            k = causal_conv1d_update(
                k_proj_states_non_spec,
                conv_state_k,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_indices,
                validate_data=True,
            )
            v = causal_conv1d_update(
                v_proj_states_non_spec,
                conv_state_v,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_indices,
                validate_data=True,
            )
        else:
            q = None
            k = None
            v = None

        if q_spec is not None:
            q_spec, k_spec, v_spec = map(
                lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim),
                (q_spec, k_spec, v_spec),
            )
        if q is not None:
            q, k, v = map(
                lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim),
                (q, k, v),
            )

        if spec_sequence_masks is not None:
            assert q_spec is not None
            assert k_spec is not None
            assert v_spec is not None
            assert g1_spec is not None
            assert beta_spec is not None
            assert spec_query_start_loc is not None
            assert spec_state_indices is not None
            assert num_accepted_tokens is not None
            out_spec, _ = fused_recurrent_kda(
                q=q_spec,
                k=k_spec,
                v=v_spec,
                g=g1_spec,
                beta=beta_spec,
                initial_state=recurrent_state_active,
                use_qk_l2norm_in_kernel=True,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=spec_query_start_loc[
                    : attn_metadata.num_spec_decodes + 1
                ],
                ssm_state_indices=spec_state_indices,
                num_accepted_tokens=num_accepted_tokens,
            )
        else:
            out_spec = None

        if attn_metadata.num_prefills > 0:
            assert q is not None
            assert k is not None
            assert v is not None
            assert g1_non_spec is not None
            assert beta_non_spec is not None
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices is not None
            zero_idx = non_spec_state_indices[~has_initial_state]
            recurrent_state[zero_idx] = 0
            initial_state = recurrent_state_active[non_spec_state_indices].contiguous()
            out, last_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g1_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=non_spec_query_start_loc,
            )
            recurrent_state_active[non_spec_state_indices] = last_state
        elif attn_metadata.num_decodes > 0:
            assert q is not None
            assert k is not None
            assert v is not None
            assert g1_non_spec is not None
            assert beta_non_spec is not None
            assert non_spec_query_start_loc is not None
            out, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g1_non_spec,
                beta=beta_non_spec,
                initial_state=recurrent_state_active,
                use_qk_l2norm_in_kernel=True,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices,
            )
        else:
            out = None

        if spec_sequence_masks is not None and out is not None:
            assert out_spec is not None
            assert spec_token_indx is not None
            assert non_spec_token_indx is not None
            merged_out = torch.empty(
                (1, num_actual_tokens, *out_spec.shape[2:]),
                dtype=out.dtype,
                device=out.device,
            )
            merged_out.index_copy_(1, spec_token_indx, out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, out)
            core_attn_out[0, :num_actual_tokens] = merged_out[0, :num_actual_tokens]
        elif spec_sequence_masks is not None:
            assert out_spec is not None
            core_attn_out[0, :num_actual_tokens] = out_spec[0, :num_actual_tokens]
        else:
            assert out is not None
            core_attn_out[0, :num_actual_tokens] = out[0, :num_actual_tokens]