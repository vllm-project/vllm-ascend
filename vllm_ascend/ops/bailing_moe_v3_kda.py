# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ling's three convolution projections share one paged KDA state."""

import torch
from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3KimiDeltaAttention
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from vllm_ascend.attention.gdn_attn_builder import AscendGDNHostMetadataBackend
from vllm_ascend.ops.kda import run_chunk_kda, run_recurrent_kda
from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_


@PluggableLayer.register_oot(name="BailingMoeV3KimiDeltaAttention")
class AscendBailingMoeV3KimiDeltaAttention(BailingMoeV3KimiDeltaAttention):
    def get_attn_backend(self):
        return AscendGDNHostMetadataBackend

    def forward(self, hidden_states, positions, output):
        width = self.projection_size_per_partition
        if self.separate_b_proj:
            q, k, v = self.qkv_proj(hidden_states)[0].split(width, dim=-1)
            beta = self.b_proj(hidden_states)[0]
        else:
            q, k, v, beta = self.qkvb_proj(hidden_states)[0].split((width, width, width, self.local_num_heads), dim=-1)
        # AscendC applies dt_bias and A_log to the raw gate inside KDA.
        shape = (1, hidden_states.shape[0], self.local_num_heads, self.head_dim)
        gate = self.f_proj(hidden_states)[0].reshape(shape)
        core = hidden_states.new_zeros(shape)
        torch.ops.vllm.bailing_v3_kda_attention(q, k, v, gate, beta.float().sigmoid().unsqueeze(0), core, self.prefix)
        norm_gate = self.g_proj(hidden_states)[0].reshape(shape[1:])
        output[:] = self.o_proj(self.o_norm(core, norm_gate).reshape(hidden_states.shape[0], width))[0]

    def _convolve(self, projections, meta):
        state = self.kv_cache[0]
        indices = meta.non_spec_state_indices_tensor
        starts = meta.non_spec_query_start_loc
        active = (indices > NULL_BLOCK_ID) & (indices < state.shape[0]) & (starts[1:] > starts[:-1])
        keep = active if not meta.num_prefills else active & meta.has_initial_state
        mixed = torch.cat(projections, dim=-1)
        # The native convolution needs dense activation-dtype SD rows. A cache
        # row can be FP16 and its outer stride includes the other hybrid state.
        rows = gather_ssm_states(state, indices, keep, output_dtype=mixed.dtype)
        if is_conv_state_dim_first():
            rows = rows.transpose(1, 2).contiguous()
        compact = torch.where(
            active, torch.arange(indices.numel(), device=indices.device, dtype=indices.dtype), PAD_SLOT_ID
        )
        convs = (self.q_conv1d, self.k_conv1d, self.v_conv1d)
        weights = torch.cat([c.weight.squeeze(1) for c in convs]).T.contiguous().to(mixed.dtype)
        bias = torch.cat([c.bias for c in convs]).to(mixed.dtype) if convs[0].bias is not None else None
        if meta.num_prefills:
            result = causal_conv1d_fn(
                mixed,
                weights,
                bias,
                conv_states=rows,
                query_start_loc=starts,
                cache_indices=compact,
                has_initial_state=keep,
                activation="silu",
                pad_slot_id=PAD_SLOT_ID,
                null_block_id=PAD_SLOT_ID,
            )
        else:
            result = causal_conv1d_update(
                mixed,
                rows,
                weights,
                bias=bias,
                activation="silu",
                conv_state_indices=compact,
                query_start_loc=starts,
                max_query_len=1,
                null_block_id=PAD_SLOT_ID,
                out=torch.empty_like(mixed),
            )
        if is_conv_state_dim_first():
            rows = rows.transpose(1, 2).contiguous()
        scatter_ssm_states_(state, indices, rows, active_rows=active)
        return [x.reshape(1, mixed.shape[0], self.local_num_heads, self.head_dim) for x in result.chunk(3, dim=-1)]

    def _forward(self, q_proj_states, k_proj_states, v_proj_states, g1, beta, core_attn_out):
        metadata = get_forward_context().attn_metadata
        if metadata is None or (meta := metadata.get(self.prefix)) is None:
            return
        if not meta.num_prefills and not meta.num_decodes:
            return
        count = meta.num_actual_tokens
        state = self.kv_cache[1]
        gate = g1
        q, k, v = self._convolve((q_proj_states[:count], k_proj_states[:count], v_proj_states[:count]), meta)
        bound = self.lower_bound if self.safe_gate else None
        # The metadata builder puts one-token decodes before prefill requests.
        split = meta.num_decode_tokens if meta.num_prefills else count
        if meta.num_decodes:
            rows = meta.num_decodes if meta.num_prefills else meta.non_spec_state_indices_tensor.numel()
            core_attn_out[:, :split] = run_recurrent_kda(
                q[:, :split],
                k[:, :split],
                v[:, :split],
                gate[:, :split],
                beta[:, :split],
                state,
                meta.non_spec_query_start_loc[: rows + 1],
                meta.non_spec_state_indices_tensor[:rows],
                self.A_log,
                self.dt_bias,
                lower_bound=bound,
            )
        if meta.num_prefills:
            chunk = meta.non_spec_prefill_metadata.chunk
            indices, flags = meta.prefill_state_indices, meta.prefill_has_initial_state
            if chunk.keep_meta is not None:
                indices, flags = indices[chunk.keep_meta], flags[chunk.keep_meta]
            active = (indices > NULL_BLOCK_ID) & (indices < state.shape[0])
            initial = gather_ssm_states(state, indices, flags & active)
            out, final = run_chunk_kda(
                q[:, split:],
                k[:, split:],
                v[:, split:],
                gate[:, split:count],
                beta[:, split:count],
                initial,
                chunk.cu_seqlens_host if chunk.cu_seqlens_kern is None else chunk.cu_seqlens_kern,
                chunk.chunk_indices_chunk64_host,
                self.A_log,
                self.dt_bias,
                lower_bound=bound,
                use_qk_l2norm_in_kernel=True,
            )
            scatter_ssm_states_(state, indices, final, active_rows=active)
            core_attn_out[:, split:count] = out
