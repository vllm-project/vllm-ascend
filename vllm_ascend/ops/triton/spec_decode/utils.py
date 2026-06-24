# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/utils.py

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata

BLOCK_HIDDEN = 64
BLOCK_TOKENS = 64


@triton.jit(do_not_specialize=["num_reqs"])
def prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_rejected_tokens_gpu_ptr,
    num_reqs,  # tl.int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # Grid-Stride Loop:
    block_start_step = num_programs * BLOCK_SIZE

    for block_start in tl.range(pid * BLOCK_SIZE, num_reqs, block_start_step):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_reqs

        # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
        # cumulative sum (first entry is the first value, not zero).
        cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + offsets, mask=mask)

        prev_indices = offsets - 1
        has_prev = offsets > 0
        cu_draft_prev = tl.load(
            cu_num_draft_tokens_ptr + prev_indices,
            mask=mask & has_prev,
            other=0,
        )

        num_draft_tokens = tl.where(has_prev, cu_draft_curr - cu_draft_prev, cu_draft_curr)

        valid_count = tl.load(valid_sampled_tokens_count_ptr + offsets, mask=mask)
        num_rejected = num_draft_tokens + 1 - valid_count
        num_rejected = tl.where(num_draft_tokens > 0, num_rejected, 0)

        # query_start_loc[req_idx + 1] is the start position of the next request,
        # which is one past the last token of this request.
        q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + offsets + 1, mask=mask) - 1

        index_to_sample = q_last_tok_idx - num_rejected
        tl.store(token_indices_to_sample_ptr + offsets, index_to_sample, mask=mask)
        tl.store(num_rejected_tokens_gpu_ptr + offsets, num_rejected, mask=mask)


@triton.jit
def copy_and_expand_dflash_inputs_kernel_single_grid(
    # Inputs
    next_token_ids_ptr,  # [num_reqs]
    target_positions_ptr,  # [num_context]
    context_slot_mapping_ptr,  # [num_context]
    # Outputs
    out_input_ids_ptr,  # [num_query_total] (output)
    out_context_positions_ptr,  # [num_context] (output)
    out_query_positions_ptr,  # [num_query_total] (output)
    out_context_slot_mapping_ptr,  # [num_context] (output)
    out_query_slot_mapping_ptr,  # [num_query_total] (output)
    out_token_indices_ptr,  # [num_reqs * num_speculative_tokens] (output)
    # Block table
    block_table_ptr,  # [max_reqs, max_blocks]
    block_table_stride,  # stride of block_table dim 0 (in elements)
    # Metadata
    query_start_loc_ptr,  # [num_reqs + 1]
    seq_lens_ptr,  # [num_reqs]
    num_rejected_tokens_ptr,  # [num_reqs] or null (0) when not padded
    # Scalars
    parallel_drafting_token_id,  # tl.int32
    block_size,  # tl.int32
    num_query_per_req,  # tl.int32
    num_speculative_tokens,  # tl.int32
    total_input_tokens,  # tl.int32
    batch_size,  # tl.int32
    HAS_NUM_REJECTED: tl.constexpr = False,
):
    for req_idx in range(0, batch_size):
        ctx_start = tl.load(query_start_loc_ptr + req_idx)
        ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
        num_ctx = ctx_end - ctx_start

        for j in range(0, num_ctx):
            ctx_pos_idx = ctx_start + j
            pos = tl.load(target_positions_ptr + ctx_pos_idx)
            tl.store(out_context_positions_ptr + ctx_pos_idx, pos)

            slot = tl.load(context_slot_mapping_ptr + ctx_pos_idx)
            tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, slot)

        if HAS_NUM_REJECTED:
            num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
            valid_ctx_end = ctx_end - num_rejected
        else:
            num_rejected = 0
            valid_ctx_end = ctx_end

        seq_len = tl.load(seq_lens_ptr + req_idx)
        effective_seq_len = seq_len - num_rejected
        last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)

        for q_idx in range(0, num_query_per_req):
            query_pos = last_pos + 1 + q_idx
            query_out_idx = req_idx * num_query_per_req + q_idx

            tl.store(out_query_positions_ptr + query_out_idx, query_pos)

            query_cache_pos = effective_seq_len + q_idx
            block_num_q = query_cache_pos // block_size
            block_id_q = tl.load(block_table_ptr + req_idx * block_table_stride + block_num_q).to(tl.int64)
            slot_q = block_id_q * block_size + (query_cache_pos % block_size)
            tl.store(out_query_slot_mapping_ptr + query_out_idx, slot_q)

            if q_idx == 0:
                bonus_token = tl.load(next_token_ids_ptr + req_idx)
                tl.store(out_input_ids_ptr + query_out_idx, bonus_token)
            else:
                tl.store(out_input_ids_ptr + query_out_idx, parallel_drafting_token_id)

                sample_out_idx = req_idx * num_speculative_tokens + (q_idx - 1)
                tl.store(out_token_indices_ptr + sample_out_idx, query_out_idx)


def _multi_layer_eagle_shift_and_cache(
    *,
    batch_size: int,
    max_shift: int,
    src_token_ids: torch.Tensor,
    dst_token_ids: torch.Tensor,
    src_positions: torch.Tensor,
    dst_positions: torch.Tensor,
    src_hidden_states: torch.Tensor,
    dst_hidden_states: torch.Tensor,
    src_slot_mapping: torch.Tensor,
    dst_slot_mapping: torch.Tensor,
    start_token_indices: torch.Tensor,
    end_token_indices: torch.Tensor,
    token_indices_to_sample: torch.Tensor,
    shift: torch.Tensor,
    cached_lens: torch.Tensor,
    cached_prev_token_ids: torch.Tensor,
    cached_prev_positions: torch.Tensor,
    cached_prev_hidden_states: torch.Tensor,
    cached_slot_mappings: torch.Tensor,
    common_attn_metadata: CommonAttentionMetadata,
):
    import torch

    if batch_size == 0:
        return

    assert max_shift > 0

    assert cached_prev_positions.is_contiguous()
    assert cached_prev_token_ids.is_contiguous()
    assert cached_prev_hidden_states.is_contiguous()
    assert cached_slot_mappings.is_contiguous()
    assert src_hidden_states.is_contiguous()
    assert dst_hidden_states.is_contiguous()

    if src_slot_mapping.data_ptr() == dst_slot_mapping.data_ptr():
        src_slot_mapping = src_slot_mapping.clone()

    store_start = torch.maximum(
        start_token_indices,
        (token_indices_to_sample + 1 - max_shift),
    )
    store_lens = torch.clamp(
        token_indices_to_sample - store_start + 1,
        min=0,
        max=max_shift,
    )

    # Triton kernel path: parallelized shift+gather
    num_reqs = start_token_indices.shape[0]
    max_window_len = int(
        (common_attn_metadata.query_start_loc_cpu[1:] - common_attn_metadata.query_start_loc_cpu[:-1]).max().item()
    )
    num_blocks = max(1, (max_window_len + BLOCK_TOKENS - 1) // BLOCK_TOKENS)

    _shift_and_gather_cache_1d_kernel[(num_reqs, num_blocks)](
        src_token_ids,
        dst_token_ids,
        cached_prev_token_ids,
        start_token_indices,
        end_token_indices,
        shift,
        cached_lens,
        store_start,
        store_lens,
        MAX_SHIFT=max_shift,
        PADDED_SHIFT=triton.next_power_of_2(max_shift),
        BLOCK_TOKENS=BLOCK_TOKENS,
    )

    _shift_and_gather_cache_1d_kernel[(num_reqs, num_blocks)](
        src_slot_mapping,
        dst_slot_mapping,
        cached_slot_mappings,
        start_token_indices,
        end_token_indices,
        shift,
        cached_lens,
        store_start,
        store_lens,
        MAX_SHIFT=max_shift,
        PADDED_SHIFT=triton.next_power_of_2(max_shift),
        BLOCK_TOKENS=BLOCK_TOKENS,
    )

    _shift_and_gather_cache_1d_kernel[(num_reqs, num_blocks)](
        src_positions,
        dst_positions,
        cached_prev_positions,
        start_token_indices,
        end_token_indices,
        shift,
        cached_lens,
        store_start,
        store_lens,
        MAX_SHIFT=max_shift,
        PADDED_SHIFT=triton.next_power_of_2(max_shift),
        BLOCK_TOKENS=BLOCK_TOKENS,
    )

    hidden_size = int(dst_hidden_states.shape[1])
    num_hidden_blocks = max(1, (hidden_size + BLOCK_HIDDEN - 1) // BLOCK_HIDDEN)

    _shift_and_gather_hidden_kernel[(num_reqs, num_blocks, num_hidden_blocks)](
        src_hidden_states,
        dst_hidden_states,
        cached_prev_hidden_states,
        start_token_indices,
        end_token_indices,
        shift,
        cached_lens,
        store_start,
        store_lens,
        MAX_SHIFT=max_shift,
        PADDED_SHIFT=triton.next_power_of_2(max_shift),
        HIDDEN_SIZE=hidden_size,
        BLOCK_TOKENS=BLOCK_TOKENS,
        BLOCK_HIDDEN=BLOCK_HIDDEN,
        num_warps=4,
    )

    cached_lens.copy_(store_lens)
    return


@triton.jit
def _shift_and_gather_cache_1d_kernel(
    src_ptr,
    dst_ptr,
    cached_ptr,
    start_ptr,
    end_ptr,
    shift_ptr,
    cached_len_ptr,
    store_start_ptr,
    store_len_ptr,
    MAX_SHIFT: tl.constexpr,
    PADDED_SHIFT: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_blk = tl.program_id(1)

    start = tl.load(start_ptr + pid_seq).to(tl.int32)
    end = tl.load(end_ptr + pid_seq).to(tl.int32)
    shift = tl.load(shift_ptr + pid_seq).to(tl.int32)
    cached_len = tl.load(cached_len_ptr + pid_seq).to(tl.int32)

    # get dst indices
    base = pid_blk * BLOCK_TOKENS
    k = tl.arange(0, BLOCK_TOKENS)
    offs = base + k
    dst_idx = start + offs

    # get dst mask
    window_len = end - start + 1
    mask = offs < window_len

    # load from cached
    base_cached = cached_ptr + pid_seq * MAX_SHIFT
    cached_idx = cached_len - shift + offs
    cached_mask = offs < shift
    safe_cached_idx = tl.where(cached_mask, cached_idx, 0)
    val_cached = tl.load(base_cached + safe_cached_idx, mask=mask & cached_mask, other=0)

    # load from src
    src_idx = start + offs - shift
    safe_src_idx = tl.where(src_idx >= 0, src_idx, 0)
    val_src = tl.load(src_ptr + safe_src_idx, mask=mask & ~cached_mask, other=0)

    # store to dst
    val = tl.where(cached_mask, val_cached, val_src)
    tl.store(dst_ptr + dst_idx, val, mask=mask)

    # Store into the per-sequence cache.
    store_start = tl.load(store_start_ptr + pid_seq).to(tl.int32)
    store_len = tl.load(store_len_ptr + pid_seq).to(tl.int32)
    m = tl.arange(0, PADDED_SHIFT)
    store_mask = m < MAX_SHIFT
    dst_idx = store_start + m
    safe_dst_idx = tl.where(store_mask & (m < store_len), dst_idx, 0)
    safe_m = tl.where(store_mask, m, 0)
    val = tl.load(dst_ptr + safe_dst_idx, mask=store_mask & (m < store_len), other=0)
    tl.store(base_cached + safe_m, val, mask=store_mask)


@triton.jit
def _shift_and_gather_hidden_kernel(
    src_ptr,
    dst_ptr,
    cached_ptr,
    start_ptr,
    end_ptr,
    shift_ptr,
    cached_len_ptr,
    store_start_ptr,
    store_len_ptr,
    MAX_SHIFT: tl.constexpr,
    PADDED_SHIFT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_blk = tl.program_id(1)
    pid_hid = tl.program_id(2)

    start = tl.load(start_ptr + pid_seq).to(tl.int32)
    end = tl.load(end_ptr + pid_seq).to(tl.int32)
    shift = tl.load(shift_ptr + pid_seq).to(tl.int32)
    cached_len = tl.load(cached_len_ptr + pid_seq).to(tl.int32)

    # get dst indices
    base = pid_blk * BLOCK_TOKENS
    k = tl.arange(0, BLOCK_TOKENS)
    tok_offs = base + k
    dst_tok = start + tok_offs
    n = pid_hid * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    dst_ptrs = dst_ptr + dst_tok[:, None] * HIDDEN_SIZE + n[None, :] * 1

    # get dst mask
    window_len = end - start + 1
    tok_mask = tok_offs < window_len
    n_mask = n < HIDDEN_SIZE
    mask = tok_mask[:, None] & n_mask[None, :]

    # load from cached
    base_cached = cached_ptr + pid_seq * HIDDEN_SIZE * MAX_SHIFT
    cached_tok = cached_len - shift + tok_offs
    safe_cached_tok = tl.where(tok_offs < shift, cached_tok, 0)
    cached_ptrs = base_cached + safe_cached_tok[:, None] * HIDDEN_SIZE + n[None, :] * 1
    cached_mask = tok_offs < shift
    val_cached = tl.load(cached_ptrs, mask=mask & cached_mask[:, None], other=0)

    # load from src
    src_tok = start + tok_offs - shift
    safe_src_tok = tl.where(src_tok >= 0, src_tok, 0)
    src_ptrs = src_ptr + safe_src_tok[:, None] * HIDDEN_SIZE + n[None, :] * 1
    val_src = tl.load(src_ptrs, mask=mask & ~cached_mask[:, None], other=0)

    # store to dst
    val = tl.where(cached_mask[:, None], val_cached, val_src)
    tl.store(dst_ptrs, val, mask=mask)

    # store to cached
    store_start = tl.load(store_start_ptr + pid_seq).to(tl.int32)
    store_len = tl.load(store_len_ptr + pid_seq).to(tl.int32)
    m = tl.arange(0, PADDED_SHIFT)
    m_mask = (m < MAX_SHIFT) & (m < store_len)
    store_tok = store_start + m
    safe_store_tok = tl.where(m_mask, store_tok, 0)
    safe_m = tl.where(m < MAX_SHIFT, m, 0)
    dst_ptrs = dst_ptr + safe_store_tok[:, None] * HIDDEN_SIZE + n[None, :] * 1
    store_ptrs = base_cached + safe_m[:, None] * HIDDEN_SIZE + n[None, :] * 1
    mask = m_mask[:, None] & n_mask[None, :]
    val = tl.load(dst_ptrs, mask=mask, other=0)
    tl.store(store_ptrs, val, mask=mask)
