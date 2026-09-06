# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused triton fast path for the GLM5 Next A3 indexer pre-compress sequence.

One launch per sparse layer per step does all of the following:

1. Write each token's full-resolution ``[k | gate_score]`` row into the paged
   compressor state cache (tokens with an invalid slot are masked out).
2. For tokens that complete a pool (valid indexer slot), gather the
   ``index_kpool`` window of states ending at the token's position. Window
   entries covered by the current query chunk are read straight from the
   ``k``/``gate_score`` inputs (same-launch cache reads would race with the
   state writes above); older entries come from the paged state cache.
3. Compress the window with ``softmax(gate_score + ape)`` over the pool axis
   and write the BF16 vector into the paged indexer cache.

Doing this in torch lowers to an aclnnIndex/SearchSorted/where small-op flood
per layer, so the whole sequence stays inside a single kernel.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

TRITON_MAX_BLOCK_D = 128


# JIT-variant hygiene: dims that change from step to step (num_tokens and
# num_reqs vary with the batch) are runtime arguments, not constexpr. As
# constexpr they would join triton's cache key, so every batch change in the
# eager paths (prefill chunks, MTP draft steps) would trigger a multi-second
# recompile and starve the device. They are also excluded from value
# specialization (==1 / divisible-by-16), so the variant set is exactly the
# REQ_POW2 values — powers of two up to the max batch size, all compiled once
# at graph-capture warmup. REQ_POW2 itself must stay constexpr for tl.arange.
@triton.jit(do_not_specialize=["num_reqs", "num_tokens"])
def _glm5_next_kpool_state_compress_kernel(
    state_cache_ptr,
    indexer_cache_ptr,
    k_ptr,
    gate_score_ptr,
    ape_ptr,
    positions_ptr,
    cum_query_lens_ptr,
    seq_lens_ptr,
    state_slot_mapping_ptr,
    state_block_table_ptr,
    indexer_slot_mapping_ptr,
    num_reqs,
    num_tokens,
    k_stride_t: tl.constexpr,
    gate_score_stride_t: tl.constexpr,
    ape_stride_p: tl.constexpr,
    state_cache_stride_block: tl.constexpr,
    state_cache_stride_offset: tl.constexpr,
    state_cache_stride_d: tl.constexpr,
    indexer_cache_stride_block: tl.constexpr,
    indexer_cache_stride_offset: tl.constexpr,
    indexer_cache_stride_d: tl.constexpr,
    state_block_table_stride_req: tl.constexpr,
    state_block_table_stride_page: tl.constexpr,
    state_num_slots: tl.constexpr,
    indexer_num_slots: tl.constexpr,
    state_num_blocks: tl.constexpr,
    state_max_pages: tl.constexpr,
    state_block_size: tl.constexpr,
    indexer_block_size: tl.constexpr,
    REQ_POW2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    dim_offsets = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM

    # Request bucketize: requests are packed contiguously in the batch.
    req_offsets = tl.arange(0, REQ_POW2)
    query_ends = tl.load(
        cum_query_lens_ptr + req_offsets,
        mask=req_offsets < num_reqs,
        other=2147483647,
    )
    req_id = tl.sum(tl.where(token_idx >= query_ends, 1, 0))
    # Full ACL graphs keep padded rows beyond the last request; keep their
    # pointer arithmetic in bounds even though their stores are masked.
    req_id = tl.minimum(req_id, num_reqs - 1)

    k_row = tl.load(
        k_ptr + token_idx * k_stride_t + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )
    gate_row = tl.load(
        gate_score_ptr + token_idx * gate_score_stride_t + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )

    # 1) Full-resolution state write. One state row is [k | gate_score].
    state_slot = tl.load(state_slot_mapping_ptr + token_idx).to(tl.int64)
    state_valid = (state_slot >= 0) & (state_slot < state_num_slots)
    safe_state_slot = tl.where(state_valid, state_slot, 0)
    state_block = safe_state_slot // state_block_size
    state_offset = safe_state_slot % state_block_size
    state_row_ptr = (
        state_cache_ptr
        + state_block * state_cache_stride_block
        + state_offset * state_cache_stride_offset
    )
    state_write_mask = dim_mask & state_valid
    tl.store(
        state_row_ptr + dim_offsets * state_cache_stride_d,
        k_row,
        mask=state_write_mask,
    )
    tl.store(
        state_row_ptr
        + (HEAD_DIM + dim_offsets) * state_cache_stride_d,
        gate_row,
        mask=state_write_mask,
    )

    # 2) Pool window gather for pool-completing tokens.
    indexer_slot = tl.load(indexer_slot_mapping_ptr + token_idx).to(tl.int64)
    indexer_valid = (indexer_slot >= 0) & (indexer_slot < indexer_num_slots)

    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    query_end = tl.load(cum_query_lens_ptr + req_id)
    prev_query_end = tl.load(
        cum_query_lens_ptr + req_id - 1,
        mask=req_id > 0,
        other=0,
    )
    seq_len = tl.load(seq_lens_ptr + req_id)
    request_query_start = seq_len - (query_end - prev_query_end)

    pool_offsets = tl.arange(0, BLOCK_P)
    pool_mask = pool_offsets < POOL_SIZE
    # Column j holds the state at position pos - (POOL_SIZE - 1 - j).
    pool_pos = pos - (POOL_SIZE - 1 - pool_offsets)
    eff_pos = tl.maximum(pool_pos, 0)
    in_window = pool_mask & (pool_pos >= request_query_start)

    # In-window rows live in this launch's k/gate_score inputs; the matching
    # input row is the batch row of the token at that position.
    src_row = prev_query_end + eff_pos - request_query_start
    src_row = tl.minimum(tl.maximum(src_row, 0), num_tokens - 1)
    window_mask = in_window[:, None] & dim_mask[None, :]
    pool_k_in = tl.load(
        k_ptr + src_row[:, None] * k_stride_t + dim_offsets[None, :],
        mask=window_mask,
        other=0.0,
    ).to(tl.float32)
    pool_g_in = tl.load(
        gate_score_ptr
        + src_row[:, None] * gate_score_stride_t
        + dim_offsets[None, :],
        mask=window_mask,
        other=0.0,
    ).to(tl.float32)

    # Older rows come from the paged state cache (written by earlier steps).
    page = tl.minimum(eff_pos // state_block_size, state_max_pages - 1)
    page_offset = eff_pos % state_block_size
    physical = tl.load(
        state_block_table_ptr
        + req_id * state_block_table_stride_req
        + page * state_block_table_stride_page,
        mask=pool_mask,
        other=0,
    ).to(tl.int64)
    physical = tl.minimum(tl.maximum(physical, 0), state_num_blocks - 1)
    hist_addr = (
        physical[:, None] * state_cache_stride_block
        + page_offset[:, None] * state_cache_stride_offset
    )
    hist_mask = (
        pool_mask[:, None]
        & (~in_window)[:, None]
        & dim_mask[None, :]
    )
    pool_k_hist = tl.load(
        state_cache_ptr
        + hist_addr
        + dim_offsets[None, :] * state_cache_stride_d,
        mask=hist_mask,
        other=0.0,
    ).to(tl.float32)
    pool_g_hist = tl.load(
        state_cache_ptr
        + hist_addr
        + (HEAD_DIM + dim_offsets[None, :]) * state_cache_stride_d,
        mask=hist_mask,
        other=0.0,
    ).to(tl.float32)

    pool_k = tl.where(in_window[:, None], pool_k_in, pool_k_hist)
    pool_g = tl.where(in_window[:, None], pool_g_in, pool_g_hist)

    # 3) softmax(gate + ape) over the pool axis, weighted sum of K.
    ape = tl.load(
        ape_ptr
        + pool_offsets[:, None] * ape_stride_p
        + dim_offsets[None, :],
        mask=pool_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.where(pool_mask[:, None], pool_g + ape, float("-inf"))
    score_max = tl.max(scores, axis=0)
    weights = tl.exp(scores - score_max[None, :])
    weights = weights / tl.sum(weights, axis=0)[None, :]
    compressed = tl.sum(weights * pool_k, axis=0)

    safe_indexer_slot = tl.where(indexer_valid, indexer_slot, 0)
    indexer_block = safe_indexer_slot // indexer_block_size
    indexer_offset = safe_indexer_slot % indexer_block_size
    tl.store(
        indexer_cache_ptr
        + indexer_block * indexer_cache_stride_block
        + indexer_offset * indexer_cache_stride_offset
        + dim_offsets * indexer_cache_stride_d,
        compressed,
        mask=dim_mask & indexer_valid,
    )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def glm5_next_kpool_state_compress_and_write_cache_triton(
    state_cache: torch.Tensor,
    indexer_cache: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    cum_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    state_block_table: torch.Tensor,
    indexer_slot_mapping: torch.Tensor,
    index_kpool: int,
) -> None:
    """Fused state write + pool compress + indexer cache write (triton)."""
    num_tokens, head_dim = k.shape
    if num_tokens == 0:
        return

    if not k.is_contiguous():
        k = k.contiguous()
    if not gate_score.is_contiguous():
        gate_score = gate_score.contiguous()
    if not ape.is_contiguous():
        ape = ape.contiguous()
    if not positions.is_contiguous():
        positions = positions.contiguous()
    if not cum_query_lens.is_contiguous():
        cum_query_lens = cum_query_lens.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not state_slot_mapping.is_contiguous():
        state_slot_mapping = state_slot_mapping.contiguous()
    if not state_block_table.is_contiguous():
        state_block_table = state_block_table.contiguous()
    if not indexer_slot_mapping.is_contiguous():
        indexer_slot_mapping = indexer_slot_mapping.contiguous()

    block_p = _next_power_of_2(index_kpool)
    block_d = min(_next_power_of_2(head_dim), TRITON_MAX_BLOCK_D)
    num_reqs = cum_query_lens.shape[0]
    grid = num_tokens, triton.cdiv(head_dim, block_d)
    _glm5_next_kpool_state_compress_kernel[grid](
        state_cache,
        indexer_cache,
        k,
        gate_score,
        ape,
        positions,
        cum_query_lens,
        seq_lens,
        state_slot_mapping,
        state_block_table,
        indexer_slot_mapping,
        num_reqs,
        num_tokens,
        k.stride(0),
        gate_score.stride(0),
        ape.stride(0),
        state_cache.stride(0),
        state_cache.stride(1),
        state_cache.stride(2),
        indexer_cache.stride(0),
        indexer_cache.stride(1),
        indexer_cache.stride(3),
        state_block_table.stride(0),
        state_block_table.stride(1),
        state_cache.shape[0] * state_cache.shape[1],
        indexer_cache.shape[0] * indexer_cache.shape[1],
        state_cache.shape[0],
        state_block_table.shape[1],
        state_cache.shape[1],
        indexer_cache.shape[1],
        _next_power_of_2(max(1, num_reqs)),
        head_dim,
        index_kpool,
        block_p,
        block_d,
    )
