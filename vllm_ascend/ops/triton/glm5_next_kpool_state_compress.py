# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fast path for the GLM5 Next indexer pre-compress sequence.

Two ordered launches per sparse layer perform the following:

1. For tokens that complete a pool (valid indexer slot), gather the
   ``index_kpool`` window of states ending at the token's position. Window
   entries covered by the current query chunk are read straight from the
   ``k``/``gate_score`` inputs; older entries come from the paged state cache.
2. Compress the window with ``softmax(gate_score + ape)`` over the pool axis
   and write the BF16 vector into the paged indexer cache.
3. A second launch writes each valid token to its scheduler-provided state slot.
   This ordering keeps long prefills from overwriting historical rows while
   another program is still reading them to complete the first pool.

Doing this in torch lowers to an aclnnIndex/SearchSorted/where small-op flood
per layer, so the sequence stays in Triton.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2

TRITON_MAX_BLOCK_D = 128


# Keep batch-varying inputs unspecialized to avoid recompiling per step.
# REQ_POW2 stays constexpr for tl.arange; warm up its power-of-two variants.
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

    # 2) Pool window gather for pool-completing tokens.
    indexer_slot = tl.load(indexer_slot_mapping_ptr + token_idx).to(tl.int64)
    last_query_end = tl.load(cum_query_lens_ptr + num_reqs - 1)
    indexer_valid = (indexer_slot >= 0) & (indexer_slot < indexer_num_slots) & (token_idx < last_query_end)

    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    query_end = tl.load(cum_query_lens_ptr + req_id)
    prev_query_end = tl.load(cum_query_lens_ptr + req_id - 1, mask=req_id > 0, other=0)
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
        gate_score_ptr + src_row[:, None] * gate_score_stride_t + dim_offsets[None, :],
        mask=window_mask,
        other=0.0,
    ).to(tl.float32)

    # Older rows come from the paged state cache (written by earlier steps).
    page = eff_pos // state_block_size
    history_valid = indexer_valid & pool_mask & (pool_pos >= 0) & (~in_window) & (page < state_max_pages)
    page_offset = eff_pos % state_block_size
    physical = tl.load(
        state_block_table_ptr + req_id * state_block_table_stride_req + page * state_block_table_stride_page,
        mask=history_valid,
        other=-1,
    ).to(tl.int64)
    history_valid = history_valid & (physical >= 0) & (physical < state_num_blocks)
    physical = tl.where(history_valid, physical, 0)
    hist_addr = physical[:, None] * state_cache_stride_block + page_offset[:, None] * state_cache_stride_offset
    hist_mask = history_valid[:, None] & dim_mask[None, :]
    pool_k_hist = tl.load(
        state_cache_ptr + hist_addr + dim_offsets[None, :] * state_cache_stride_d,
        mask=hist_mask,
        other=0.0,
    ).to(tl.float32)
    pool_g_hist = tl.load(
        state_cache_ptr + hist_addr + (HEAD_DIM + dim_offsets[None, :]) * state_cache_stride_d,
        mask=hist_mask,
        other=0.0,
    ).to(tl.float32)

    pool_k = tl.where(in_window[:, None], pool_k_in, pool_k_hist)
    pool_g = tl.where(in_window[:, None], pool_g_in, pool_g_hist)

    # 3) softmax(gate + ape) over the pool axis, weighted sum of K.
    ape = tl.load(
        ape_ptr + pool_offsets[:, None] * ape_stride_p + dim_offsets[None, :],
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


@triton.jit(do_not_specialize=["num_reqs"])
def _store_kpool_state_kernel(
    state,
    k,
    gate,
    query_ends,
    state_slots,
    num_reqs,
    state_stride_b: tl.constexpr,
    state_stride_t: tl.constexpr,
    state_stride_d: tl.constexpr,
    k_stride_t: tl.constexpr,
    gate_stride_t: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    end = tl.load(query_ends + num_reqs - 1)
    slot = tl.load(state_slots + row).to(tl.int64)
    valid = (row < end) & (slot >= 0) & (slot < NUM_BLOCKS * BLOCK_SIZE)
    safe_slot = tl.where(valid, slot, 0)
    block, offset = safe_slot // BLOCK_SIZE, safe_slot % BLOCK_SIZE
    mask = valid & (dims < HEAD_DIM)
    key = tl.load(k + row * k_stride_t + dims, mask=mask, other=0)
    score = tl.load(gate + row * gate_stride_t + dims, mask=mask, other=0)
    addr = state + block * state_stride_b + offset * state_stride_t
    tl.store(addr + dims * state_stride_d, key, mask=mask)
    tl.store(addr + (HEAD_DIM + dims) * state_stride_d, score, mask=mask)


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
    """Compress pools, then write states using the original paged slot mapping."""
    num_tokens, head_dim = k.shape
    if num_tokens == 0 or cum_query_lens.numel() == 0:
        return
    if state_block_table.shape[1] == 0 or state_cache.shape[1] < index_kpool:
        raise ValueError("KPool requires a nonempty state page table and block size >= pool size.")

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

    block_p = next_power_of_2(index_kpool)
    block_d = min(next_power_of_2(head_dim), TRITON_MAX_BLOCK_D)
    num_reqs = cum_query_lens.shape[0]
    _glm5_next_kpool_state_compress_kernel[(num_tokens, triton.cdiv(head_dim, block_d))](
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
        next_power_of_2(max(1, num_reqs)),
        head_dim,
        index_kpool,
        block_p,
        block_d,
    )
    _store_kpool_state_kernel[(num_tokens, triton.cdiv(head_dim, block_d))](
        state_cache,
        k,
        gate_score,
        cum_query_lens,
        state_slot_mapping,
        num_reqs,
        state_cache.stride(0),
        state_cache.stride(1),
        state_cache.stride(2),
        k.stride(0),
        gate_score.stride(0),
        state_cache.shape[0],
        state_cache.shape[1],
        head_dim,
        block_d,
    )
