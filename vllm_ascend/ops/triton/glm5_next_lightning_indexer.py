# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fast path for the narrow GLM5 Next KPool lightning indexer.

The heavy part is scoring every compressed pool against the token's
head-weighted query: a paged-cache gather plus a 128-dim matvec. Doing that in
torch lowers to an aclnnIndex/SearchSorted small-op flood, so it stays in one
triton kernel that writes the raw pool scores to a scratch buffer. The top-k
selection itself is a single fused aclnn ``topk`` call, followed by a few
element-wise ops for pool->token expansion and causal tail append in the
reference path. The model selects compact_indices=True to fuse that postprocess
and graph-padding masking into one kernel per token chunk.

There is no hard length limit: the kernel tiles pools dynamically, and the
wrapper chunks the token dimension so the fp32 scores scratch stays under
``TRITON_SCORES_CHUNK_BYTES`` even for long-context prefill (e.g. 128K input).

Note: greedy/sort-based in-kernel top-k is deliberately avoided — reductions
over wide vectors scalarize on the Ascend triton backend, which is slower than
the torch fallback by two orders of magnitude.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2

# Fix the pool tile width to avoid recompiling as sequence lengths grow.
TRITON_POOL_CHUNK_SIZE = 2048
# Sub-tile of pools whose K rows are loaded as one coalesced 2D block.
# 128 pools x 128 dims x fp32 = 64KB, keeping the UB footprint small.
TRITON_POOL_SUB_TILE_SIZE = 128
# Chunk tokens to limit the FP32 score buffer to this budget where possible.
TRITON_SCORES_CHUNK_BYTES = 256 * 1024 * 1024
# Bound the postprocess vector working set independently of output width.
TRITON_INDEX_POSTPROCESS_TILE_SIZE = 256

# CANN's fused TopKV2 high_performance kernel faults (aicore VEC 507035) on -inf
# cells produced by a ragged mask at a non-aligned width. The fp32-lowest finite
# value orders identically -- no real score can reach it -- so the wrapper
# rewrites the mask sentinel to it on the way into top-k. Producers keep using
# -inf; the guard sits at the consumer that faults.
NEG_INF_SENTINEL = torch.finfo(torch.float32).min


# Keep batch-varying inputs unspecialized to avoid recompiling per step.
# REQ_POW2 stays constexpr for tl.arange; warm up its power-of-two variants.
@triton.jit(do_not_specialize=["token_offset", "max_pool_seq_len", "num_reqs", "num_cache_blocks"])
def _glm5_next_lightning_indexer_score_kernel(
    qbar_ptr,
    indexer_cache_ptr,
    cum_query_lens_ptr,
    indexer_seq_lens_ptr,
    indexer_block_table_ptr,
    positions_ptr,
    scores_ptr,
    token_offset,
    max_pool_seq_len,
    num_reqs,
    num_cache_blocks,
    cache_stride_block: tl.constexpr,
    cache_stride_offset: tl.constexpr,
    cache_stride_d: tl.constexpr,
    block_table_stride_req: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    pool_block_size: tl.constexpr,
    REQ_POW2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INDEX_KPOOL: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
    SUB_POOL: tl.constexpr,
):
    # qbar/scores rows are chunk-local; positions/query-boundary lookups use
    # the batch-global token index.
    local_token_idx = tl.program_id(0)
    token_idx = local_token_idx + token_offset
    chunk = tl.program_id(1)

    req_offsets = tl.arange(0, REQ_POW2)
    query_ends = tl.load(cum_query_lens_ptr + req_offsets, mask=req_offsets < num_reqs, other=2147483647)
    req_id = tl.sum(tl.where(token_idx >= query_ends, 1, 0))
    # Full ACL graphs keep padded rows beyond the last request; keep their
    # pointer arithmetic in bounds even though their outputs are unused.
    req_id = tl.minimum(req_id, num_reqs - 1)

    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    request_pool_len = tl.load(indexer_seq_lens_ptr + req_id).to(tl.int32)
    causal_pool_len = (pos + 1) // INDEX_KPOOL
    visible_pool_len = tl.minimum(causal_pool_len, request_pool_len)

    dim_offsets = tl.arange(0, HEAD_DIM)
    qbar = tl.load(qbar_ptr + local_token_idx * HEAD_DIM + dim_offsets)

    chunk_start = chunk * BLOCK_POOL
    # Dynamic trip count: requests shorter than the static max pool length
    # skip their out-of-range sub-tiles even inside captured graphs. Cells
    # beyond ``visible_pool_len`` keep the -inf the wrapper initialized.
    chunk_visible = tl.maximum(tl.minimum(visible_pool_len, chunk_start + BLOCK_POOL) - chunk_start, 0)
    num_subs = tl.cdiv(chunk_visible, SUB_POOL)
    for sub in tl.range(num_subs):
        pool_offsets = chunk_start + sub * SUB_POOL + tl.arange(0, SUB_POOL)
        in_range = pool_offsets < max_pool_seq_len
        valid_pool = in_range & (pool_offsets < visible_pool_len)
        logical_pages = pool_offsets // pool_block_size
        page_offsets = pool_offsets % pool_block_size
        physical_blocks = tl.load(
            indexer_block_table_ptr + req_id * block_table_stride_req + logical_pages * block_table_stride_page,
            mask=in_range,
            other=0,
        ).to(tl.int64)
        # Clamp both sides: padded/stale block-table entries must never form
        # an out-of-range cache address, even though their loads are masked.
        physical_blocks = tl.minimum(tl.maximum(physical_blocks, 0), num_cache_blocks - 1)
        k_addrs = (
            physical_blocks[:, None] * cache_stride_block
            + page_offsets[:, None] * cache_stride_offset
            + dim_offsets[None, :] * cache_stride_d
        )
        k_tile = tl.load(indexer_cache_ptr + k_addrs, mask=valid_pool[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k_tile * qbar[None, :], axis=1)
        scores = tl.where(valid_pool, scores, float("-inf"))
        tl.store(scores_ptr + local_token_idx * max_pool_seq_len + pool_offsets, scores, mask=in_range)


@triton.jit(do_not_specialize=["token_offset", "num_reqs"])
def _glm5_next_kpool_postprocess_kernel(
    pool_ids_ptr,
    topk_vals_ptr,
    positions_ptr,
    cum_query_lens_ptr,
    output_ptr,
    token_offset,
    num_reqs,
    TOPK: tl.constexpr,
    INDEX_TOPK: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    INVALID_SCORE: tl.constexpr,
):
    """Write expanded history, compact causal tail and graph padding once."""
    row = tl.program_id(0)
    token = token_offset + row
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    width = INDEX_TOPK + POOL_SIZE - 1
    actual_tokens = tl.load(cum_query_lens_ptr + num_reqs - 1)
    valid_row = token < actual_tokens
    pos = tl.load(positions_ptr + token).to(tl.int64)
    tail_start = (pos + 1) // POOL_SIZE * POOL_SIZE
    # Match append_causal_tail: the tail immediately follows available
    # complete pools, rather than leaving holes at a fixed top-k column.
    tail_col = tl.minimum(tail_start, INDEX_TOPK)
    tail_count = pos + 1 - tail_start
    value = tl.full((BLOCK,), -1, tl.int32)
    if TOPK > 0:
        slot = col // POOL_SIZE
        history_mask = valid_row & (col < width) & (col < INDEX_TOPK) & (slot < TOPK)
        pool = tl.load(pool_ids_ptr + row * TOPK + slot, mask=history_mask, other=0).to(tl.int32)
        score = tl.load(topk_vals_ptr + row * TOPK + slot, mask=history_mask, other=float("-inf"))
        value = tl.where(history_mask & (score > INVALID_SCORE), pool * POOL_SIZE + col % POOL_SIZE, -1)
    # The legacy scatter writes POOL_SIZE-1 lanes, including -1 lanes.
    # Preserve that overwrite behavior even for inconsistent/stale metadata.
    is_tail = (col >= tail_col) & (col < tail_col + POOL_SIZE - 1)
    tail_value = tl.where(col - tail_col < tail_count, tail_start + col - tail_col, -1)
    value = tl.where(is_tail, tail_value, value)
    value = tl.where(valid_row, value, -1).to(tl.int32)
    tl.store(output_ptr + token * width + col, value, mask=col < width)


def _write_compact_indices(
    pool_ids, topk_vals, positions, cum_query_lens, output, token_offset, rows, topk, index_topk, index_kpool
):
    # Tile the output width to keep vector working sets bounded at real top-k.
    block = TRITON_INDEX_POSTPROCESS_TILE_SIZE
    _glm5_next_kpool_postprocess_kernel[(rows, triton.cdiv(output.shape[-1], block))](
        pool_ids,
        topk_vals,
        positions,
        cum_query_lens,
        output,
        token_offset,
        cum_query_lens.shape[0],
        topk,
        index_topk,
        index_kpool,
        block,
        NEG_INF_SENTINEL,
    )


def glm5_next_lightning_indexer_triton(
    query: torch.Tensor,
    indexer_cache: torch.Tensor,
    weights: torch.Tensor,
    cum_query_lens: torch.Tensor,
    indexer_seq_lens: torch.Tensor,
    indexer_block_table: torch.Tensor,
    positions: torch.Tensor,
    *,
    index_topk: int,
    index_kpool: int,
    max_pool_seq_len: int,
    compact_indices: bool = False,
) -> torch.Tensor:
    pool_topk = index_topk // index_kpool
    output_width = index_topk + index_kpool - 1
    num_tokens = query.shape[0]
    if num_tokens == 0:
        return torch.empty(
            (0, 1, output_width),
            dtype=torch.int32,
            device=query.device,
        )

    output = torch.empty(
        (num_tokens, 1, output_width),
        dtype=torch.int32,
        device=query.device,
    )
    if max_pool_seq_len == 0 and compact_indices:
        # TOPK=0 removes all pool loads at compile time; use existing tensors
        # as unused pointer arguments, without allocating dummy device buffers.
        _write_compact_indices(
            positions, positions, positions, cum_query_lens, output, 0, num_tokens, 0, index_topk, index_kpool
        )
        return output
    if max_pool_seq_len == 0:
        output.fill_(-1)
        tail_offsets = torch.arange(index_kpool - 1, device=query.device)
        tail_start = (positions + 1) // index_kpool * index_kpool
        tail = torch.where(
            tail_offsets[None, :] < (positions + 1 - tail_start)[:, None],
            tail_start[:, None] + tail_offsets[None, :],
            -1,
        )
        output[:, 0, index_topk:] = tail.to(torch.int32)
        return output
    block_pool = TRITON_POOL_CHUNK_SIZE
    num_chunks = (max_pool_seq_len + block_pool - 1) // block_pool
    topk = min(pool_topk, max_pool_seq_len)
    if not compact_indices:
        token_offsets = torch.arange(index_kpool, device=query.device)
        tail_offsets = torch.arange(index_kpool - 1, device=query.device)

    # Chunk the token dimension so the fp32 scores scratch stays bounded;
    # long-context prefill would otherwise need num_tokens x max_pool_seq_len
    # x 4 bytes (over 1GB at 128K context with a full prefill batch).
    token_chunk = max(1, TRITON_SCORES_CHUNK_BYTES // (max_pool_seq_len * 4))
    for token_start in range(0, num_tokens, token_chunk):
        token_end = min(token_start + token_chunk, num_tokens)
        rows = token_end - token_start
        # Head-weighted query, computed once here instead of per chunk program.
        qbar = (
            (query[token_start:token_end].float() * weights[token_start:token_end].float().unsqueeze(-1))
            .sum(dim=1)
            .contiguous()
        )
        # -inf init: the kernel skips sub-tiles beyond a request's visible pools,
        # and those cells must stay excluded from the top-k. The wrapper rewrites
        # every -inf to NEG_INF_SENTINEL on the way into top-k.
        scores = torch.full(
            (rows, max_pool_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=query.device,
        )
        _glm5_next_lightning_indexer_score_kernel[(rows, num_chunks)](
            qbar,
            indexer_cache,
            cum_query_lens,
            indexer_seq_lens,
            indexer_block_table,
            positions,
            scores,
            token_start,
            max_pool_seq_len,
            cum_query_lens.shape[0],
            indexer_cache.shape[0],
            indexer_cache.stride(0),
            indexer_cache.stride(1),
            indexer_cache.stride(3),
            indexer_block_table.stride(0),
            indexer_block_table.stride(1),
            indexer_cache.shape[1],
            next_power_of_2(max(1, cum_query_lens.shape[0])),
            query.shape[2],
            index_kpool,
            block_pool,
            TRITON_POOL_SUB_TILE_SIZE,
        )

        # TopKV2 faults on -inf at a non-aligned width, so hand it the finite
        # sentinel instead. In place: the scratch is not read again after the
        # top-k, and a second full-size buffer on this path is real memory.
        # NaN is mapped to the same sentinel so it can never outrank a valid
        # score (the default rewrite to 0.0 could).
        scores.nan_to_num_(nan=NEG_INF_SENTINEL, neginf=NEG_INF_SENTINEL)
        topk_vals, pool_ids = torch.topk(scores, topk, dim=1)
        if compact_indices:
            _write_compact_indices(
                pool_ids, topk_vals, positions, cum_query_lens, output, token_start, rows, topk, index_topk, index_kpool
            )
            continue
        pool_ids = torch.where(
            topk_vals <= NEG_INF_SENTINEL,
            torch.full_like(pool_ids, -1),
            pool_ids,
        )
        history = pool_ids.unsqueeze(-1) * index_kpool + token_offsets
        history = torch.where(
            pool_ids.unsqueeze(-1) >= 0,
            history,
            torch.full_like(history, -1),
        ).reshape(rows, topk * index_kpool)
        if topk < pool_topk:
            history = torch.nn.functional.pad(
                history,
                (0, (pool_topk - topk) * index_kpool),
                value=-1,
            )

        pos = positions[token_start:token_end]
        tail_start = (pos + 1) // index_kpool * index_kpool
        tail_count = pos + 1 - tail_start
        tail = torch.where(
            tail_offsets[None, :] < tail_count[:, None],
            tail_start[:, None] + tail_offsets[None, :],
            -1,
        )
        output[token_start:token_end, 0] = torch.cat([history, tail], dim=1).to(torch.int32)
    return output
