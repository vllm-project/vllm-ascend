# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fast path for the narrow GLM5 Next KPool lightning indexer.

The heavy part is scoring every compressed pool against the token's
head-weighted query: a paged-cache gather plus a 128-dim matvec. Doing that in
torch lowers to an aclnnIndex/SearchSorted small-op flood, so it stays in one
triton kernel that writes the raw pool scores to a scratch buffer. The top-k
selection itself is a single fused aclnn ``topk`` call, followed by a few
element-wise ops for pool->token expansion and causal tail append.

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

# Pool scores are produced in tiles of this width per program; the grid's
# second dimension covers ``max_pool_seq_len / TRITON_POOL_CHUNK_SIZE`` chunks.
# The width is deliberately fixed: a length-dependent tile width would become
# part of triton's JIT cache key and trigger a recompile for every power of
# two the context length crosses.
TRITON_POOL_CHUNK_SIZE = 2048
# Sub-tile of pools whose K rows are loaded as one coalesced 2D block.
# 128 pools x 128 dims x fp32 = 64KB, keeping the UB footprint small.
TRITON_POOL_SUBTILE_SIZE = 128
# Budget for the fp32 scores scratch of one token chunk. The launcher chunks
# the token dimension so arbitrary batch sizes and context lengths stay in
# triton without an unbounded [num_tokens, max_pool_seq_len] allocation.
TRITON_SCORES_CHUNK_BYTES = 256 * 1024 * 1024


# JIT-variant hygiene: dims that change from step to step (max_pool_seq_len
# grows with the context, num_reqs with the batch) are runtime arguments, not
# constexpr. As constexpr they would join triton's cache key, so every length
# or batch change in the eager paths (prefill chunks, MTP draft steps) would
# trigger a multi-second recompile and starve the device. They are also
# excluded from value specialization (==1 / divisible-by-16), so the variant
# set is exactly the REQ_POW2 values — powers of two up to the max batch size,
# all compiled once at graph-capture warmup. REQ_POW2 itself must stay
# constexpr for tl.arange.
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


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


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
    block_pool = TRITON_POOL_CHUNK_SIZE
    num_chunks = (max_pool_seq_len + block_pool - 1) // block_pool
    topk = min(pool_topk, max_pool_seq_len)
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
            query[token_start:token_end].float() * weights[token_start:token_end].float().unsqueeze(-1)
        ).sum(dim=1).contiguous()
        # -inf init: the kernel skips sub-tiles beyond a request's visible pools,
        # and those cells must stay excluded from the top-k.
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
            _next_power_of_2(max(1, cum_query_lens.shape[0])),
            query.shape[2],
            index_kpool,
            block_pool,
            TRITON_POOL_SUBTILE_SIZE,
        )

        topk_vals, pool_ids = torch.topk(scores, topk, dim=1)
        pool_ids = torch.where(
            topk_vals == float("-inf"),
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
