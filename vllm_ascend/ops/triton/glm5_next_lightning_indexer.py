# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged, head-wise GLM KPool scoring and fused token-index expansion."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2

TRITON_POOL_CHUNK_SIZE = 128
TRITON_PREFILL_POOL_CHUNK_SIZE = 256
TRITON_SCORES_CHUNK_BYTES = 256 * 1024 * 1024
TRITON_MAX_PROGRAMS = 256
TRITON_INDEX_BLOCK = 256
TRITON_PREFILL_INDEX_BLOCK = 4096
TRITON_PREFILL_INDEX_MIN_TOKENS = 512
TRITON_PREFILL_MIN_TOKENS = 32
TRITON_PREFILL_POOL_TILE = 2048
TRITON_PACKED_CACHE_BYTES = 256 * 1024 * 1024
NEG_INF_SENTINEL = torch.finfo(torch.float32).min


@triton.jit(do_not_specialize=["requests", "pools_count", "blocks_count"])
def _gather_pool_cache(
    cache,
    table,
    lengths,
    packed,
    requests,
    pools_count,
    blocks_count,
    CACHE_BLOCK: tl.constexpr,
    CACHE_SB: tl.constexpr,
    CACHE_ST: tl.constexpr,
    CACHE_SD: tl.constexpr,
    TABLE_SR: tl.constexpr,
    TABLE_SP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    chunks = tl.cdiv(pools_count, BLOCK)
    for tile in range(tl.program_id(0), requests * chunks, tl.num_programs(0)):
        request, chunk = tile // chunks, tile % chunks
        pools = chunk * BLOCK + tl.arange(0, BLOCK)
        dims = tl.arange(0, HEAD_DIM)
        valid = (pools < pools_count) & (pools < tl.load(lengths + request))
        blocks = tl.load(table + request * TABLE_SR + (pools // CACHE_BLOCK) * TABLE_SP, valid, other=0)
        blocks = tl.minimum(tl.maximum(blocks, 0), blocks_count - 1)
        offsets = blocks[:, None] * CACHE_SB + (pools % CACHE_BLOCK)[:, None] * CACHE_ST
        keys = tl.load(cache + offsets + dims[None, :] * CACHE_SD, valid[:, None], other=0)
        tl.store(
            packed + (request * pools_count + pools[:, None]) * HEAD_DIM + dims[None, :],
            keys,
            (pools < pools_count)[:, None],
        )


@triton.jit(do_not_specialize=["token_offset", "rows", "max_pool_seq_len", "num_reqs", "num_cache_blocks"])
def _glm5_next_lightning_indexer_score_kernel(
    query,
    cache,
    weights,
    query_ends,
    pool_lens,
    block_table,
    positions,
    output,
    token_offset,
    rows,
    max_pool_seq_len,
    num_reqs,
    num_cache_blocks,
    query_stride_t: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_d: tl.constexpr,
    weight_stride_t: tl.constexpr,
    weight_stride_h: tl.constexpr,
    cache_stride_b: tl.constexpr,
    cache_stride_t: tl.constexpr,
    cache_stride_d: tl.constexpr,
    table_stride_r: tl.constexpr,
    table_stride_p: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    REQ_POW2: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
    LOWEST: tl.constexpr,
    INDEX32: tl.constexpr,
    OUTER_POOL: tl.constexpr,
    PACKED_CACHE: tl.constexpr,
):
    chunks = tl.cdiv(max_pool_seq_len, OUTER_POOL)
    for tile in range(tl.program_id(0), rows * chunks, tl.num_programs(0)):
        row, chunk = tile // chunks, tile % chunks
        token = row + token_offset
        requests = tl.arange(0, REQ_POW2)
        ends = tl.load(query_ends + requests, requests < num_reqs, other=2147483647)
        request = tl.minimum(tl.sum((token >= ends).to(tl.int32)), num_reqs - 1)
        position = tl.load(positions + token).to(tl.int32)
        visible = tl.minimum((position + 1) // POOL, tl.load(pool_lens + request))
        live = token < tl.load(query_ends + num_reqs - 1)
        heads = tl.arange(0, BLOCK_H)
        dims = tl.arange(0, HEAD_DIM)
        q = tl.load(
            query + token * query_stride_t + heads[:, None] * query_stride_h + dims[None, :] * query_stride_d,
            heads[:, None] < HEADS,
            other=0,
        )
        w = tl.load(weights + token * weight_stride_t + heads * weight_stride_h, heads < HEADS, other=0).to(tl.float32)
        sub_tiles = tl.cdiv(tl.minimum(max_pool_seq_len - chunk * OUTER_POOL, OUTER_POOL), BLOCK_POOL)
        for sub in range(sub_tiles):
            pool_start = chunk * OUTER_POOL + sub * BLOCK_POOL
            pools = pool_start + tl.arange(0, BLOCK_POOL)
            scores = tl.full((BLOCK_POOL,), LOWEST, tl.float32)
            if live & (pool_start < visible):
                valid = (pools < max_pool_seq_len) & (pools < visible)
                if PACKED_CACHE:
                    k = tl.load(
                        cache + (request * max_pool_seq_len + pools[:, None]) * HEAD_DIM + dims[None, :],
                        valid[:, None],
                        other=0,
                    )
                elif INDEX32:
                    pages = pools // CACHE_BLOCK
                    blocks = tl.load(
                        block_table + request * table_stride_r + pages * table_stride_p,
                        pools < max_pool_seq_len,
                        other=0,
                    )
                    blocks = tl.minimum(tl.maximum(blocks, 0), num_cache_blocks - 1)
                    offsets = blocks[:, None] * cache_stride_b + (pools % CACHE_BLOCK)[:, None] * cache_stride_t
                    k = tl.load(cache + offsets + dims[None, :] * cache_stride_d, valid[:, None], other=0)
                else:
                    # Large caches need a scalar int64 page base; vector offsets
                    # remain within one page to avoid overflowing signed int32.
                    block = tl.load(
                        block_table + request * table_stride_r + (pool_start // CACHE_BLOCK) * table_stride_p
                    )
                    block = tl.minimum(tl.maximum(block, 0), num_cache_blocks - 1).to(tl.int64)
                    page_base = cache + block * cache_stride_b
                    offsets = tl.arange(0, BLOCK_POOL)
                    k = tl.load(
                        page_base + offsets[:, None] * cache_stride_t + dims[None, :] * cache_stride_d,
                        valid[:, None],
                        other=0,
                    )
                # ReLU is applied per head before the weighted head reduction.
                # Summing weighted queries first is not algebraically equivalent.
                per_head = tl.dot(k, tl.trans(q), input_precision="ieee")
                # The comparison preserves NaN, unlike maxnum semantics.
                per_head = tl.where(per_head < 0.0, 0.0, per_head)
                scores = tl.sum(per_head * w[None, :], axis=1)
                scores = tl.where(valid & (scores == scores), tl.minimum(tl.maximum(scores, LOWEST), -LOWEST), LOWEST)
            tl.store(output + row * max_pool_seq_len + pools, scores, pools < max_pool_seq_len)


@triton.jit(do_not_specialize=["rows", "token_offset", "selected", "last_query"])
def _expand_pool_indices(
    pool_ids,
    topk_scores,
    positions,
    query_ends,
    output,
    rows,
    token_offset,
    last_query,
    selected,
    HAS_POOLS: tl.constexpr,
    TOPK: tl.constexpr,
    POOL: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    PACK_TAIL: tl.constexpr,
    LOWEST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tiles = tl.cdiv(OUTPUT_WIDTH, BLOCK)
    for tile in range(tl.program_id(0), rows * tiles, tl.num_programs(0)):
        row, column_tile = tile // tiles, tile % tiles
        columns = column_tile * BLOCK + tl.arange(0, BLOCK)
        token = row + token_offset
        position = tl.load(positions + token).to(tl.int64)
        live = token < tl.load(query_ends + last_query)
        values = tl.full((BLOCK,), -1, tl.int32)
        if HAS_POOLS:
            if BLOCK % POOL == 0:
                selected_pools = column_tile * (BLOCK // POOL) + tl.arange(0, BLOCK // POOL)
                load_mask = (selected_pools < selected) & (selected_pools * POOL < TOPK)
                pool_values = tl.load(pool_ids + row * selected + selected_pools, load_mask, other=-1)
                score_values = tl.load(topk_scores + row * selected + selected_pools, load_mask, other=LOWEST)
                repeat_offsets = tl.arange(0, BLOCK) // POOL
                # Ascend gather supports fp32 but not int32; preserve ID bits.
                ids = tl.gather(pool_values.to(tl.float32, bitcast=True), repeat_offsets, axis=0).to(
                    tl.int32, bitcast=True
                )
                scores = tl.gather(score_values, repeat_offsets, axis=0)
            else:
                selected_offsets = columns // POOL
                load_mask = (columns < TOPK) & (selected_offsets < selected)
                ids = tl.load(pool_ids + row * selected + selected_offsets, load_mask, other=-1)
                scores = tl.load(topk_scores + row * selected + selected_offsets, load_mask, other=LOWEST)
            values = tl.where((columns < TOPK) & (scores > LOWEST), ids * POOL + columns % POOL, -1)
        tail_start = (position + 1) // POOL * POOL
        tail_column = tl.minimum(tail_start, TOPK).to(tl.int32) if PACK_TAIL else TOPK
        tail_offset = columns - tail_column
        tail_count = (position + 1 - tail_start).to(tl.int32)
        is_tail = (tail_offset >= 0) & (tail_offset < POOL - 1)
        values = tl.where(
            is_tail, tl.where(tail_offset < tail_count, tail_start.to(tl.int32) + tail_offset, -1), values
        )
        values = tl.where(live, values, -1)
        tl.store(output + token * OUTPUT_STRIDE + columns, values, columns < OUTPUT_WIDTH)


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
    output_buffer: torch.Tensor | None = None,
    pack_tail: bool = False,
) -> torch.Tensor:
    """Select pools, expand tokens and optionally write the final SFA buffer."""
    num_tokens, num_heads, head_dim = query.shape
    output_width = index_topk + index_kpool - 1
    if output_buffer is None:
        output = torch.empty((num_tokens, output_width), dtype=torch.int32, device=query.device)
    else:
        if (
            output_buffer.ndim != 2
            or output_buffer.dtype != torch.int32
            or output_buffer.device != query.device
            or output_buffer.stride(1) != 1
            or (num_tokens > 1 and output_buffer.stride(0) < output_buffer.shape[1])
        ):
            raise ValueError("GLM KPool output buffer requires int32 contiguous columns on the query device.")
        if output_buffer.shape[0] < num_tokens or output_buffer.shape[1] < output_width:
            raise ValueError("GLM KPool output buffer must cover the token rows and top-k width.")
        output = output_buffer[:num_tokens]
    if num_tokens == 0:
        return output[:, :output_width].unsqueeze(1)
    if cum_query_lens.numel() == 0:
        raise ValueError("GLM KPool nonempty queries require request boundaries.")
    cache_span = sum((size - 1) * stride for size, stride in zip(indexer_cache.shape, indexer_cache.stride()))
    index32 = cache_span <= torch.iinfo(torch.int32).max
    page_span = (indexer_cache.shape[1] - 1) * indexer_cache.stride(1) + (head_dim - 1) * indexer_cache.stride(3)
    if page_span > torch.iinfo(torch.int32).max:
        raise ValueError("GLM KPool cache page offsets must fit in int32.")
    if not index32 and indexer_cache.shape[1] != next_power_of_2(indexer_cache.shape[1]):
        raise ValueError("GLM KPool large-cache pages require a power-of-two block size.")
    selected = min(index_topk // index_kpool, max_pool_seq_len)
    packed_cache = (
        num_tokens >= TRITON_PREFILL_MIN_TOKENS
        and selected > 0
        and index32
        and cum_query_lens.numel() * max_pool_seq_len * head_dim * indexer_cache.element_size()
        <= TRITON_PACKED_CACHE_BYTES
    )
    if packed_cache:
        cache = torch.empty(
            (cum_query_lens.numel(), max_pool_seq_len, 1, head_dim),
            dtype=indexer_cache.dtype,
            device=indexer_cache.device,
        )
        _gather_pool_cache[
            (min(cum_query_lens.numel() * triton.cdiv(max_pool_seq_len, TRITON_POOL_CHUNK_SIZE), TRITON_MAX_PROGRAMS),)
        ](
            indexer_cache,
            indexer_block_table,
            indexer_seq_lens,
            cache,
            cum_query_lens.numel(),
            max_pool_seq_len,
            indexer_cache.shape[0],
            indexer_cache.shape[1],
            indexer_cache.stride(0),
            indexer_cache.stride(1),
            indexer_cache.stride(3),
            *indexer_block_table.stride(),
            head_dim,
            TRITON_POOL_CHUNK_SIZE,
        )
    else:
        cache = indexer_cache
    token_chunk = max(1, TRITON_SCORES_CHUNK_BYTES // max(4, max_pool_seq_len * 4))
    for start in range(0, num_tokens, token_chunk):
        rows = min(token_chunk, num_tokens - start)
        if selected:
            scores = torch.empty((rows, max_pool_seq_len), dtype=torch.float32, device=query.device)
            block_pool = TRITON_POOL_CHUNK_SIZE if index32 else indexer_cache.shape[1]
            if packed_cache:
                block_pool = TRITON_PREFILL_POOL_CHUNK_SIZE
            outer_pool = max(TRITON_PREFILL_POOL_TILE, block_pool) if rows >= TRITON_PREFILL_MIN_TOKENS else block_pool
            chunks = triton.cdiv(max_pool_seq_len, outer_pool)
            _glm5_next_lightning_indexer_score_kernel[(min(rows * chunks, TRITON_MAX_PROGRAMS),)](
                query,
                cache,
                weights,
                cum_query_lens,
                indexer_seq_lens,
                indexer_block_table,
                positions,
                scores,
                start,
                rows,
                max_pool_seq_len,
                cum_query_lens.numel(),
                indexer_cache.shape[0],
                *query.stride(),
                *weights.stride(),
                0 if packed_cache else cache.stride(0),
                cache.stride(1),
                cache.stride(3),
                *indexer_block_table.stride(),
                indexer_cache.shape[1],
                next_power_of_2(cum_query_lens.numel()),
                num_heads,
                max(16, next_power_of_2(num_heads)),
                head_dim,
                index_kpool,
                block_pool,
                NEG_INF_SENTINEL,
                index32,
                outer_pool,
                packed_cache,
            )
            topk_scores, pool_ids = torch.topk(scores, selected, dim=1)
            # Ascend scalarizes Triton's int64-to-int32 vector conversion.
            # Convert once before expanding each selected pool into tokens.
            pool_ids = pool_ids.to(torch.int32)
        else:
            topk_scores = pool_ids = None
        # Wide expansion amortizes row scheduling only for large token batches.
        index_block = TRITON_PREFILL_INDEX_BLOCK if rows >= TRITON_PREFILL_INDEX_MIN_TOKENS else TRITON_INDEX_BLOCK
        tiles = triton.cdiv(output.shape[1], index_block)
        _expand_pool_indices[(min(rows * tiles, TRITON_MAX_PROGRAMS),)](
            pool_ids,
            topk_scores,
            positions,
            cum_query_lens,
            output,
            rows,
            start,
            cum_query_lens.numel() - 1,
            selected,
            selected > 0,
            index_topk,
            index_kpool,
            output.shape[1],
            output.stride(0),
            pack_tail,
            NEG_INF_SENTINEL,
            index_block,
        )
    return output[:, :output_width].unsqueeze(1)
