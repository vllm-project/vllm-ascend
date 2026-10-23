# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged GLM KPool scoring and fused token-index expansion.

LI C8 pooled rows hold ``head_dim`` INT8 bytes plus one little-endian FP32
scale. Ascend Triton cannot tile-load single bytes, and unpacking a row once
per query row costs more than the dot product itself: at 128K context one
prefill call spent 290 ms of its 294 ms there. A prefill batch therefore
unpacks the pooled rows its requests use into contiguous BF16 keys plus FP32
scales once per layer, which costs 0.03 ms for four 128K requests, and scores
those rows with a single ``head_dim``-deep dot per tile while several query
rows share one program. The rows stay contiguous because a block table gather
turns every key tile into per-element loads.

Each head's dot product passes through ReLU before the weighted head reduction,
matching CANN PoolKeyIndexer: collapsing the query first would change pool
selection whenever a dot product is negative. Decode batches keep the paged
INT8 kernel below, whose per-row tiles a packed batch cannot replace because
each request contributes a single row. The legacy BF16 helper retains its
direct row loads.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2

TRITON_POOL_CHUNK_SIZE = 128
# Bound the temporary tiles used by INT8 unpacking and per-head scoring.
TRITON_POOL_CHUNK_SIZE_C8 = 64
# Query rows per packed C8 score program: one key tile feeds the whole block.
TRITON_C8_ROWS_PER_BLOCK = 4
TRITON_C8_ROWS_PER_BLOCK_WIDE = 32
TRITON_C8_POOL_TILE = 128
TRITON_C8_UNPACK_BLOCK = 128
# The score kernel's tile space always covers the whole token batch, so one
# launch per forward needs a buffer for every row: at 128K that is 8,192 rows x
# 32,768 pools = 1 GiB of fp32. Splitting it four ways would score the batch
# four times over.
TRITON_SCORES_CHUNK_BYTES = 1024 * 1024 * 1024
TRITON_MAX_PROGRAMS = 256
TRITON_INDEX_BLOCK = 256
TRITON_PREFILL_INDEX_BLOCK = 4096
TRITON_PREFILL_INDEX_MIN_TOKENS = 512
TRITON_PREFILL_MIN_TOKENS = 32
TRITON_PREFILL_POOL_TILE = 2048
TRITON_PACKED_CACHE_BYTES = 256 * 1024 * 1024
NEG_INF_SENTINEL = torch.finfo(torch.float32).min
# LI C8 pooled rows end with one little-endian FP32 scale.
LI_C8_SCALE_BYTES = 4
# Masked request ends must sort after every valid token offset.
QUERY_END_SENTINEL = torch.iinfo(torch.int32).max


@triton.jit(do_not_specialize=["requests", "pools_count", "pool_stride"])
def _c8_pack_pool_rows(
    cache,
    keys,
    scales,
    table,
    lengths,
    requests,
    pools_count,
    pool_stride,
    blocks_count,
    ROW_I16: tl.constexpr,
    HALF_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_SB: tl.constexpr,
    CACHE_ST: tl.constexpr,
    TABLE_SR: tl.constexpr,
    TABLE_SP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Unpack the pooled rows a request uses into contiguous BF16 keys."""
    chunks = tl.cdiv(pools_count, BLOCK)
    for tile in range(tl.program_id(0), requests * chunks, tl.num_programs(0)):
        request, chunk = tile // chunks, tile % chunks
        pools = chunk * BLOCK + tl.arange(0, BLOCK)
        in_range = pools < pools_count
        valid = in_range & (pools < tl.load(lengths + request))
        blocks = tl.load(table + request * TABLE_SR + (pools // CACHE_BLOCK) * TABLE_SP, valid, other=0)
        blocks = tl.minimum(tl.maximum(blocks, 0), blocks_count - 1)
        slot = (pools % CACHE_BLOCK) * CACHE_ST
        row = blocks * CACHE_SB + slot
        half = tl.arange(0, HALF_DIM)
        words = tl.load(cache + row[:, None] + half[None, :], valid[:, None], other=0)
        scale_low = tl.load(cache + row + HALF_DIM, valid, other=0).to(tl.int32) & 0xFFFF
        scale_high = tl.load(cache + row + HALF_DIM + 1, valid, other=0).to(tl.int32) & 0xFFFF
        scale = (scale_low | (scale_high << 16)).to(tl.float32, bitcast=True)
        # INT16 arithmetic shifts sign-extend each packed byte in one op. The
        # byte pair is then recombined as one INT32 word so the store runs at
        # unit stride instead of stride two. SCALE folds the row scale into the
        # keys, which is what a BF16 scorer needs and is equivalent to scaling
        # the score afterwards because the scale is positive.
        even = ((words << 8) >> 8).to(tl.float32)
        odd = (words >> 8).to(tl.float32)
        if SCALE:
            even = even * scale[:, None]
            odd = odd * scale[:, None]
        even = even.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        odd = odd.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        tl.store(
            keys + (request * pool_stride + pools[:, None]) * HALF_DIM + half[None, :],
            even | (odd << 16),
            in_range[:, None],
        )
        tl.store(scales + request * pool_stride + pools, scale, in_range)


@triton.jit(
    do_not_specialize=["rows", "row_lo", "row_hi", "max_pool_seq_len", "pool_stride", "chunks", "blocks", "tiles_total"]
)
def _c8_kpool_score_kernel(
    query,
    weights,
    keys,
    scales,
    positions,
    request_ends,
    pool_lens,
    block_rows,
    block_requests,
    output,
    rows,
    row_lo,
    row_hi,
    max_pool_seq_len,
    pool_stride,
    chunks,
    blocks,
    tiles_total,
    QUERY_STRIDE_T: tl.constexpr,
    QUERY_STRIDE_H: tl.constexpr,
    QUERY_STRIDE_D: tl.constexpr,
    WEIGHT_STRIDE_T: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    POOL: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    POOL_TILE: tl.constexpr,
    LOWEST: tl.constexpr,
):
    block_heads: tl.constexpr = ROW_BLOCK * NUM_HEADS
    lanes = tl.arange(0, block_heads)
    dims = tl.arange(0, HEAD_DIM)
    tile_rows = tl.arange(0, ROW_BLOCK)
    lane_row = lanes // NUM_HEADS
    lane_head = lanes % NUM_HEADS
    for tile in range(tl.program_id(0), tiles_total, tl.num_programs(0)):
        # Flat tile space: consecutive programs walk the same chunk across
        # row blocks, so the key tile they read is shared through the cache.
        block = tile % blocks
        chunk = tile // blocks
        row_base = tl.load(block_rows + block)
        request = tl.load(block_requests + block)
        rows_here = row_base + tile_rows
        limit = tl.minimum(tl.load(request_ends + request), row_hi)
        live = (rows_here >= row_lo) & (rows_here < limit)
        # Row masks make the Ascend backend materialise whole-tile predicates
        # and overflow UB, so rows clamp to a safe address instead; the store
        # below is the only place a row is dropped.
        safe_rows = tl.minimum(tl.maximum(rows_here, 0), rows - 1)
        lane_rows = row_base + lane_row
        safe_lane = tl.minimum(tl.maximum(lane_rows, 0), rows - 1)
        position = tl.load(positions + safe_rows).to(tl.int32)
        visible = tl.minimum((position + 1) // POOL, tl.load(pool_lens + request))
        pools = chunk * POOL_TILE + tl.arange(0, POOL_TILE)
        in_range = pools < max_pool_seq_len
        scores = tl.full((ROW_BLOCK, POOL_TILE), LOWEST, tl.float32)
        # The key rows are padded to whole tiles, so the load needs no mask: a
        # masked key tile makes the backend spill the whole predicate and
        # overflow UB. Pools past the visible range are masked out below.
        q = tl.load(
            query
            + safe_lane[:, None] * QUERY_STRIDE_T
            + lane_head[:, None] * QUERY_STRIDE_H
            + dims[None, :] * QUERY_STRIDE_D
        )
        key_rows = request * pool_stride + pools
        key = tl.load(keys + key_rows[None, :] * HEAD_DIM + dims[:, None])
        dots = tl.dot(q, key, tl.zeros((block_heads, POOL_TILE), tl.float32))
        relu = tl.maximum(dots, 0.0)
        weight = tl.load(weights + safe_lane * WEIGHT_STRIDE_T + lane_head * WEIGHT_STRIDE_H).to(tl.float32)
        # A one-hot selector keeps the head reduction on the cube; a vector sum
        # would serialise the block's rows.
        selector = tl.where(lane_row[None, :] == tile_rows[:, None], 1.0, 0.0) * weight[None, :]
        head_scores = tl.dot(selector, relu)
        scores = tl.where(
            pools[None, :] < visible[:, None],
            head_scores * tl.load(scales + key_rows)[None, :],
            LOWEST,
        )
        # Every pool of every row is written: top-k reads the rest of the row.
        scores = tl.where(scores == scores, tl.minimum(tl.maximum(scores, LOWEST), -LOWEST), LOWEST)
        tl.store(
            output + tl.maximum(tl.minimum(rows_here, row_hi - 1) - row_lo, 0)[:, None] * max_pool_seq_len
            + pools[None, :],
            scores,
            in_range[None, :] if ROW_BLOCK == 1 else in_range[None, :] & live[:, None],
        )


@triton.jit(do_not_specialize=["requests", "pools_count", "blocks_count"])
def _gather_pool_cache(
    cache,
    i16_cache,
    table,
    lengths,
    packed,
    i16_packed,
    requests,
    pools_count,
    blocks_count,
    CACHE_BLOCK: tl.constexpr,
    CACHE_SB: tl.constexpr,
    CACHE_ST: tl.constexpr,
    CACHE_SD: tl.constexpr,
    CACHE_SB_I16: tl.constexpr,
    CACHE_ST_I16: tl.constexpr,
    TABLE_SR: tl.constexpr,
    TABLE_SP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    ROW_I16: tl.constexpr,
    QUANT: tl.constexpr,
):
    chunks = tl.cdiv(pools_count, BLOCK)
    for tile in range(tl.program_id(0), requests * chunks, tl.num_programs(0)):
        request, chunk = tile // chunks, tile % chunks
        pools = chunk * BLOCK + tl.arange(0, BLOCK)
        valid = (pools < pools_count) & (pools < tl.load(lengths + request))
        blocks = tl.load(table + request * TABLE_SR + (pools // CACHE_BLOCK) * TABLE_SP, valid, other=0)
        blocks = tl.minimum(tl.maximum(blocks, 0), blocks_count - 1)
        if QUANT:
            # Copy the whole C8 row, scale included, through the INT16 views.
            half = tl.arange(0, HALF_DIM)
            src_row = blocks[:, None] * CACHE_SB_I16 + (pools % CACHE_BLOCK)[:, None] * CACHE_ST_I16
            dst_row = (request * pools_count + pools[:, None]) * ROW_I16
            words = tl.load(i16_cache + src_row + half[None, :], valid[:, None], other=0)
            tl.store(i16_packed + dst_row + half[None, :], words, (pools < pools_count)[:, None])
            # The row's FP32 scale occupies the two INT16 words after the row.
            pair = tl.arange(0, 2)
            scale_words = tl.load(i16_cache + src_row + HALF_DIM + pair[None, :], valid[:, None], other=0)
            tl.store(
                i16_packed + dst_row + HALF_DIM + pair[None, :],
                scale_words,
                (pools < pools_count)[:, None],
            )
        else:
            dims = tl.arange(0, HEAD_DIM)
            offsets = blocks[:, None] * CACHE_SB + (pools % CACHE_BLOCK)[:, None] * CACHE_ST
            keys = tl.load(cache + offsets + dims[None, :] * CACHE_SD, valid[:, None], other=0.0)
            tl.store(
                packed + (request * pools_count + pools[:, None]) * ROW_WIDTH + dims[None, :],
                keys,
                (pools < pools_count)[:, None],
            )


@triton.jit(do_not_specialize=["token_offset", "rows", "max_pool_seq_len", "num_reqs", "num_cache_blocks"])
def _glm5_next_lightning_indexer_score_kernel(
    query,
    weights,
    cache,
    i16_cache,
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
    cache_stride_b: tl.constexpr,
    cache_stride_t: tl.constexpr,
    cache_stride_d: tl.constexpr,
    cache_stride_b_i16: tl.constexpr,
    cache_stride_t_i16: tl.constexpr,
    table_stride_r: tl.constexpr,
    table_stride_p: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    QUERY_END_SENTINEL: tl.constexpr,
    REQ_POW2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    ROW_I16: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
    LOWEST: tl.constexpr,
    INDEX32: tl.constexpr,
    OUTER_POOL: tl.constexpr,
    PACKED_CACHE: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    QUANT: tl.constexpr,
    QUERY_STRIDE_T: tl.constexpr,
    QUERY_STRIDE_H: tl.constexpr,
    QUERY_STRIDE_D: tl.constexpr,
    WEIGHT_STRIDE_T: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    chunks = tl.cdiv(max_pool_seq_len, OUTER_POOL)
    for tile in range(tl.program_id(0), rows * chunks, tl.num_programs(0)):
        row, chunk = tile // chunks, tile % chunks
        token = row + token_offset
        requests = tl.arange(0, REQ_POW2)
        ends = tl.load(query_ends + requests, requests < num_reqs, other=QUERY_END_SENTINEL)
        request = tl.minimum(tl.sum((token >= ends).to(tl.int32)), num_reqs - 1)
        position = tl.load(positions + token).to(tl.int32)
        visible = tl.minimum((position + 1) // POOL, tl.load(pool_lens + request))
        live = token < tl.load(query_ends + num_reqs - 1)
        dims = tl.arange(0, HEAD_DIM)
        if QUANT:
            # Keep heads separate: CANN applies ReLU to each QK dot product
            # before the weighted head reduction. Collapsing Q first changes
            # pool selection when a dot product is negative.
            half = tl.arange(0, HALF_DIM)
            heads = tl.arange(0, BLOCK_HEADS)
            query_base = query + row * QUERY_STRIDE_T + heads[:, None] * QUERY_STRIDE_H
            query_even = tl.load(query_base + half[None, :] * 2 * QUERY_STRIDE_D, heads[:, None] < NUM_HEADS, 0)
            query_odd = tl.load(query_base + (half[None, :] * 2 + 1) * QUERY_STRIDE_D, heads[:, None] < NUM_HEADS, 0)
            head_weights = tl.load(weights + row * WEIGHT_STRIDE_T + heads * WEIGHT_STRIDE_H, heads < NUM_HEADS, 0)
        else:
            weighted_query = tl.load(query + row * HEAD_DIM + dims)
        sub_tiles = tl.cdiv(tl.minimum(max_pool_seq_len - chunk * OUTER_POOL, OUTER_POOL), BLOCK_POOL)
        for sub in range(sub_tiles):
            pool_start = chunk * OUTER_POOL + sub * BLOCK_POOL
            pools = pool_start + tl.arange(0, BLOCK_POOL)
            scores = tl.full((BLOCK_POOL,), LOWEST, tl.float32)
            if live & (pool_start < visible):
                valid = (pools < max_pool_seq_len) & (pools < visible)
                if QUANT:
                    if PACKED_CACHE:
                        row_i16 = (request * max_pool_seq_len + pools) * ROW_I16
                    else:
                        pages = pools // CACHE_BLOCK
                        blocks = tl.load(
                            block_table + request * table_stride_r + pages * table_stride_p,
                            valid,
                            other=0,
                        )
                        blocks = tl.minimum(tl.maximum(blocks, 0), num_cache_blocks - 1)
                        row_i16 = blocks * cache_stride_b_i16 + (pools % CACHE_BLOCK) * cache_stride_t_i16
                    words = tl.load(i16_cache + row_i16[:, None] + half[None, :], valid[:, None], other=0)
                    words = words.to(tl.int32) & 0xFFFF
                    low = words & 0xFF
                    high = (words >> 8) & 0xFF
                    # Two's-complement sign extension of each quantized byte.
                    low = tl.where(low > 127, low - 256, low).to(tl.float32)
                    high = tl.where(high > 127, high - 256, high).to(tl.float32)
                    dots = tl.dot(query_even, tl.trans(low.to(query_even.dtype)))
                    dots += tl.dot(query_odd, tl.trans(high.to(query_odd.dtype)))
                    scores = tl.sum(tl.maximum(dots, 0.0) * head_weights[:, None].to(tl.float32), axis=0)
                    # The row's FP32 scale occupies the two words after the row.
                    scale_low = tl.load(i16_cache + row_i16 + HALF_DIM, valid, other=0).to(tl.int32) & 0xFFFF
                    scale_high = tl.load(i16_cache + row_i16 + HALF_DIM + 1, valid, other=0).to(tl.int32) & 0xFFFF
                    scores = scores * (scale_low | (scale_high << 16)).to(tl.float32, bitcast=True)
                else:
                    if PACKED_CACHE:
                        k = tl.load(
                            cache + (request * max_pool_seq_len + pools[:, None]) * ROW_WIDTH + dims[None, :],
                            valid[:, None],
                            other=0.0,
                        )
                    elif INDEX32:
                        pages = pools // CACHE_BLOCK
                        blocks = tl.load(
                            block_table + request * table_stride_r + pages * table_stride_p,
                            valid,
                            other=0,
                        )
                        blocks = tl.minimum(tl.maximum(blocks, 0), num_cache_blocks - 1)
                        offsets = blocks[:, None] * cache_stride_b + (pools % CACHE_BLOCK)[:, None] * cache_stride_t
                        k = tl.load(cache + offsets + dims[None, :] * cache_stride_d, valid[:, None], other=0.0)
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
                            other=0.0,
                        )
                    scores = tl.sum(k.to(tl.float32) * weighted_query[None, :], axis=1)
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


_C8_BUFFERS: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_C8_PAGE_TABLES: dict[tuple, torch.Tensor] = {}
_C8_POOL_BOUNDS: dict[tuple, torch.Tensor] = {}


def _c8_buffers(
    device: torch.device, num_reqs: int, pools_count: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous BF16 key rows and FP32 scales for one request batch."""
    key = (device, num_reqs, pools_count, head_dim)
    buffers = _C8_BUFFERS.get(key)
    if buffers is None:
        if len(_C8_BUFFERS) > 16:
            _C8_BUFFERS.clear()
        buffers = (
            torch.empty((num_reqs * pools_count, head_dim // 2), dtype=torch.int32, device=device),
            torch.empty((num_reqs * pools_count,), dtype=torch.float32, device=device),
        )
        _C8_BUFFERS[key] = buffers
    return buffers


def _c8_block_map(
    cum_query_lens: torch.Tensor, num_tokens: int, row_block: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Map each C8 score block to a request and to the token it starts at.

    Several rows share a program, so a block must stay inside one request: the
    key rows of a request come from its own packed region.
    """
    num_reqs = cum_query_lens.numel()
    starts = torch.cat((cum_query_lens.new_zeros(1), cum_query_lens[:-1]))
    blocks_per_req = (cum_query_lens - starts + (row_block - 1)) // row_block
    cum_blocks = torch.cat((cum_query_lens.new_zeros(1), torch.cumsum(blocks_per_req, 0)))
    blocks = num_tokens if row_block == 1 else triton.cdiv(num_tokens, row_block) + num_reqs
    index = torch.arange(blocks, dtype=cum_query_lens.dtype, device=cum_query_lens.device)
    # The map stays on the device: capture needs static shapes and no sync.
    request = (index[:, None] >= cum_blocks[None, :]).sum(dim=1) - 1
    dead = request >= num_reqs
    request = request.clamp_max(num_reqs - 1)
    first_row = starts[request] + (index - cum_blocks[request]) * row_block
    first_row = torch.where(dead, torch.full_like(first_row, num_tokens), first_row)
    return first_row.to(torch.int32), request.to(torch.int32), blocks


def _c8_page_table(block_table: torch.Tensor, num_reqs: int, pages: int) -> torch.Tensor:
    """Identity page table for per-request packed regions: page r * pages + j."""
    width = max(block_table.shape[1], pages)
    key = (num_reqs, width, pages, block_table.device)
    table = _C8_PAGE_TABLES.get(key)
    if table is None:
        if len(_C8_PAGE_TABLES) > 16:
            _C8_PAGE_TABLES.clear()
        rows = torch.arange(num_reqs, dtype=torch.int32, device=block_table.device)[:, None] * pages
        cols = torch.arange(width, dtype=torch.int32, device=block_table.device)[None, :]
        table = (rows + torch.clamp(cols, max=pages - 1)).contiguous()
        _C8_PAGE_TABLES[key] = table
    return table


def _c8_pool_bounds(device: torch.device, num_reqs: int, pools: int) -> torch.Tensor:
    """Per-request pool count of a fully packed region, for the unpacker."""
    key = (device, num_reqs, pools)
    bounds = _C8_POOL_BOUNDS.get(key)
    if bounds is None:
        if len(_C8_POOL_BOUNDS) > 16:
            _C8_POOL_BOUNDS.clear()
        bounds = torch.full((num_reqs,), pools, dtype=torch.int32, device=device)
        _C8_POOL_BOUNDS[key] = bounds
    return bounds


def glm5_next_c8_dequant_pool_key(
    indexer_cache: torch.Tensor,
    indexer_block_table: torch.Tensor,
    cum_query_lens: torch.Tensor,
    max_pool_seq_len: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack the live LI C8 pooled rows into a BF16 PA_BBND cache.

    Returns the cache as ``[requests * pages, cache_block, 1, head_dim]`` BF16
    plus the identity page table that maps a request's pools onto its own packed
    region, so a BF16 scorer can be called unchanged. The per-row FP32 scale is
    folded into the keys.
    """
    if indexer_cache.dtype != torch.int8:
        raise TypeError(f"GLM KPool LI C8 dequantisation needs int8 rows, got {indexer_cache.dtype}.")
    row_width = indexer_cache.shape[3]
    if row_width != head_dim + LI_C8_SCALE_BYTES:
        raise ValueError(f"GLM KPool cache row width {row_width} does not match head_dim {head_dim}.")
    num_reqs = cum_query_lens.numel()
    capacity = indexer_cache.shape[0] * indexer_cache.shape[1]
    pool_stride = triton.cdiv(min(max_pool_seq_len, capacity), TRITON_C8_POOL_TILE) * TRITON_C8_POOL_TILE
    pool_stride = max(pool_stride, TRITON_C8_POOL_TILE)
    if num_reqs * pool_stride * head_dim * 2 > TRITON_PACKED_CACHE_BYTES:
        raise ValueError(
            f"GLM KPool LI C8 BF16 working set needs {num_reqs * pool_stride * head_dim * 2} bytes, "
            f"above the {TRITON_PACKED_CACHE_BYTES}-byte budget."
        )
    keys, scales = _c8_buffers(indexer_cache.device, num_reqs, pool_stride, head_dim)
    _c8_pack_pool_rows[
        (min(num_reqs * triton.cdiv(pool_stride, TRITON_C8_UNPACK_BLOCK), TRITON_MAX_PROGRAMS),)
    ](
        indexer_cache.view(torch.int16),
        keys,
        scales,
        indexer_block_table,
        _c8_pool_bounds(indexer_cache.device, num_reqs, pool_stride),
        num_reqs,
        pool_stride,
        pool_stride,
        indexer_cache.shape[0],
        row_width // 2,
        head_dim // 2,
        True,
        indexer_cache.shape[1],
        indexer_cache.stride(0) // 2,
        indexer_cache.stride(1) // 2,
        *indexer_block_table.stride(),
        TRITON_C8_UNPACK_BLOCK,
    )
    pages = pool_stride // TRITON_C8_POOL_TILE
    pool_key = keys.view(torch.bfloat16).view(num_reqs * pages, TRITON_C8_POOL_TILE, 1, head_dim)
    return pool_key, _c8_page_table(indexer_block_table, num_reqs, pages)


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
    allow_cache_packing: bool = True,
) -> torch.Tensor:
    """Select pools, expand tokens and optionally write the final SFA buffer."""
    num_tokens, _, head_dim = query.shape
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
    # LI C8 rows pack head_dim quantized bytes plus one FP32 scale; the cache
    # view's last extent is the physical row width in that case.
    quant = indexer_cache.dtype in (torch.int8, torch.float8_e4m3fn)
    row_width = indexer_cache.shape[3]
    if row_width != head_dim + (LI_C8_SCALE_BYTES if quant else 0):
        raise ValueError(
            f"GLM KPool cache row width {row_width} does not match head_dim {head_dim} for {indexer_cache.dtype}."
        )
    if quant:
        # The score kernel reads C8 rows as INT16 word pairs, so every row
        # address and the trailing scale must be 2-byte aligned.
        if indexer_cache.dtype != torch.int8:
            raise NotImplementedError(
                f"GLM KPool LI C8 scoring supports int8 pooled rows only, got {indexer_cache.dtype}."
            )
        if head_dim % 2 or row_width % 2 or any(stride % 2 for stride in indexer_cache.stride()[:3]):
            raise ValueError(
                f"GLM KPool LI C8 requires 2-byte aligned rows: head_dim={head_dim}, row_width={row_width}, "
                f"strides={tuple(indexer_cache.stride())}."
            )
        if not index32:
            raise NotImplementedError("GLM KPool LI C8 requires a cache addressable with int32 offsets.")
    page_span = (indexer_cache.shape[1] - 1) * indexer_cache.stride(1) + (row_width - 1) * indexer_cache.stride(3)
    if page_span > torch.iinfo(torch.int32).max:
        raise ValueError("GLM KPool cache page offsets must fit in int32.")
    if not index32 and indexer_cache.shape[1] != next_power_of_2(indexer_cache.shape[1]):
        raise ValueError("GLM KPool large-cache pages require a power-of-two block size.")
    selected = min(index_topk // index_kpool, max_pool_seq_len)
    use_c8_pack = quant and selected > 0 and num_tokens >= TRITON_PREFILL_MIN_TOKENS
    pool_stride = max_pool_seq_len
    if use_c8_pack:
        # A packed batch unpacks the pooled rows its requests use once and
        # scores whole tiles, which is what makes long-context prefill viable.
        # The cache is sized for one pooled row per token while one row covers
        # index_kpool tokens, so only this prefix can ever be live. The bound
        # arrives from host-side sequence lengths: reading a device maximum
        # back here would drain the whole queued pipeline once per layer.
        pool_stride = triton.cdiv(max_pool_seq_len, TRITON_C8_POOL_TILE) * TRITON_C8_POOL_TILE
        pool_stride = max(pool_stride, TRITON_C8_POOL_TILE)
        if cum_query_lens.numel() * pool_stride * head_dim * 2 > TRITON_PACKED_CACHE_BYTES:
            raise ValueError(
                f"GLM KPool LI C8 key rows need {cum_query_lens.numel() * pool_stride * head_dim * 2} bytes, "
                f"above the {TRITON_PACKED_CACHE_BYTES}-byte budget."
            )
        keys, scales = _c8_buffers(query.device, cum_query_lens.numel(), pool_stride, head_dim)
        _c8_pack_pool_rows[
            (
                min(
                    cum_query_lens.numel() * triton.cdiv(max_pool_seq_len, TRITON_C8_UNPACK_BLOCK),
                    TRITON_MAX_PROGRAMS,
                ),
            )
        ](
            indexer_cache.view(torch.int16),
            keys,
            scales,
            indexer_block_table,
            indexer_seq_lens,
            cum_query_lens.numel(),
            pool_stride,
            pool_stride,
            indexer_cache.shape[0],
            row_width // 2,
            head_dim // 2,
            False,
            indexer_cache.shape[1],
            indexer_cache.stride(0) // 2,
            indexer_cache.stride(1) // 2,
            *indexer_block_table.stride(),
            TRITON_C8_UNPACK_BLOCK,
        )
        # With few heads per rank the key tile dominates, so a wide row block
        # amortises it over more rows; wide blocks only fit UB at small heads.
        row_block = TRITON_C8_ROWS_PER_BLOCK if query.shape[1] > 8 else TRITON_C8_ROWS_PER_BLOCK_WIDE
        block_rows, block_requests, blocks = _c8_block_map(cum_query_lens, num_tokens, row_block)
        chunks = triton.cdiv(pool_stride, TRITON_C8_POOL_TILE)
        tiles_total = blocks * chunks
        cache = keys.view(torch.bfloat16)
    packed_cache = (
        allow_cache_packing
        and not use_c8_pack
        and num_tokens >= TRITON_PREFILL_MIN_TOKENS
        and num_tokens > cum_query_lens.numel()
        and selected > 0
        and index32
        and cum_query_lens.numel() * max_pool_seq_len * row_width * indexer_cache.element_size()
        <= TRITON_PACKED_CACHE_BYTES
    )
    if packed_cache:
        cache = torch.empty(
            (cum_query_lens.numel(), max_pool_seq_len, 1, row_width),
            dtype=indexer_cache.dtype,
            device=indexer_cache.device,
        )
        _gather_pool_cache[
            (min(cum_query_lens.numel() * triton.cdiv(max_pool_seq_len, TRITON_POOL_CHUNK_SIZE), TRITON_MAX_PROGRAMS),)
        ](
            indexer_cache,
            indexer_cache.view(torch.int16) if quant else indexer_cache,
            indexer_block_table,
            indexer_seq_lens,
            cache,
            cache.view(torch.int16) if quant else cache,
            cum_query_lens.numel(),
            max_pool_seq_len,
            indexer_cache.shape[0],
            indexer_cache.shape[1],
            indexer_cache.stride(0),
            indexer_cache.stride(1),
            indexer_cache.stride(3),
            indexer_cache.stride(0) // 2,
            indexer_cache.stride(1) // 2,
            *indexer_block_table.stride(),
            head_dim,
            head_dim // 2,
            TRITON_POOL_CHUNK_SIZE,
            row_width,
            row_width // 2,
            quant,
        )
    elif not use_c8_pack:
        cache = indexer_cache
    score_width = pool_stride if use_c8_pack else max_pool_seq_len
    token_chunk = max(1, TRITON_SCORES_CHUNK_BYTES // max(4, score_width * 4))
    for start in range(0, num_tokens, token_chunk):
        rows = min(token_chunk, num_tokens - start)
        if selected:
            scores = torch.empty((rows, score_width), dtype=torch.float32, device=query.device)
            if use_c8_pack:
                # Every row needs the sentinel past its own pool prefix: the
                # emitted tiles are exactly what its prefix needs.
                scores.fill_(NEG_INF_SENTINEL)
                _c8_kpool_score_kernel[(min(tiles_total, TRITON_MAX_PROGRAMS),)](
                    query,
                    weights,
                    cache,
                    scales,
                    positions,
                    cum_query_lens,
                    indexer_seq_lens,
                    block_rows,
                    block_requests,
                    scores,
                    num_tokens,
                    start,
                    start + rows,
                    pool_stride,
                    pool_stride,
                    chunks,
                    blocks,
                    tiles_total,
                    query.stride(0),
                    query.stride(1),
                    query.stride(2),
                    weights.stride(0),
                    weights.stride(1),
                    head_dim,
                    query.shape[1],
                    index_kpool,
                    row_block,
                    TRITON_C8_POOL_TILE,
                    NEG_INF_SENTINEL,
                )
            else:
                query_rows = query[start : start + rows]
                weight_rows = weights[start : start + rows]
                # The legacy BF16 helper keeps its existing formula; production
                # BF16 selection uses CANN. INT8 must match CANN's per-head ReLU.
                score_query = (
                    query_rows
                    if quant
                    else ((query_rows.float() * weight_rows.float().unsqueeze(-1)).sum(dim=1).contiguous())
                )
                if index32:
                    block_pool = TRITON_POOL_CHUNK_SIZE_C8 if quant else TRITON_POOL_CHUNK_SIZE
                else:
                    block_pool = indexer_cache.shape[1]
                outer_pool = (
                    max(TRITON_PREFILL_POOL_TILE, block_pool) if rows >= TRITON_PREFILL_MIN_TOKENS else block_pool
                )
                chunks = triton.cdiv(max_pool_seq_len, outer_pool)
                _glm5_next_lightning_indexer_score_kernel[(min(rows * chunks, TRITON_MAX_PROGRAMS),)](
                    score_query,
                    weight_rows,
                    cache,
                    cache.view(torch.int16) if quant else cache,
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
                    0 if packed_cache else cache.stride(0),
                    cache.stride(1),
                    cache.stride(3),
                    0 if packed_cache else cache.stride(0) // 2,
                    cache.stride(1) // 2,
                    *indexer_block_table.stride(),
                    indexer_cache.shape[1],
                    QUERY_END_SENTINEL,
                    next_power_of_2(cum_query_lens.numel()),
                    head_dim,
                    head_dim // 2,
                    row_width // 2,
                    index_kpool,
                    block_pool,
                    NEG_INF_SENTINEL,
                    index32,
                    outer_pool,
                    packed_cache,
                    row_width,
                    quant,
                    query.stride(0),
                    query.stride(1),
                    query.stride(2),
                    weights.stride(0),
                    weights.stride(1),
                    query.shape[1],
                    max(16, next_power_of_2(query.shape[1])),
                )
            # Order inside the SFA buffer is irrelevant (the sparse attention
            # reads it as a set), so skip the extra sort of the selected ids.
            topk_scores, pool_ids = torch.topk(scores, selected, dim=1, sorted=False)
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
