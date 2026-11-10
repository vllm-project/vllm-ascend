# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["src_k", "src_v", "dst_k", "dst_v"])
def build_d2h_descriptors(
    slots,
    sources,
    destinations,
    lengths,
    count,
    src_k,
    src_v,
    dst_k,
    dst_v,
    ROWS: tl.constexpr,
    CAPACITY: tl.constexpr,
    K_BYTES: tl.constexpr,
    V_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tile * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < ROWS
    slot = tl.load(slots + rows, in_range, other=-1).to(tl.int64)
    valid = (slot >= 0) & (slot < CAPACITY)
    slot = tl.where(valid, slot, 0)
    tl.store(sources + rows, src_k + rows.to(tl.int64) * K_BYTES, in_range)
    tl.store(sources + ROWS + rows, src_v + rows.to(tl.int64) * V_BYTES, in_range)
    tl.store(destinations + rows, dst_k + slot * K_BYTES, in_range)
    tl.store(destinations + ROWS + rows, dst_v + slot * V_BYTES, in_range)
    tl.store(lengths + rows, tl.where(valid, K_BYTES, 0), in_range)
    tl.store(lengths + ROWS + rows, tl.where(valid, V_BYTES, 0), in_range)
    if tile == 0:
        tl.store(count, 2 * ROWS)


@triton.jit
def write_tail(
    slots,
    tail_rows,
    k,
    v,
    cache_k,
    cache_v,
    STRIDE: tl.constexpr,
    HOT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K_WIDTH: tl.constexpr,
    V_WIDTH: tl.constexpr,
    K_TILE: tl.constexpr,
    V_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    host_slot = tl.load(slots + row)
    tail_slot = tl.load(tail_rows + 2 * row)
    valid = (tail_slot >= 0) & (host_slot >= 0)
    tail_pos = tl.maximum(tail_slot, 0) * STRIDE + HOT + tl.maximum(host_slot, 0) % BLOCK_SIZE
    ki = tl.arange(0, K_TILE)
    vi = tl.arange(0, V_TILE)
    kval = tl.load(k + row * K_WIDTH + ki, ki < K_WIDTH, other=0)
    vval = tl.load(v + row * V_WIDTH + vi, vi < V_WIDTH, other=0)
    tl.store(cache_k + tail_pos * K_WIDTH + ki, kval, valid & (ki < K_WIDTH))
    tl.store(cache_v + tail_pos * V_WIDTH + vi, vval, valid & (vi < V_WIDTH))


@triton.jit(do_not_specialize=["src_k", "src_v", "dst_k", "dst_v"])
def build_tail_flush_descriptors(
    slots,
    tail_rows,
    sources,
    destinations,
    lengths,
    count,
    src_k,
    src_v,
    dst_k,
    dst_v,
    ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    HOT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K_BYTES: tl.constexpr,
    V_BYTES: tl.constexpr,
    TILE: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tile * TILE + tl.arange(0, TILE)
    mask = row < ROWS
    host_slot = tl.load(slots + row, mask, other=-1).to(tl.int64)
    tail_slot = tl.load(tail_rows + 2 * row, mask, other=-1)
    first = tl.load(tail_rows + 2 * row + 1, mask, other=0)
    valid = (tail_slot >= 0) & (host_slot >= 0) & (host_slot % BLOCK_SIZE == BLOCK_SIZE - 1)
    first = tl.maximum(first, 0)
    tail_base = tl.maximum(tail_slot, 0) * STRIDE + HOT + first
    host_base = tl.maximum(host_slot, 0) // BLOCK_SIZE * BLOCK_SIZE + first
    tl.store(sources + row, src_k + tail_base * K_BYTES, mask)
    tl.store(sources + ROWS + row, src_v + tail_base * V_BYTES, mask)
    tl.store(destinations + row, dst_k + host_base * K_BYTES, mask)
    tl.store(destinations + ROWS + row, dst_v + host_base * V_BYTES, mask)
    tl.store(lengths + row, tl.where(valid, (BLOCK_SIZE - first) * K_BYTES, 0), mask)
    tl.store(lengths + ROWS + row, tl.where(valid, (BLOCK_SIZE - first) * V_BYTES, 0), mask)
    if tile == 0:
        tl.store(count, 2 * ROWS)


@triton.jit
def map_tail_sources(
    topk,
    blocks,
    lengths,
    requests,
    tails,
    host_slots,
    host_valid,
    tail_indices,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COLUMN_STRIDE: tl.constexpr,
    HOST_BLOCKS: tl.constexpr,
    HOT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    mask = col < TOPK
    index = row * TOPK + col
    logical = tl.load(topk + index, mask, other=-1).to(tl.int64)
    length = tl.load(lengths + row).to(tl.int64)
    request = tl.load(requests + row)
    tail_row = tl.load(tails + row * 2)
    tail_first = tl.load(tails + row * 2 + 1)
    block = tl.maximum(logical, 0) // BLOCK_SIZE
    offset = tl.maximum(logical, 0) % BLOCK_SIZE
    physical = tl.load(blocks + row * TABLE_ROW_STRIDE + tl.minimum(block, BLOCK_COLUMNS - 1) * TABLE_COLUMN_STRIDE).to(
        tl.int64
    )
    valid = (request != 0) & (logical >= 0) & (logical < length) & (block < BLOCK_COLUMNS)
    valid = valid & (physical >= 0) & (physical < HOST_BLOCKS)
    in_tail = valid & (tail_row >= 0) & (block == (length - 1) // BLOCK_SIZE) & (offset >= tail_first)
    source = tl.minimum(tl.maximum(physical, 0), HOST_BLOCKS - 1) * BLOCK_SIZE + offset
    tl.store(host_slots + index, source, mask)
    tl.store(host_valid + index, valid & ~in_tail, mask)
    tl.store(tail_indices + index, tl.where(in_tail, HOT + offset, -1), mask)
