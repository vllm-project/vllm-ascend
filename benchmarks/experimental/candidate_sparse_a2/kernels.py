# SPDX-License-Identifier: Apache-2.0
"""A2 experiment: Triton gather + unchanged native INT8 candidate QLI.

No model dispatch is modified. Buffers are explicit and owned by one forward.
The four C1 consumers can reuse a pack, but not their Q, weights or results.
"""

import math
from dataclasses import dataclass

import torch
from vllm.triton_utils import tl, triton

BLOCK_SIZE = 8
GATHER_ROWS = 64
INDEX_WIDTH = 128
INVALID_BLOCK = 1 << 24


@triton.jit
def _plan(
    candidates,
    positions,
    query_starts,
    cache_lengths,
    sorted_blocks,
    mapped_candidates,
    prefix_counts,
    packed_lengths,
    request_ids,
    C: tl.constexpr,
    CP2: tl.constexpr,
    REQUESTS: tl.constexpr,
    RP2: tl.constexpr,
    PREFIX: tl.constexpr,
    RATIO: tl.constexpr,
    INVALID: tl.constexpr,
):
    row = tl.program_id(0)
    requests = tl.arange(0, RP2)
    ends = tl.load(query_starts + requests + 1, requests < REQUESTS, other=2147483647)
    request = tl.minimum(tl.sum((row >= ends).to(tl.int32)), REQUESTS - 1)
    live = row < tl.load(query_starts + REQUESTS)
    visible = tl.minimum((tl.load(positions + row) + 1) // RATIO, tl.load(cache_lengths + request))
    visible = tl.where(live, tl.maximum(visible, 0), 0)
    columns = tl.arange(0, CP2)
    original = tl.load(candidates + row * C + columns, columns < C, other=-1)
    valid = (original >= 0) & (original < tl.cdiv(visible, BLOCK_SIZE))
    # All real block IDs are < 2**24 in the model's supported context range.
    values = tl.where(valid, original, INVALID).to(tl.float32)
    # Same row-wise Ascend sort interface used by prepare_indexer_indices.
    values = tl.extra.cann.extension.sort(values[None, :], dim=1, descending=False)
    values = tl.reshape(values, (CP2,)).to(tl.int32)
    is_valid = values < INVALID
    prefix_candidates = tl.sum((is_valid & (values < PREFIX)).to(tl.int32))
    count = tl.sum(is_valid.to(tl.int32))
    prefix_blocks = tl.minimum(PREFIX, tl.cdiv(visible, BLOCK_SIZE))
    mapped = tl.where(values < PREFIX, values, PREFIX + columns - prefix_candidates)
    mapped = tl.where(is_valid, mapped, -1)
    tl.store(sorted_blocks + row * C + columns, values, columns < C)
    tl.store(mapped_candidates + row * C + columns, mapped, columns < C)
    total_blocks = prefix_blocks + count - prefix_candidates
    last_visible = tl.cdiv(visible, BLOCK_SIZE) - 1
    has_partial = ((visible % BLOCK_SIZE) != 0) & (
        (last_visible < PREFIX) | (tl.sum((is_valid & (values == last_visible)).to(tl.int32)) > 0)
    )
    length = total_blocks * BLOCK_SIZE - tl.where(has_partial, BLOCK_SIZE - visible % BLOCK_SIZE, 0)
    tl.store(prefix_counts + row, prefix_candidates)
    tl.store(packed_lengths + row, length)
    tl.store(request_ids + row, request)


@triton.jit
def _gather(
    key,
    scale,
    table,
    sorted_blocks,
    prefix_counts,
    lengths,
    requests,
    packed_key,
    packed_scale,
    C: tl.constexpr,
    PREFIX: tl.constexpr,
    CAPACITY: tl.constexpr,
    PAGE: tl.constexpr,
    KEY_SB: tl.constexpr,
    KEY_ST: tl.constexpr,
    KEY_SD: tl.constexpr,
    SCALE_SB: tl.constexpr,
    SCALE_ST: tl.constexpr,
    TABLE_SB: tl.constexpr,
    TABLE_ST: tl.constexpr,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(1)
    offsets = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    compact_block = offsets // BLOCK_SIZE
    valid = offsets < tl.load(lengths + row)
    large_index = compact_block - PREFIX + tl.load(prefix_counts + row)
    large_block = tl.load(
        sorted_blocks + row * C + large_index,
        valid & (compact_block >= PREFIX),
        other=0,
    )
    logical_block = tl.where(compact_block < PREFIX, compact_block, large_block)
    logical = logical_block * BLOCK_SIZE + offsets % BLOCK_SIZE
    request = tl.load(requests + row)
    physical = tl.load(table + request * TABLE_SB + (logical // PAGE) * TABLE_ST, valid, other=0).to(tl.int64)
    dims = tl.arange(0, WIDTH)
    source = physical[:, None] * KEY_SB + (logical % PAGE)[:, None] * KEY_ST + dims[None, :] * KEY_SD
    data = tl.load(key + source, valid[:, None], other=0)
    scales = tl.load(scale + physical * SCALE_SB + (logical % PAGE) * SCALE_ST, valid, other=0)
    # Deterministically initialize padding; the native sequence length excludes it.
    tl.store(
        packed_key + (row.to(tl.int64) * CAPACITY + offsets[:, None]) * WIDTH + dims[None, :],
        data,
        (offsets < CAPACITY)[:, None],
    )
    tl.store(packed_scale + row * CAPACITY + offsets, scales, offsets < CAPACITY)


@triton.jit
def _restore(
    selected,
    sorted_blocks,
    prefix_counts,
    lengths,
    output,
    C: tl.constexpr,
    TOPK: tl.constexpr,
    KP2: tl.constexpr,
    PREFIX: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, KP2)
    index = tl.load(selected + row * TOPK + columns, columns < TOPK, other=-1)
    valid = (index >= 0) & (index < tl.load(lengths + row))
    block = index // BLOCK_SIZE
    large_index = block - PREFIX + tl.load(prefix_counts + row)
    original = tl.load(sorted_blocks + row * C + large_index, valid & (block >= PREFIX), other=0)
    original = tl.where(block < PREFIX, block, original)
    value = tl.where(valid, original * BLOCK_SIZE + index % BLOCK_SIZE, -1)
    tl.store(output + row * TOPK + columns, value, columns < TOPK)


@dataclass
class PackedCandidates:
    """Fixed-address scratch, allocated outside timed runs / graph capture."""

    key: torch.Tensor
    scale: torch.Tensor
    sorted_blocks: torch.Tensor
    candidates: torch.Tensor
    prefix_counts: torch.Tensor
    lengths: torch.Tensor
    requests: torch.Tensor
    table: torch.Tensor
    query_starts: torch.Tensor
    capacity: int
    topk: int


def allocate(rows, candidate_count, topk, page_size, device):
    prefix = math.ceil(topk / BLOCK_SIZE)
    capacity = math.ceil((candidate_count + prefix) * BLOCK_SIZE / page_size) * page_size
    pages = capacity // page_size
    ints = {"dtype": torch.int32, "device": device}
    return PackedCandidates(
        torch.empty((rows * pages, page_size, 1, INDEX_WIDTH), dtype=torch.int8, device=device),
        torch.empty((rows * pages, page_size, 1), dtype=torch.float16, device=device),
        torch.empty((rows, candidate_count), **ints),
        torch.empty((rows, 1, candidate_count), **ints),
        torch.empty(rows, **ints),
        torch.empty(rows, **ints),
        torch.empty(rows, **ints),
        torch.arange(rows * pages, **ints).reshape(rows, pages),
        torch.arange(rows + 1, **ints),
        capacity,
        topk,
    )


def pack(
    key,
    scale,
    table,
    candidates,
    positions,
    query_starts,
    cache_lengths,
    ratio,
    buffers,
):
    """TND rows, PA_BBND INT8 K, FP16 scales, unique source candidate IDs.

    Cache columns are contiguous; page strides may include Hybrid KV storage.
    Positions and original request boundaries determine per-query visibility.
    """
    rows, _, count = candidates.shape
    requests = cache_lengths.shape[0]
    prefix = math.ceil(buffers.topk / BLOCK_SIZE)
    _plan[(rows,)](
        candidates,
        positions,
        query_starts,
        cache_lengths,
        buffers.sorted_blocks,
        buffers.candidates,
        buffers.prefix_counts,
        buffers.lengths,
        buffers.requests,
        count,
        triton.next_power_of_2(count),
        requests,
        triton.next_power_of_2(requests),
        prefix,
        ratio,
        INVALID_BLOCK,
        multibuffer=False,
    )
    _gather[(triton.cdiv(buffers.capacity, GATHER_ROWS), rows)](
        key,
        scale,
        table,
        buffers.sorted_blocks,
        buffers.prefix_counts,
        buffers.lengths,
        buffers.requests,
        buffers.key,
        buffers.scale,
        count,
        prefix,
        buffers.capacity,
        key.shape[1],
        key.stride(0),
        key.stride(1),
        key.stride(3),
        scale.stride(0),
        scale.stride(1),
        table.stride(0),
        table.stride(1),
        GATHER_ROWS,
        INDEX_WIDTH,
        multibuffer=False,
    )


def metadata(query, buffers):
    """AICPU schedule generation is intentionally included in full-path timing."""
    return torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        query.shape[1],
        1,
        query.shape[2],
        buffers.topk,
        2,
        cu_seqlens_q=buffers.query_starts,
        seqused_k=buffers.lengths,
        batch_size=query.shape[0],
        max_seqlen_q=1,
        max_seqlen_k=buffers.capacity,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
        device=str(query.device),
    )


def score(query, weights, query_scale, op_metadata, buffers, output):
    """Same native candidate scoring; map its result back to logical positions."""
    selected, _, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
        query,
        buffers.key,
        weights,
        query_scale,
        buffers.scale,
        buffers.topk,
        2,
        cu_seqlens_q=buffers.query_starts,
        seqused_k=buffers.lengths,
        block_table=buffers.table,
        metadata=op_metadata,
        max_seqlen_q=1,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
        candidate_topk_index=buffers.candidates,
        candidate_mode=2,
        candidate_topk_blocks=buffers.candidates.shape[-1],
        candidate_block_size=BLOCK_SIZE,
    )
    _restore[(query.shape[0],)](
        selected,
        buffers.sorted_blocks,
        buffers.prefix_counts,
        buffers.lengths,
        output,
        buffers.candidates.shape[-1],
        buffers.topk,
        triton.next_power_of_2(buffers.topk),
        math.ceil(buffers.topk / BLOCK_SIZE),
        multibuffer=False,
    )
    return output
