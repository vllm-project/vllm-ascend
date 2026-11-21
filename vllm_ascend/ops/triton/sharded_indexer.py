# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate transport and causal metadata for a token-sharded indexer."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.sparse_index_remap import _a5_sort


@triton.jit
def _pack_candidates(
    indices,
    scores,
    packed,
    index_row_stride: tl.constexpr,
    score_row_stride: tl.constexpr,
    index_col_stride: tl.constexpr,
    score_col_stride: tl.constexpr,
    K: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    local = tl.load(indices + row * index_row_stride + col * index_col_stride, col < K, -1)
    score = tl.load(scores + row * score_row_stride + col * score_col_stride, col < K, -float("inf"))
    valid = (local >= 0) & (tl.abs(score) < float("inf"))
    global_id = local // INTERLEAVE * (WORLD * INTERLEAVE) + RANK * INTERLEAVE + local % INTERLEAVE
    global_id = tl.where(valid, global_id, -1)
    score = tl.where(valid, score, -float("inf"))
    # Interleave in UB so the transport buffer uses one contiguous GM store.
    pair = tl.reshape(tl.join(global_id, score.to(tl.int32, bitcast=True)), (2 * BLOCK,))
    lanes = tl.program_id(1) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(packed + row * K * 2 + lanes, pair, lanes < 2 * K)


@triton.jit
def _unpack_candidates(
    packed, id_keys, scores, ROWS: tl.constexpr, K: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    tiles = tl.cdiv(K, BLOCK)
    rank = tl.program_id(1) // tiles
    tile = tl.program_id(1) - rank * tiles
    col = tile * BLOCK + tl.arange(0, BLOCK)
    lanes = tile * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    source = (rank * ROWS + row) * (2 * K) + lanes
    pair = tl.load(packed + source, lanes < K * 2, 0)
    idx, bits = tl.split(tl.reshape(pair, (BLOCK, 2)))
    score = bits.to(tl.float32, bitcast=True)
    # Model length is guarded at 2**24 by AscendSFAIndexer. Every valid ID
    # is therefore exact in the FP32 sort key; invalid entries sort last.
    key = tl.where(idx.to(tl.float32) >= 0, idx.to(tl.float32), float("inf"))
    tl.store(id_keys + row * WORLD * K + rank * K + col, key, col < K)
    tl.store(scores + row * WORLD * K + rank * K + col, score, col < K)


@triton.jit
def _select_candidates(
    selected_keys,
    sorted_scores,
    output,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    WORLD: tl.constexpr = 1,
    RANK: tl.constexpr = -1,
    INTERLEAVE: tl.constexpr = 1,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    score = tl.load(sorted_scores + row * N + col, col < K, -float("inf"))
    idx = tl.load(selected_keys + row * K + col, col < K, 0)
    valid = tl.abs(score) < float("inf")
    # Avoid converting the invalid +inf sentinel to an integer.
    result = tl.where(valid, idx, -1).to(tl.int32)
    if RANK >= 0:
        # Global score/ID selection is already complete. Sort unique position
        # keys only to compact this owner's winners without reordering ties.
        block = result // INTERLEAVE
        cycle = block // WORLD
        owner = block - cycle * WORLD
        keep = (col < K) & valid & (result >= 0) & (owner == RANK)
        local = cycle * INTERLEAVE + (result - block * INTERLEAVE)
        order = _a5_sort(tl.where(keep, col.to(tl.float32), float("inf")), descending=False)
        lookup = tl.minimum(order, BLOCK - 1).to(tl.int32)
        selected = tl.gather(local.to(tl.float32, bitcast=True), lookup, 0).to(tl.int32, bitcast=True)
        result = tl.where(order < float("inf"), selected, -1)
    tl.store(output + row * K + col, result, col < K)


def select_local_candidates(indices, scores, world_size: int, rank: int, interleave: int, *, out=None):
    """Fuse final score masking, local remap and stable compaction on A5."""
    output = torch.empty_like(indices, dtype=torch.int32) if out is None else out
    if indices.numel() == 0:
        return output
    topk = indices.shape[-1]
    rows = indices.numel() // topk
    _select_candidates[(rows, 1)](
        indices,
        scores,
        output,
        scores.numel() // rows,
        topk,
        triton.next_power_of_2(topk),
        WORLD=world_size,
        RANK=rank,
        INTERLEAVE=interleave,
    )
    return output


def pack_local_candidates(indices, scores, world_size: int, rank: int, interleave: int):
    """Transport exact INT32 IDs and FP32 score bits in one collective."""
    rows, _, topk = indices.shape
    packed = torch.empty((rows, topk, 2), dtype=torch.int32, device=indices.device)
    if rows:
        _pack_candidates[(rows, triton.cdiv(topk, 256))](
            indices,
            scores,
            packed,
            indices.stride(0),
            scores.stride(0),
            indices.stride(-1),
            scores.stride(-1),
            topk,
            world_size,
            rank,
            interleave,
            256,
        )
    return packed


def merge_packed_candidates(
    packed, rows: int, world_size: int, topk: int, *, local_rank: int | None = None, interleave: int = 1, out=None
):
    """Keep the reference's stable score/ID comparator, fusing its gathers."""
    local_k = packed.shape[1]
    count = world_size * local_k
    keys = torch.empty((rows, count), dtype=torch.float32, device=packed.device)
    values = torch.empty_like(keys)
    block = min(2048, triton.next_power_of_2(local_k))
    _unpack_candidates[(rows, world_size * triton.cdiv(local_k, block))](
        packed, keys, values, rows, local_k, world_size, block
    )
    sorted_keys, id_order = torch.sort(keys, dim=-1, stable=True)
    sorted_scores = torch.gather(values, -1, id_order)
    ordered_scores, score_order = torch.sort(sorted_scores, dim=-1, descending=True, stable=True)
    # Reuse both sorts' values instead of following two dependent random GM
    # indices in Triton. The native gather vectorizes the remaining lookup;
    # the final kernel only reads contiguous IDs and comparator scores.
    selected_keys = torch.gather(sorted_keys, -1, score_order[:, :topk])
    output = torch.empty((rows, 1, topk), dtype=torch.int32, device=packed.device) if out is None else out
    if local_rank is not None:
        select_local_candidates(
            selected_keys, ordered_scores, world_size, local_rank, interleave, out=output.view(rows, topk)
        )
        return output
    _select_candidates[(rows, triton.cdiv(topk, 256))](
        selected_keys,
        ordered_scores,
        output,
        count,
        topk,
        256,
    )
    return output


@triton.jit
def _local_lengths_small_batch(
    seq_lens,
    query_ends,
    key_lengths,
    visible_lengths,
    REQUESTS: tl.constexpr,
    ROWS: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK_REQUESTS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    request = tl.arange(0, BLOCK_REQUESTS)
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    seq = tl.load(seq_lens + request, request < REQUESTS, 0).to(tl.int32)
    end = tl.load(query_ends + request, request < REQUESTS, 0).to(tl.int32)
    start = tl.load(query_ends + request - 1, (request > 0) & (request < REQUESTS), 0).to(tl.int32)
    if tl.program_id(0) == 0:
        local = seq // (WORLD * INTERLEAVE) * INTERLEAVE
        local += tl.minimum(tl.maximum(seq % (WORLD * INTERLEAVE) - RANK * INTERLEAVE, 0), INTERLEAVE)
        tl.store(key_lengths + request, local, request < REQUESTS)
    # All request bounds fit in UB for the common decode batches. A bounded
    # row/request mask avoids dependent random GM loads during binary search.
    own = (row[:, None] >= start[None, :]) & (row[:, None] < end[None, :])
    own &= request[None, :] < REQUESTS
    visible = tl.maximum(seq[None, :] - end[None, :] + row[:, None] + 1, 0)
    visible = tl.sum(tl.where(own, visible, 0), 1)
    local = visible // (WORLD * INTERLEAVE) * INTERLEAVE
    local += tl.minimum(tl.maximum(visible % (WORLD * INTERLEAVE) - RANK * INTERLEAVE, 0), INTERLEAVE)
    tl.store(visible_lengths + row, local, row < ROWS)


@triton.jit
def _local_lengths(
    seq_lens,
    query_ends,
    key_lengths,
    visible_lengths,
    REQUESTS: tl.constexpr,
    ROWS: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(seq_lens + offsets, offsets < REQUESTS, 0)
    local = length // (WORLD * INTERLEAVE) * INTERLEAVE
    local += tl.minimum(tl.maximum(length % (WORLD * INTERLEAVE) - RANK * INTERLEAVE, 0), INTERLEAVE)
    tl.store(key_lengths + offsets, local, offsets < REQUESTS)
    # Upper_bound on device query ends also handles empty requests. Values
    # change on graph replay without rebuilding or copying a CPU row map.
    lower = tl.full((BLOCK,), 0, tl.int32)
    upper = tl.full((BLOCK,), REQUESTS, tl.int32)
    for _ in range(STEPS):
        mid = (lower + upper) // 2
        end = tl.load(query_ends + mid, mid < REQUESTS, 2147483647)
        right = (mid < REQUESTS) & (end <= offsets)
        lower = tl.where(right, mid + 1, lower)
        upper = tl.where(right, upper, mid)
    request = tl.minimum(lower, REQUESTS - 1)
    seq = tl.load(seq_lens + request)
    end = tl.load(query_ends + request)
    total = tl.load(query_ends + REQUESTS - 1)
    visible = tl.maximum(seq - end + offsets + 1, 0)
    visible = tl.where(offsets < total, visible, 0)
    local = visible // (WORLD * INTERLEAVE) * INTERLEAVE
    local += tl.minimum(tl.maximum(visible % (WORLD * INTERLEAVE) - RANK * INTERLEAVE, 0), INTERLEAVE)
    tl.store(visible_lengths + offsets, local, offsets < ROWS)


@triton.jit
def _zero_visible_lengths(visible_lengths, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(visible_lengths + row, 0, row < ROWS)


def build_local_lengths(
    seq_lens, query_ends, key_lengths, visible_lengths, world_size: int, rank: int, interleave: int
):
    """Write stable metadata buffers used by native causal local QLI."""
    requests, rows = seq_lens.numel(), visible_lengths.numel()
    if requests == 0:
        if rows:
            _zero_visible_lengths[(triton.cdiv(rows, 256),)](visible_lengths, rows, 256)
        return
    if requests <= 32 and rows:
        _local_lengths_small_batch[(triton.cdiv(rows, 32),)](
            seq_lens,
            query_ends,
            key_lengths,
            visible_lengths,
            requests,
            rows,
            world_size,
            rank,
            interleave,
            triton.next_power_of_2(requests),
            32,
        )
        return
    _local_lengths[(triton.cdiv(max(requests, rows), 128),)](
        seq_lens,
        query_ends,
        key_lengths,
        visible_lengths,
        requests,
        rows,
        world_size,
        rank,
        interleave,
        128,
        requests.bit_length(),
    )
