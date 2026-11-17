# SPDX-License-Identifier: Apache-2.0
"""Fused integer preparation for Qwen4Exp Union GBSA."""

import torch
from vllm.triton_utils import tl, triton

UNION_SIZE = 8
BLOCK_Y = 4
INVALID_KEY = 1 << 30


@triton.jit
def _qsa_token_mask_bits(block, seen, lane, BLOCK_Y: tl.constexpr = BLOCK_Y):
    token_count = seen - block * BLOCK_Y
    factor = tl.where(
        token_count >= 4,
        0x01010101,
        tl.where(
            token_count == 3, 0x00010101, tl.where(token_count == 2, 0x00000101, tl.where(token_count == 1, 1, 0))
        ),
    )
    # Lane is uniform per encoder program, so this is one scalar shift.
    return factor.to(tl.uint32) * (1 << lane).to(tl.uint32)


@triton.jit
def _prepare_qsa_union_sort_keys_kernel(
    indices,
    counts,
    visible,
    keys,
    encoded_masks,
    rows,
    index_row_stride,
    index_col_stride,
    count_stride,
    visible_stride,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    UNION_SIZE: tl.constexpr = UNION_SIZE,
    INVALID_KEY: tl.constexpr = INVALID_KEY,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    count = tl.load(counts + row * count_stride, row < rows, other=0)
    value = tl.load(
        indices + row * index_row_stride + columns * index_col_stride, (row < rows) & (columns < TOPK), other=-1
    ).to(tl.int32)
    encoded = tl.where((columns < count) & (value >= 0), value * UNION_SIZE + row % UNION_SIZE, INVALID_KEY)
    offset = (row // UNION_SIZE) * (UNION_SIZE * TOPK) + (row % UNION_SIZE) * TOPK + columns
    tl.store(keys + offset, encoded.to(tl.float32), columns < TOPK)
    seen = tl.load(visible + row * visible_stride, row < rows, other=0).to(tl.int32)
    bits = _qsa_token_mask_bits(value, seen, row % UNION_SIZE)
    tl.store(
        encoded_masks + offset,
        tl.where((columns < count) & (value >= 0) & (row < rows), bits, 0).to(tl.int32),
        columns < TOPK,
    )


@triton.jit
def _prepare_qsa_union_compact_keys_kernel(
    indices,
    positions,
    lengths,
    requests,
    keys,
    encoded_masks,
    rows,
    index_row_stride,
    index_col_stride,
    position_stride,
    length_stride,
    request_stride,
    requests_count,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    UNION_SIZE: tl.constexpr = UNION_SIZE,
    BLOCK_Y: tl.constexpr = BLOCK_Y,
    INVALID_KEY: tl.constexpr = INVALID_KEY,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    request = tl.load(requests + row * request_stride, row < rows, other=0).to(tl.int64)
    request = tl.where(request < 0, request + requests_count, request)
    position = tl.load(positions + row * position_stride, row < rows, other=-1).to(tl.int32)
    length = tl.load(lengths + request * length_stride, row < rows, other=0).to(tl.int32)
    seen = tl.minimum(tl.maximum(position + 1, 0), length)
    value = tl.load(
        indices + row * index_row_stride + columns * index_col_stride, (row < rows) & (columns < GROUPS), other=-1
    ).to(tl.int32)
    valid = (columns < GROUPS) & (value >= 0) & (value < seen // BLOCK_Y)
    has_tail = seen % BLOCK_Y != 0
    # The following sort discards input order. Keep valid candidates in place
    # instead of compacting holes; reserve one invalid slot for the causal tail.
    # If all 512 slots are valid, replace the last one, matching the reference.
    first_invalid = tl.min(tl.where((columns < GROUPS) & (~valid), columns, GROUPS), 0)
    tail_slot = tl.minimum(first_invalid, TOPK - 1)
    selected_tail = has_tail & (columns == tail_slot)
    packed = tl.where(selected_tail, seen // BLOCK_Y, value)
    selected = (valid | selected_tail) & (row < rows)
    encoded = tl.where(selected, packed * UNION_SIZE + row % UNION_SIZE, INVALID_KEY)
    offset = (row // UNION_SIZE) * (UNION_SIZE * TOPK) + (row % UNION_SIZE) * TOPK
    tl.store(keys + offset + columns, encoded.to(tl.float32), columns < GROUPS)
    bits = _qsa_token_mask_bits(packed, seen, row % UNION_SIZE)
    tl.store(encoded_masks + offset + columns, tl.where(selected, bits, 0).to(tl.int32), columns < GROUPS)
    if TOPK > GROUPS:
        extra_valid = has_tail & (tail_slot == GROUPS) & (row < rows)
        extra = tl.where(extra_valid, (seen // BLOCK_Y) * UNION_SIZE + row % UNION_SIZE, INVALID_KEY)
        tl.store(keys + offset + GROUPS, extra.to(tl.float32))
        bits = _qsa_token_mask_bits(seen // BLOCK_Y, seen, row % UNION_SIZE)
        tl.store(encoded_masks + offset + GROUPS, tl.where(extra_valid, bits, 0).to(tl.int32))


@triton.jit
def _prepare_qsa_union_heads_kernel(
    keys,
    heads,
    mask_bits,
    sorted_masks,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    UNION_SIZE: tl.constexpr = UNION_SIZE,
    BLOCK_Y: tl.constexpr = BLOCK_Y,
    INVALID_KEY: tl.constexpr = INVALID_KEY,
):
    group = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offset = group * WIDTH + columns
    key = tl.load(keys + offset, columns < WIDTH, other=INVALID_KEY).to(tl.int32)
    previous = tl.load(keys + offset - 1, (columns > 0) & (columns < WIDTH), other=-8).to(tl.int32)
    valid = (columns < WIDTH) & (key < INVALID_KEY)
    block = key // UNION_SIZE
    first = valid & ((columns == 0) | (block != previous // UNION_SIZE))
    bits = tl.load(sorted_masks + offset, columns < WIDTH, other=0)
    unique = valid & ((columns == 0) | (key != previous))
    tl.store(heads + offset, first.to(tl.int32), columns < WIDTH)
    tl.store(mask_bits + offset, tl.where(unique, bits, 0).to(tl.int32), columns < WIDTH)


@triton.jit
def _build_qsa_union_scatter_payload_kernel(
    keys,
    ranks,
    mask_prefix,
    destinations,
    values,
    union_counts,
    GROUPS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    UNION_SIZE: tl.constexpr = UNION_SIZE,
    BLOCK_Y: tl.constexpr = BLOCK_Y,
    INVALID_KEY: tl.constexpr = INVALID_KEY,
):
    group = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offset = group * WIDTH
    key = tl.load(keys + offset + columns, columns < WIDTH, other=INVALID_KEY).to(tl.int32)
    previous = tl.load(keys + offset + columns - 1, (columns > 0) & (columns < WIDTH), other=-8).to(tl.int32)
    rank = tl.load(ranks + offset + columns, columns < WIDTH, other=0) - 1
    block = key // UNION_SIZE
    valid = (columns < WIDTH) & (key < INVALID_KEY)
    first = valid & ((columns == 0) | (block != previous // UNION_SIZE))
    following = tl.load(keys + offset + columns + 1, columns + 1 < WIDTH, other=INVALID_KEY).to(tl.int32)
    last = valid & ((columns + 1 >= WIDTH) | (block != following // UNION_SIZE))
    before = tl.load(mask_prefix + offset + columns - 1, (columns > 0) & (columns < WIDTH), other=0)
    after = tl.load(mask_prefix + offset + columns, columns < WIDTH, other=0)
    first_destination = tl.where(first, rank, WIDTH)
    last_destination = tl.where(last, rank, WIDTH)
    plane = GROUPS * WIDTH
    # Three int32 payload planes use the efficient native ScatterElementsV2
    # path. Invalid lanes write identical values only to discarded sentinels.
    tl.store(destinations + offset + columns, first_destination, columns < WIDTH)
    tl.store(destinations + plane + offset + columns, first_destination, columns < WIDTH)
    tl.store(destinations + 2 * plane + offset + columns, last_destination, columns < WIDTH)
    tl.store(values + offset + columns, tl.where(first, block, 0), columns < WIDTH)
    tl.store(values + plane + offset + columns, tl.where(first, before, 0), columns < WIDTH)
    tl.store(values + 2 * plane + offset + columns, tl.where(last, after, 0), columns < WIDTH)
    if tl.program_id(1) == 0:
        tl.store(union_counts + group, tl.load(ranks + offset + WIDTH - 1))


def _finish_union(keys: torch.Tensor, encoded_masks: torch.Tensor):
    groups, width = keys.shape
    ordered = keys.sort(dim=-1)
    sorted_keys = ordered.values
    sorted_masks = encoded_masks.gather(1, ordered.indices)
    heads = torch.empty_like(keys, dtype=torch.int32)
    mask_bits = torch.empty_like(heads)
    counts = torch.empty((1, groups), dtype=torch.int32, device=keys.device)
    grid = (groups, triton.cdiv(width, 1024))
    _prepare_qsa_union_heads_kernel[grid](
        sorted_keys,
        heads,
        mask_bits,
        sorted_masks,
        WIDTH=width,
        BLOCK=1024,
        num_warps=1,
    )
    ranks = heads.cumsum(-1, dtype=torch.int32)
    mask_prefix = mask_bits.cumsum(-1, dtype=torch.int32)
    destinations = torch.empty((3 * groups, width), dtype=torch.int32, device=keys.device)
    values = torch.empty_like(destinations)
    _build_qsa_union_scatter_payload_kernel[grid](
        sorted_keys,
        ranks,
        mask_prefix,
        destinations,
        values,
        counts,
        GROUPS=groups,
        WIDTH=width,
        BLOCK=1024,
        num_warps=1,
    )
    table = torch.empty((3, groups, width + 1), dtype=torch.int32, device=keys.device)
    table[0].fill_(-1)
    table[1:].zero_()
    table.view(3 * groups, width + 1).scatter_(1, destinations.long(), values)
    # Ascend uses little-endian base-format storage. Reinterpretation avoids
    # four separate shifts/gathers; these views do not launch device kernels.
    indices = table[0, :, :width].contiguous().unsqueeze(0)
    packed = table[2] - table[1]
    mask = packed.view(torch.uint8)[:, : width * BLOCK_Y].to(torch.int32).contiguous().unsqueeze(0)
    return indices, counts, mask


def build_qsa_union_inputs_fused(indices: torch.Tensor, counts: torch.Tensor, visible: torch.Tensor):
    rows, topk = indices.shape[1:]
    groups = triton.cdiv(rows, UNION_SIZE)
    keys = torch.empty((groups, UNION_SIZE * topk), dtype=torch.float32, device=indices.device)
    encoded_masks = torch.empty_like(keys, dtype=torch.int32)
    if rows == 0:
        return (
            torch.empty((1, 0, UNION_SIZE * topk), dtype=torch.int32, device=indices.device),
            torch.empty((1, 0), dtype=torch.int32, device=indices.device),
            torch.empty((1, 0, UNION_SIZE * topk * BLOCK_Y), dtype=torch.int32, device=indices.device),
        )
    _prepare_qsa_union_sort_keys_kernel[(groups * UNION_SIZE,)](
        indices,
        counts,
        visible,
        keys,
        encoded_masks,
        rows,
        indices.stride(1),
        indices.stride(2),
        counts.stride(1),
        visible.stride(0),
        TOPK=topk,
        BLOCK=triton.next_power_of_2(topk),
        num_warps=1,
    )
    return _finish_union(keys, encoded_masks)


def prepare_and_build_qsa_union_inputs(
    compact_indices: torch.Tensor,
    positions: torch.Tensor,
    lengths: torch.Tensor,
    requests: torch.Tensor,
):
    rows = compact_indices.shape[0]
    groups = compact_indices.shape[1] - 3
    topk = groups + (groups < 512)
    union_groups = triton.cdiv(rows, UNION_SIZE)
    keys = torch.empty((union_groups, UNION_SIZE * topk), dtype=torch.float32, device=compact_indices.device)
    encoded_masks = torch.empty_like(keys, dtype=torch.int32)
    if rows == 0:
        raise ValueError("Union attention requires nonempty queries")
    _prepare_qsa_union_compact_keys_kernel[(union_groups * UNION_SIZE,)](
        compact_indices,
        positions,
        lengths,
        requests,
        keys,
        encoded_masks,
        rows,
        compact_indices.stride(0),
        compact_indices.stride(1),
        positions.stride(0),
        lengths.stride(0),
        requests.stride(0),
        lengths.shape[0],
        GROUPS=groups,
        TOPK=topk,
        BLOCK=triton.next_power_of_2(groups),
        num_warps=1,
    )
    return _finish_union(keys, encoded_masks)
