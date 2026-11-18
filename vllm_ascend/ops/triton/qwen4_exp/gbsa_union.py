# SPDX-License-Identifier: Apache-2.0
"""Fused construction of Qwen4Exp Union-GBSA sparse inputs."""

import torch
from vllm.triton_utils import tl, triton


UNION_SIZE = 8
BLOCK_Y = 4
INVALID_KEY = 1 << 30


@triton.jit
def _pack_encode_qsa_union_keys_kernel(
    compact_indices,
    visible,
    group_row_map,
    encoded_keys,
    union_idx,
    exact_mask,
    compact_row_stride,
    compact_column_stride,
    visible_stride,
    map_group_stride,
    map_lane_stride,
    INPUT_GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    UNION: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    INVALID: tl.constexpr,
):
    """Pack one query lane and directly encode its Union sort keys."""
    group = tl.program_id(0)
    lane = tl.program_id(1)
    columns = tl.arange(0, GROUP_BLOCK)
    row = tl.load(group_row_map + group * map_group_stride + lane * map_lane_stride).to(tl.int32)
    row_valid = row >= 0
    safe_row = tl.maximum(row, 0)
    row_visible = tl.load(visible + safe_row * visible_stride, mask=row_valid, other=0).to(tl.int32)
    selected = tl.load(
        compact_indices + safe_row * compact_row_stride + columns * compact_column_stride,
        mask=row_valid & (columns < INPUT_GROUPS),
        other=-1,
    ).to(tl.int32)
    valid = row_valid & (columns < INPUT_GROUPS) & (selected >= 0) & (selected < row_visible // BLOCK)
    prefix = tl.cumsum(valid.to(tl.int32), axis=0)
    rank = prefix - 1
    count = tl.sum(valid.to(tl.int32), axis=0)
    has_tail = row_valid & (row_visible % BLOCK != 0)
    valid_capacity = TOPK - has_tail.to(tl.int32)
    kept = valid & (rank < valid_capacity)

    lane_base = group * WIDTH + lane * TOPK
    output_columns = tl.arange(0, TOPK)
    tl.store(encoded_keys + lane_base + output_columns, INVALID)
    tl.store(union_idx + lane_base + output_columns, -1)
    mask_columns = tl.arange(0, TOPK * BLOCK)
    tl.store(exact_mask + group * WIDTH * BLOCK + lane * TOPK * BLOCK + mask_columns, 0)

    tl.store(
        encoded_keys + lane_base + rank,
        (selected * UNION + lane).to(tl.float32),
        mask=kept,
    )
    tail_slot = tl.minimum(count, TOPK - 1)
    tl.store(
        encoded_keys + lane_base + tail_slot,
        ((row_visible // BLOCK) * UNION + lane).to(tl.float32),
        mask=has_tail,
    )


@triton.jit
def _encode_qsa_union_keys_kernel(
    sparse_idx,
    sparse_count,
    sorted_keys,
    union_idx,
    exact_mask,
    sparse_idx_row_stride,
    sparse_idx_column_stride,
    sparse_count_stride,
    rows,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    EXACT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    UNION: tl.constexpr,
    BLOCK: tl.constexpr,
    INVALID: tl.constexpr,
):
    """Encode ``(block, query lane)`` keys for the native sort."""
    group = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_valid = offsets < WIDTH
    lane = offsets // TOPK
    column = offsets % TOPK
    row = group * UNION + lane
    row_valid = row < rows
    count = tl.load(
        sparse_count + row * sparse_count_stride,
        mask=row_valid,
        other=0,
    ).to(tl.int32)
    block = tl.load(
        sparse_idx + row * sparse_idx_row_stride + column * sparse_idx_column_stride,
        mask=offset_valid & row_valid & (column < count),
        other=-1,
    ).to(tl.int32)
    valid = offset_valid & row_valid & (column < count) & (block >= 0)
    key = tl.where(valid, block * UNION + lane, INVALID)
    # The Union package requires exact FP32 sort keys. The configured maximum
    # logical block keeps every encoded key below 2**24.
    tl.store(sorted_keys + group * WIDTH + offsets, key.to(tl.float32), mask=offset_valid)

    # These outputs are caller-visible, so make the unused capacity
    # deterministic before the compact/mask kernel writes the valid prefix.
    tl.store(union_idx + group * WIDTH + offsets, -1, mask=offset_valid)
    mask_offsets = chunk * BLOCK_SIZE * BLOCK + tl.arange(0, BLOCK_SIZE * BLOCK)
    tl.store(
        exact_mask + group * EXACT_WIDTH + mask_offsets,
        0,
        mask=mask_offsets < EXACT_WIDTH,
    )


@triton.jit
def _compact_qsa_union_keys_kernel(
    sorted_keys,
    visible,
    union_idx,
    union_count,
    exact_mask,
    rows,
    visible_stride,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNKS: tl.constexpr,
    UNION: tl.constexpr,
    BLOCK: tl.constexpr,
    INVALID: tl.constexpr,
):
    """Compact unique blocks and build the exact 8-lane causal mask."""
    group = tl.program_id(0)
    prefix = 0
    for chunk in tl.static_range(CHUNKS):
        offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offset_valid = offsets < WIDTH
        keys = tl.load(
            sorted_keys + group * WIDTH + offsets,
            mask=offset_valid,
            other=INVALID,
        ).to(tl.int32)
        valid = keys < INVALID
        blocks = keys // UNION
        previous_key = tl.load(
            sorted_keys + group * WIDTH + offsets - 1,
            mask=offset_valid & (offsets > 0),
            other=-1,
        ).to(tl.int32)
        previous_block = tl.where(offsets > 0, previous_key // UNION, -1)
        first = valid & (blocks != previous_block)
        local_rank = tl.cumsum(first.to(tl.int32), axis=0) - 1
        rank = prefix + local_rank
        chunk_count = tl.sum(first.to(tl.int32), axis=0)
        tl.store(union_idx + group * WIDTH + rank, blocks, mask=first)

        lane = keys % UNION
        row = group * UNION + lane
        row_valid = valid & (row < rows)
        row_visible = tl.load(visible + row * visible_stride, mask=row_valid, other=0).to(tl.int32)
        token_count = tl.minimum(tl.maximum(row_visible - blocks * BLOCK, 0), BLOCK)
        bit = tl.full((BLOCK_SIZE,), 1, tl.int32) << lane

        # Entries sharing a block have the same rank. Atomic OR combines their
        # query-lane bits; token_count also handles the partial tail block.
        for token in tl.static_range(BLOCK):
            destination = group * WIDTH * BLOCK + rank * BLOCK + token
            tl.atomic_or(
                exact_mask + destination,
                bit,
                mask=row_valid & (token < token_count),
            )
        prefix += chunk_count
    tl.store(union_count + group, prefix)


@triton.jit
def _compact_qsa_union_keys_mapped_kernel(
    sorted_keys,
    visible,
    group_row_map,
    union_idx,
    union_count,
    exact_mask,
    visible_stride,
    map_group_stride,
    map_lane_stride,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNKS: tl.constexpr,
    UNION: tl.constexpr,
    BLOCK: tl.constexpr,
    INVALID: tl.constexpr,
):
    """Compact encoded keys using request-safe padded-lane row mappings."""
    group = tl.program_id(0)
    prefix = 0
    for chunk in tl.static_range(CHUNKS):
        offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offset_valid = offsets < WIDTH
        keys = tl.load(sorted_keys + group * WIDTH + offsets, mask=offset_valid, other=INVALID).to(tl.int32)
        valid = keys < INVALID
        blocks = keys // UNION
        previous_key = tl.load(
            sorted_keys + group * WIDTH + offsets - 1,
            mask=offset_valid & (offsets > 0),
            other=-1,
        ).to(tl.int32)
        previous_block = tl.where(offsets > 0, previous_key // UNION, -1)
        first = valid & (blocks != previous_block)
        local_rank = tl.cumsum(first.to(tl.int32), axis=0) - 1
        rank = prefix + local_rank
        chunk_count = tl.sum(first.to(tl.int32), axis=0)
        tl.store(union_idx + group * WIDTH + rank, blocks, mask=first)

        lane = keys % UNION
        source_row = tl.load(
            group_row_map + group * map_group_stride + lane * map_lane_stride,
            mask=valid,
            other=-1,
        ).to(tl.int32)
        row_valid = valid & (source_row >= 0)
        row_visible = tl.load(
            visible + tl.maximum(source_row, 0) * visible_stride,
            mask=row_valid,
            other=0,
        ).to(tl.int32)
        token_count = tl.minimum(tl.maximum(row_visible - blocks * BLOCK, 0), BLOCK)
        bit = tl.full((BLOCK_SIZE,), 1, tl.int32) << lane
        for token in tl.static_range(BLOCK):
            destination = group * WIDTH * BLOCK + rank * BLOCK + token
            tl.atomic_or(exact_mask + destination, bit, mask=row_valid & (token < token_count))
        prefix += chunk_count
    tl.store(union_count + group, prefix)


def build_qsa_union_inputs_triton(
    sparse_idx: torch.Tensor,
    sparse_count: torch.Tensor,
    visible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Union-GBSA indices, counts and exact masks in two launches.

    The input is the per-query compact sparse selection produced by the QSA
    pack kernel. Every consecutive eight query rows form one Union-GBSA row.
    Callers handling packed requests must pad each request independently so a
    union never crosses a request boundary.
    """
    if sparse_idx.ndim != 3 or sparse_idx.shape[0] != 1:
        raise ValueError("Union preparation requires one KV head")
    if sparse_count.shape != sparse_idx.shape[:2]:
        raise ValueError("Union sparse counts must match sparse rows")
    rows, topk = sparse_idx.shape[1:]
    if visible.shape != (rows,):
        raise ValueError("Union visible lengths must match sparse rows")
    if topk <= 0 or topk & (topk - 1):
        raise ValueError("Union sparse width must be a positive power of two")

    groups = (rows + UNION_SIZE - 1) // UNION_SIZE
    width = UNION_SIZE * topk
    exact_width = width * BLOCK_Y
    block_size = min(width, 512)
    chunks = triton.cdiv(width, block_size)
    encoded_keys = torch.empty((groups, width), dtype=torch.float32, device=sparse_idx.device)
    union_idx = torch.empty((1, groups, width), dtype=torch.int32, device=sparse_idx.device)
    union_count = torch.empty((1, groups), dtype=torch.int32, device=sparse_count.device)
    exact = torch.empty((1, groups, exact_width), dtype=torch.int32, device=sparse_idx.device)
    if groups == 0:
        return union_idx, union_count, exact

    _encode_qsa_union_keys_kernel[(groups, chunks)](
        sparse_idx,
        sparse_count,
        encoded_keys,
        union_idx,
        exact,
        sparse_idx.stride(1),
        sparse_idx.stride(2),
        sparse_count.stride(1),
        rows,
        TOPK=topk,
        WIDTH=width,
        EXACT_WIDTH=exact_width,
        BLOCK_SIZE=block_size,
        UNION=UNION_SIZE,
        BLOCK=BLOCK_Y,
        INVALID=INVALID_KEY,
        num_warps=8,
    )
    sorted_keys = encoded_keys.sort(dim=-1).values
    _compact_qsa_union_keys_kernel[(groups,)](
        sorted_keys,
        visible,
        union_idx,
        union_count,
        exact,
        rows,
        visible.stride(0),
        WIDTH=width,
        BLOCK_SIZE=block_size,
        CHUNKS=chunks,
        UNION=UNION_SIZE,
        BLOCK=BLOCK_Y,
        INVALID=INVALID_KEY,
        num_warps=8,
    )
    return union_idx, union_count, exact


def build_qsa_union_inputs_from_compact_triton(
    compact_indices: torch.Tensor,
    visible: torch.Tensor,
    group_row_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Union inputs directly from LI-v2 output without sparse staging."""
    if compact_indices.ndim != 2:
        raise ValueError("Union compact indices must be two-dimensional")
    if visible.shape != (compact_indices.shape[0],):
        raise ValueError("Union visible lengths must match compact rows")
    if group_row_map.ndim != 2 or group_row_map.shape[1] != UNION_SIZE:
        raise ValueError("Union row map must have eight lanes per group")
    input_groups = compact_indices.shape[1] - 3
    topk = input_groups + int(input_groups < 512)
    if input_groups <= 0 or topk & (topk - 1):
        raise ValueError("Union packed width must be a positive power of two")

    groups = group_row_map.shape[0]
    width = UNION_SIZE * topk
    exact_width = width * BLOCK_Y
    custom_namespace = getattr(torch.ops, "_C_ascend", None)
    custom_op = getattr(custom_namespace, "qsa_union_inputs", None)
    if custom_op is not None:
        return custom_op(
            compact_indices.contiguous(),
            visible.contiguous(),
            group_row_map.contiguous(),
        )
    block_size = min(width, 512)
    chunks = triton.cdiv(width, block_size)
    encoded_keys = torch.empty((groups, width), dtype=torch.float32, device=compact_indices.device)
    union_idx = torch.empty((1, groups, width), dtype=torch.int32, device=compact_indices.device)
    union_count = torch.empty((1, groups), dtype=torch.int32, device=compact_indices.device)
    exact = torch.empty((1, groups, exact_width), dtype=torch.int32, device=compact_indices.device)
    if groups == 0:
        return union_idx, union_count, exact

    _pack_encode_qsa_union_keys_kernel[(groups, UNION_SIZE)](
        compact_indices,
        visible,
        group_row_map,
        encoded_keys,
        union_idx,
        exact,
        compact_indices.stride(0),
        compact_indices.stride(1),
        visible.stride(0),
        group_row_map.stride(0),
        group_row_map.stride(1),
        INPUT_GROUPS=input_groups,
        TOPK=topk,
        WIDTH=width,
        UNION=UNION_SIZE,
        BLOCK=BLOCK_Y,
        GROUP_BLOCK=triton.next_power_of_2(input_groups),
        INVALID=INVALID_KEY,
        num_warps=8,
    )
    sorted_keys = encoded_keys.sort(dim=-1).values
    _compact_qsa_union_keys_mapped_kernel[(groups,)](
        sorted_keys,
        visible,
        group_row_map,
        union_idx,
        union_count,
        exact,
        visible.stride(0),
        group_row_map.stride(0),
        group_row_map.stride(1),
        WIDTH=width,
        BLOCK_SIZE=block_size,
        CHUNKS=chunks,
        UNION=UNION_SIZE,
        BLOCK=BLOCK_Y,
        INVALID=INVALID_KEY,
        num_warps=8,
    )
    return union_idx, union_count, exact


__all__ = ["build_qsa_union_inputs_from_compact_triton", "build_qsa_union_inputs_triton"]
