# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused QSA support kernels used around the CANN LI-v2 and GBSA path."""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton


@triton.jit
def _pack_qsa_group_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    token_to_req_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    rows,
    num_requests,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.arange(0, OUTPUT_WIDTH)
    query_position = tl.load(query_positions_ptr + row)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    complete_groups = tl.minimum(
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
        BLOCK_TOPK,
    )
    complete_groups = tl.maximum(complete_groups, 0).to(tl.int32)
    is_group = columns < BLOCK_TOPK
    safe_group_column = tl.minimum(columns, BLOCK_TOPK - 1)
    group = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_group_column * stride_blocks_column,
        mask=is_group & (columns < complete_groups),
        other=-1,
    )
    tail_offset = columns - BLOCK_TOPK
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = tl.minimum((query_position + 1) - tail_start, COMPRESS_RATIO - 1)
    # Keep wide scalar position arithmetic, but use bounded int32 vectors.
    valid_tail_count = tl.maximum(tl.minimum(tail_count, sequence_length - tail_start), 0).to(tl.int32)
    tail = tail_start.to(tl.int32) + tail_offset
    valid_tail = (~is_group) & (tail_offset >= 0) & (tail_offset < valid_tail_count)
    packed = tl.where(is_group, group, tl.where(valid_tail, tail, -1))
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        packed,
        mask=(row < rows) & (columns < OUTPUT_WIDTH),
    )


@triton.jit
def _pack_qsa_group_indices_and_store_kv_kernel(
    block_indices_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    token_to_req_ptr,
    output_ptr,
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slots_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    stride_key_row,
    stride_key_dim,
    stride_value_row,
    stride_value_dim,
    stride_key_cache_block,
    stride_key_cache_token,
    stride_key_cache_dim,
    stride_value_cache_block,
    stride_value_cache_token,
    stride_value_cache_dim,
    rows,
    num_requests,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    KV_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    query_position = tl.load(query_positions_ptr + row)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    complete_groups = tl.minimum(
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
        BLOCK_TOPK,
    )
    complete_groups = tl.maximum(complete_groups, 0).to(tl.int32)
    is_group = columns < BLOCK_TOPK
    safe_group_column = tl.minimum(columns, BLOCK_TOPK - 1)
    group = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_group_column * stride_blocks_column,
        mask=(columns < OUTPUT_WIDTH) & is_group & (columns < complete_groups),
        other=-1,
    )
    tail_offset = columns - BLOCK_TOPK
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = tl.minimum((query_position + 1) - tail_start, COMPRESS_RATIO - 1)
    valid_tail_count = tl.maximum(tl.minimum(tail_count, sequence_length - tail_start), 0).to(tl.int32)
    tail = tail_start.to(tl.int32) + tail_offset
    valid_tail = (~is_group) & (tail_offset >= 0) & (tail_offset < valid_tail_count)
    packed = tl.where(is_group, group, tl.where(valid_tail, tail, -1))
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        packed,
        mask=columns < OUTPUT_WIDTH,
    )

    slot = tl.load(slots_ptr + row)
    cache_capacity = num_blocks * PAGE_SIZE
    valid_slot = (slot >= 0) & (slot < cache_capacity)
    safe_slot = tl.minimum(tl.maximum(slot, 0), cache_capacity - 1).to(tl.int64)
    block = safe_slot // PAGE_SIZE
    token = safe_slot % PAGE_SIZE
    key_values = tl.load(
        key_ptr + row * stride_key_row + columns * stride_key_dim,
        mask=columns < KV_WIDTH,
        other=0,
    )
    value_values = tl.load(
        value_ptr + row * stride_value_row + columns * stride_value_dim,
        mask=columns < KV_WIDTH,
        other=0,
    )
    tl.store(
        key_cache_ptr
        + block * stride_key_cache_block
        + token * stride_key_cache_token
        + columns * stride_key_cache_dim,
        key_values,
        mask=valid_slot & (columns < KV_WIDTH),
    )
    tl.store(
        value_cache_ptr
        + block * stride_value_cache_block
        + token * stride_value_cache_token
        + columns * stride_value_cache_dim,
        value_values,
        mask=valid_slot & (columns < KV_WIDTH),
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    cache_capacity = num_blocks * PAGE_SIZE
    valid = (row < num_rows) & (slot >= 0) & (slot < cache_capacity)
    # Triton-Ascend can validate addresses before applying the store mask.
    # Keep every lane in bounds even when graph padding supplies an invalid
    # positive slot. Cast before multiplying by a potentially large stride so
    # the physical cache offset cannot overflow int32 either.
    safe_slot = tl.minimum(tl.maximum(slot, 0), cache_capacity - 1).to(tl.int64)
    block = safe_slot // PAGE_SIZE
    token = safe_slot % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr + block * stride_cache_block + token * stride_cache_token + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,
    raw_positions_ptr,
    compressor_state_cache_ptr,
    rope_cache_ptr,
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(query_start_loc_ptr + safe_request, mask=valid_request, other=0)
    query_row_end = tl.load(query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0)
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row & ~use_raw & valid_compressor_state_block & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr + raw_first_row * stride_raw_positions_row + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row & ~first_from_raw & valid_compressor_state_block & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    position_values = tl.where(valid_row, position_values, 0)
    tl.store(
        first_positions_ptr + row * stride_positions_row + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )


def pack_qsa_group_indices_npu(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
    *,
    key: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    key_cache: torch.Tensor | None = None,
    value_cache: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack LI-v2 groups and the open-group tail into one fixed buffer."""
    if block_indices.device.type != "npu" or not HAS_TRITON:
        raise RuntimeError("QSA NPU compact packing requires Triton")
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = token_topk // compress_ratio
    output_width = block_topk + compress_ratio - 1
    if block_indices.shape != (query_positions.numel(), block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if token_to_req.shape != query_positions.shape:
        raise ValueError("QSA request mapping must match query positions")
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.shape != (block_indices.shape[0], output_width):
        raise ValueError("QSA compact selection output has an invalid shape")
    if not block_indices.shape[0]:
        return out
    fused_kv = all(t is not None for t in (key, value, key_cache, value_cache, slot_mapping))
    if fused_kv:
        assert key is not None and value is not None
        assert key_cache is not None and value_cache is not None and slot_mapping is not None
        if key.ndim == 3:
            if key.shape[1] != 1 or value.shape[1] != 1:
                raise ValueError("QSA fused K/V rows require one local KV head")
            key = key[:, 0]
            value = value[:, 0]
        if key.ndim != 2 or value.shape != key.shape:
            raise ValueError("QSA fused K/V rows must have matching two-dimensional shapes")
        if key_cache.ndim != 4 or value_cache.shape != key_cache.shape or key_cache.shape[2] != 1:
            raise ValueError("QSA fused K/V caches must have matching singleton-head layouts")
        if key.shape[1] != key_cache.shape[3] or key.dtype != key_cache.dtype or value.dtype != value_cache.dtype:
            raise ValueError("QSA fused K/V rows do not match their cache layout")
        slots = slot_mapping.reshape(-1)
        if min(key.shape[0], value.shape[0], slots.shape[0]) < block_indices.shape[0]:
            raise ValueError("QSA fused K/V inputs do not cover all selected rows")
        block_size = triton.next_power_of_2(max(output_width, key.shape[1]))
        _pack_qsa_group_indices_and_store_kv_kernel[(block_indices.shape[0],)](
            block_indices,
            query_positions,
            sequence_lengths,
            token_to_req,
            out,
            key,
            value,
            key_cache,
            value_cache,
            slots,
            block_indices.stride(0),
            block_indices.stride(1),
            out.stride(0),
            out.stride(1),
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(3),
            block_indices.shape[0],
            sequence_lengths.shape[0],
            key_cache.shape[0],
            PAGE_SIZE=key_cache.shape[1],
            BLOCK_TOPK=block_topk,
            COMPRESS_RATIO=compress_ratio,
            OUTPUT_WIDTH=output_width,
            KV_WIDTH=key.shape[1],
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        return out
    _pack_qsa_group_indices_kernel[(block_indices.shape[0],)](
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        block_indices.shape[0],
        sequence_lengths.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        OUTPUT_WIDTH=output_width,
        num_warps=1,
    )
    return out


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a strided QSA cache."""
    if cache.device.type != "npu" or not HAS_TRITON:
        raise RuntimeError("QSA NPU cache stores require Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.ndim != 2 or rows.shape[1] != cache.shape[3]:
        raise ValueError("QSA cache rows have an incompatible width")

    slot_mapping = slot_mapping.reshape(-1)
    num_updates = min(slot_mapping.shape[0], rows.shape[0])
    slot_mapping = slot_mapping[:num_updates]
    rows = rows[:num_updates]
    if not num_updates:
        return
    _store_qsa_rows_kernel[(num_updates,)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        num_updates,
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


@triton.jit
def _store_qsa_key_position_rows_kernel(
    key_cache_ptr,
    position_cache_ptr,
    slots_ptr,
    key_rows_ptr,
    position_rows_ptr,
    stride_key_cache_block,
    stride_key_cache_token,
    stride_key_cache_dim,
    stride_position_cache_block,
    stride_position_cache_token,
    stride_position_cache_dim,
    stride_key_rows_row,
    stride_key_rows_dim,
    stride_position_rows_row,
    stride_position_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    KEY_WIDTH: tl.constexpr,
    POSITION_WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    cache_capacity = num_blocks * PAGE_SIZE
    valid = (row < num_rows) & (slot >= 0) & (slot < cache_capacity)
    safe_slot = tl.minimum(tl.maximum(slot, 0), cache_capacity - 1).to(tl.int64)
    block = safe_slot // PAGE_SIZE
    token = safe_slot % PAGE_SIZE
    key_values = tl.load(
        key_rows_ptr + row * stride_key_rows_row + dims * stride_key_rows_dim,
        mask=valid & (dims < KEY_WIDTH),
        other=0,
    )
    position_values = tl.load(
        position_rows_ptr + row * stride_position_rows_row + dims * stride_position_rows_dim,
        mask=valid & (dims < POSITION_WIDTH),
        other=0,
    )
    tl.store(
        key_cache_ptr + block * stride_key_cache_block + token * stride_key_cache_token + dims * stride_key_cache_dim,
        key_values,
        mask=valid & (dims < KEY_WIDTH),
    )
    tl.store(
        position_cache_ptr
        + block * stride_position_cache_block
        + token * stride_position_cache_token
        + dims * stride_position_cache_dim,
        position_values,
        mask=valid & (dims < POSITION_WIDTH),
    )


def qsa_store_key_position_rows(
    key_cache: torch.Tensor,
    position_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_rows: torch.Tensor,
    position_rows: torch.Tensor,
) -> None:
    """Store raw QSA keys and their MRoPE positions in one launch."""
    if key_cache.device.type != "npu" or not HAS_TRITON:
        raise RuntimeError("QSA NPU cache stores require Triton")
    if key_cache.ndim != 4 or key_cache.shape[2] != 1:
        raise ValueError("QSA key cache must be [pages, page_size, 1, width]")
    if position_cache.ndim != 4 or position_cache.shape[:3] != key_cache.shape[:3]:
        raise ValueError("QSA position cache must match the key cache layout")
    if position_cache.shape[3] != 3:
        raise ValueError("QSA position cache width must be three")
    if not all(key_cache.shape) or not all(position_cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if key_rows.ndim == 3:
        if key_rows.shape[1] != 1:
            raise ValueError("QSA key rows must have one head")
        key_rows = key_rows[:, 0]
    if position_rows.ndim == 3:
        if position_rows.shape[1] != 1:
            raise ValueError("QSA position rows must have one head")
        position_rows = position_rows[:, 0]
    if key_rows.ndim != 2 or key_rows.shape[1] != key_cache.shape[3]:
        raise ValueError("QSA key rows have an incompatible width")
    if position_rows.ndim != 2 or position_rows.shape[1] != position_cache.shape[3]:
        raise ValueError("QSA position rows have an incompatible width")
    if key_rows.dtype != key_cache.dtype or position_rows.dtype != position_cache.dtype:
        raise ValueError("QSA cache rows must match their cache dtypes")

    slot_mapping = slot_mapping.reshape(-1)
    num_updates = min(slot_mapping.shape[0], key_rows.shape[0], position_rows.shape[0])
    if not num_updates:
        return
    _store_qsa_key_position_rows_kernel[(num_updates,)](
        key_cache,
        position_cache,
        slot_mapping[:num_updates],
        key_rows[:num_updates],
        position_rows[:num_updates],
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(3),
        position_cache.stride(0),
        position_cache.stride(1),
        position_cache.stride(3),
        key_rows.stride(0),
        key_rows.stride(1),
        position_rows.stride(0),
        position_rows.stride(1),
        num_updates,
        key_cache.shape[0],
        PAGE_SIZE=key_cache.shape[1],
        KEY_WIDTH=key_cache.shape[3],
        POSITION_WIDTH=position_cache.shape[3],
        BLOCK_D=triton.next_power_of_2(max(key_cache.shape[3], position_cache.shape[3])),
        num_warps=4,
    )


def qsa_store_kv_rows(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_rows: torch.Tensor,
    value_rows: torch.Tensor,
) -> None:
    """Store main QSA key and value rows in one launch."""
    if key_rows.ndim == 3:
        if key_rows.shape[1] != 1 or value_rows.shape[1] != 1:
            raise ValueError("QSA combined K/V store requires one local KV head")
        key_rows = key_rows[:, 0]
        value_rows = value_rows[:, 0]
    if key_cache.ndim != 4 or key_cache.shape[2] != 1:
        raise ValueError("QSA combined K/V cache requires a singleton-head layout")
    if value_cache.shape != key_cache.shape or value_rows.shape != key_rows.shape:
        raise ValueError("QSA combined K/V key and value shapes must match")
    if key_rows.ndim != 2 or key_rows.shape[1] != key_cache.shape[3]:
        raise ValueError("QSA combined K/V rows have an incompatible width")
    if key_rows.dtype != key_cache.dtype or value_rows.dtype != value_cache.dtype:
        raise ValueError("QSA combined K/V rows must match their cache dtypes")
    slots = slot_mapping.reshape(-1)
    num_updates = min(slots.shape[0], key_rows.shape[0], value_rows.shape[0])
    if num_updates == 0:
        return
    _store_qsa_key_position_rows_kernel[(num_updates,)](
        key_cache,
        value_cache,
        slots[:num_updates],
        key_rows[:num_updates],
        value_rows[:num_updates],
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(3),
        key_rows.stride(0),
        key_rows.stride(1),
        value_rows.stride(0),
        value_rows.stride(1),
        num_updates,
        key_cache.shape[0],
        PAGE_SIZE=key_cache.shape[1],
        KEY_WIDTH=key_cache.shape[3],
        POSITION_WIDTH=value_cache.shape[3],
        BLOCK_D=triton.next_power_of_2(key_cache.shape[3]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,
    raw_positions: torch.Tensor,
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups from the state ring and current token rows."""
    if raw_keys.device.type != "npu" or not HAS_TRITON:
        raise RuntimeError("QSA NPU compression requires Triton")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if raw_positions.shape != (rows, 1, 3) or raw_positions.dtype != torch.int64:
        raise ValueError("QSA raw positions must be [rows, 1, 3] int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if compressor_state_cache.ndim != 4 or compressor_state_cache.shape[2] != 1:
        raise ValueError("QSA compressor-state cache has an invalid shape")
    if (
        compressor_state_cache.shape[1] < compress_ratio
        or compressor_state_cache.shape[3] != raw_keys.shape[2]
        or compressor_state_cache.dtype != raw_keys.dtype
    ):
        raise ValueError("QSA compressor-state cache does not match the compression layout")
    if compressor_state_block_table.ndim != 2 or compressor_state_block_table.shape[1] < 1:
        raise ValueError("QSA compressor-state block table must contain one block per request")
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != compressor_state_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (not all(compressor_state_cache.shape) or not all(compressor_state_block_table.shape)):
        raise ValueError("QSA compressor-state cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_keys.shape[2]),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = compressor_state_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_keys.stride(0),
        raw_keys.stride(2),
        raw_positions.stride(0),
        raw_positions.stride(2),
        compressor_state_cache.stride(0),
        compressor_state_cache.stride(1),
        compressor_state_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        compressor_state_block_table.stride(0),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        compressor_state_cache.shape[0],
        num_requests,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_keys.shape[2],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),
        num_warps=4,
    )
    return pooled, first_positions


__all__ = [
    "pack_qsa_group_indices_npu",
    "qsa_compress_groups_with_ratio",
    "qsa_store_key_position_rows",
    "qsa_store_kv_rows",
    "qsa_store_cache_rows",
]
