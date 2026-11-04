# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused QSA support kernels used around the CANN LI-v2 and GBSA path."""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

_FUSED_INDEX_HEAD_DIM = 128
_FUSED_ROPE_DIM = 64
_FUSED_COMPRESS_RATIO = 4
_FUSED_MROPE_SECTIONS = (11, 11, 10)


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
    norm_weight_ptr=None,
    rotary_cache_ptr=None,
    compressed_cache_ptr=None,
    rotary_cache_rows: tl.constexpr = 0,
    rotary_cache_stride: tl.constexpr = 0,
    compressed_stride_block: tl.constexpr = 0,
    compressed_stride_token: tl.constexpr = 0,
    compressed_stride_dim: tl.constexpr = 0,
    compressed_page_size: tl.constexpr = 1,
    compressed_num_slots: tl.constexpr = 0,
    norm_eps: tl.constexpr = 1e-6,
    FUSE_NORM_STORE: tl.constexpr = False,
    rotary_dim: tl.constexpr = 64,
    mrope_section_h: tl.constexpr = 11,
    mrope_section_w: tl.constexpr = 10,
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

    if not FUSE_NORM_STORE:
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
    if FUSE_NORM_STORE:
        # Keep the original materialization boundaries: pooling, normalization,
        # both rotary products and the final add/sub all round to BF16.
        pooled = (accumulator / COMPRESS_RATIO).to(tl.bfloat16).to(tl.float32)
        weight = tl.load(norm_weight_ptr + dims, mask=dims < HEAD_DIM, other=0).to(tl.float32) + 1.0
        variance = tl.sum(pooled * pooled, axis=0) / HEAD_DIM
        normalized = (pooled * tl.rsqrt(variance + norm_eps) * weight).to(tl.bfloat16)
        half_rope: tl.constexpr = rotary_dim // 2
        offsets = dims % half_rope
        axis = tl.where(
            (offsets % 3 == 1) & (offsets < 3 * mrope_section_h),
            1,
            tl.where((offsets % 3 == 2) & (offsets < 3 * mrope_section_w), 2, 0),
        )
        if LOAD_ROPE_POSITIONS:
            rotary_position = tl.gather(position_values, axis, axis=0)
        else:
            rotary_position = position_values
        rotary_position = tl.minimum(tl.maximum(rotary_position, 0), rotary_cache_rows - 1)
        cache_row = rotary_cache_ptr + rotary_position.to(tl.int64) * rotary_cache_stride
        cos = tl.load(cache_row + offsets, mask=dims < rotary_dim, other=1).to(tl.float32)
        sin = tl.load(cache_row + offsets + half_rope, mask=dims < rotary_dim, other=0).to(tl.float32)
        partner_dims = tl.where(dims < half_rope, dims + half_rope, tl.where(dims < rotary_dim, dims - half_rope, dims))
        partner = tl.gather(normalized, partner_dims, axis=0)
        product_cos = (normalized * cos).to(tl.bfloat16)
        product_sin = (partner * sin).to(tl.bfloat16)
        rotated = (product_cos + tl.where(dims < half_rope, -product_sin, product_sin)).to(tl.bfloat16)
        values = tl.where(dims < rotary_dim, rotated, normalized)
        # Match the old store mask, including zero results for invalid owners.
        valid_slot = (compressed_slot >= 0) & (compressed_slot < compressed_num_slots)
        slot = tl.maximum(compressed_slot, 0).to(tl.int64)
        destination = (
            compressed_cache_ptr
            + slot // compressed_page_size * compressed_stride_block
            + slot % compressed_page_size * compressed_stride_token
            + dims * compressed_stride_dim
        )
        tl.store(destination, values, mask=valid_slot & (dims < HEAD_DIM))
    else:
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
    *,
    norm_weight: torch.Tensor | None = None,
    rotary_cache: torch.Tensor | None = None,
    compressed_cache: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pool groups, optionally normalizing and storing directly into the cache.

    The fused ABI is BF16, head_dim=128, ratio=4, rotary_dim=64 and
    interleaved MRoPE sections [11, 11, 10]. Raw rings remain read-only in
    this launch: their updates must follow completion of all group readers.
    """
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
    fuse_norm_store = compressed_cache is not None
    if fuse_norm_store:
        if norm_weight is None or rotary_cache is None:
            raise ValueError("QSA fused compression requires norm weight and rotary cache")
        if (
            raw_keys.dtype != torch.bfloat16
            or raw_keys.shape[2] != _FUSED_INDEX_HEAD_DIM
            or compress_ratio != _FUSED_COMPRESS_RATIO
        ):
            raise ValueError("QSA fused compression requires BF16 width 128 and ratio 4")
        if norm_weight.shape != (_FUSED_INDEX_HEAD_DIM,) or not norm_weight.is_contiguous():
            raise ValueError("QSA fused compression requires a contiguous 128-element norm weight")
        if (
            rotary_cache.ndim != 2
            or rotary_cache.shape[0] == 0
            or rotary_cache.shape[1] != _FUSED_ROPE_DIM
            or rotary_cache.stride(1) != 1
        ):
            raise ValueError("QSA fused compression requires a rotary cache with width 64")
        if (
            compressed_cache.ndim != 4
            or compressed_cache.shape[2:] != (1, _FUSED_INDEX_HEAD_DIM)
            or not all(compressed_cache.shape)
        ):
            raise ValueError("QSA fused compression requires a nonempty compressed-key cache")
        if any(
            t.dtype != raw_keys.dtype or t.device != raw_keys.device
            for t in (norm_weight, rotary_cache, compressed_cache)
        ):
            raise ValueError("QSA fused compression tensors must share BF16 dtype and device")
        # Unused pointers/strides in the fused specialization; allocate no intermediates.
        pooled, first_positions = raw_keys, raw_positions[:, 0]
    else:
        if norm_weight is not None or rotary_cache is not None:
            raise ValueError("QSA norm inputs require a compressed-key output cache")
        pooled = torch.empty((rows, 1, raw_keys.shape[2]), dtype=raw_keys.dtype, device=raw_keys.device)
        first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return None if fuse_norm_store else (pooled, first_positions)
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
        norm_weight_ptr=norm_weight,
        rotary_cache_ptr=rotary_cache,
        compressed_cache_ptr=compressed_cache,
        rotary_cache_rows=rotary_cache.shape[0] if fuse_norm_store else 0,
        rotary_cache_stride=rotary_cache.stride(0) if fuse_norm_store else 0,
        compressed_stride_block=compressed_cache.stride(0) if fuse_norm_store else 0,
        compressed_stride_token=compressed_cache.stride(1) if fuse_norm_store else 0,
        compressed_stride_dim=compressed_cache.stride(3) if fuse_norm_store else 0,
        compressed_page_size=compressed_cache.shape[1] if fuse_norm_store else 1,
        compressed_num_slots=compressed_cache.shape[0] * compressed_cache.shape[1] if fuse_norm_store else 0,
        norm_eps=norm_eps,
        FUSE_NORM_STORE=fuse_norm_store,
        rotary_dim=_FUSED_ROPE_DIM,
        mrope_section_h=_FUSED_MROPE_SECTIONS[1],
        mrope_section_w=_FUSED_MROPE_SECTIONS[2],
        num_warps=4,
    )
    return None if fuse_norm_store else (pooled, first_positions)


__all__ = [
    "pack_qsa_group_indices_npu",
    "qsa_compress_groups_with_ratio",
    "qsa_store_cache_rows",
]
