# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build A5 cache slot metadata in one NPU launch."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_actual_reqs",
        "num_actual_tokens",
        "skip_update",
    ]
)
def _build_a5_slot_mapping_kernel(
    slots_ptr,
    positions_ptr,
    query_start_loc_ptr,
    coordinates_ptr,
    flat_slots_ptr,
    num_tokens,
    num_actual_reqs,
    num_actual_tokens,
    skip_update,
    slot_stride,
    position_stride,
    query_start_stride,
    PAGE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tokens < num_tokens
    slots = tl.load(slots_ptr + tokens * slot_stride, mask=mask, other=-1).to(tl.int64)
    request_end = tl.load(query_start_loc_ptr + num_actual_reqs * query_start_stride).to(tl.int64)
    valid_end = tl.minimum(request_end, num_actual_tokens)
    # Full graphs keep a fixed token carrier whose padded tail may still hold
    # slot IDs from an earlier replay.  Never let those rows write any A5 cache
    # plane, including the uncompressed SWA/C1 planes.
    valid = mask & (tokens < valid_end) & (slots >= 0)

    if COMPRESS_RATIO == 2:
        valid &= ((slots + 1) % 2) == 0
        positions = tl.load(
            positions_ptr + tokens * position_stride,
            mask=mask,
            other=0,
        ).to(tl.int64)
        valid &= ((positions % 2) == 1) & (skip_update == 0)
        physical = slots // 2
    else:
        physical = slots

    safe_physical = tl.maximum(physical, 0)
    flat_slots = tl.where(valid, physical, -1)
    pages = tl.where(valid, safe_physical // PAGE_SIZE, -1)
    rows = tl.where(valid, safe_physical % PAGE_SIZE, -1)
    tl.store(flat_slots_ptr + tokens, flat_slots, mask=mask)
    tl.store(coordinates_ptr + tokens * 2, pages, mask=mask)
    tl.store(coordinates_ptr + tokens * 2 + 1, rows, mask=mask)


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_actual_reqs",
        "num_actual_tokens",
        "skip_update",
    ]
)
def _build_a5_slot_mapping_batch_kernel(
    slots_ptr,
    group_ids_ptr,
    positions_ptr,
    query_start_loc_ptr,
    coordinates_ptr,
    flat_slots_ptr,
    num_tokens,
    num_actual_reqs,
    num_actual_tokens,
    skip_update,
    slots_group_stride,
    slots_token_stride,
    position_stride,
    query_start_stride,
    coordinates_group_stride,
    coordinates_token_stride,
    flat_group_stride,
    flat_token_stride,
    PAGE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    layout = tl.program_id(1)
    tokens = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tokens < num_tokens
    group = tl.load(group_ids_ptr + layout).to(tl.int64)
    slots = tl.load(
        slots_ptr + group * slots_group_stride + tokens * slots_token_stride,
        mask=mask,
        other=-1,
    ).to(tl.int64)
    request_end = tl.load(query_start_loc_ptr + num_actual_reqs * query_start_stride).to(tl.int64)
    valid_end = tl.minimum(request_end, num_actual_tokens)
    valid = mask & (tokens < valid_end) & (slots >= 0)

    if COMPRESS_RATIO == 2:
        valid &= ((slots + 1) % 2) == 0
        positions = tl.load(
            positions_ptr + tokens * position_stride,
            mask=mask,
            other=0,
        ).to(tl.int64)
        valid &= ((positions % 2) == 1) & (skip_update == 0)
        physical = slots // 2
    else:
        physical = slots

    safe_physical = tl.maximum(physical, 0)
    flat_slots = tl.where(valid, physical, -1)
    pages = tl.where(valid, safe_physical // PAGE_SIZE, -1)
    rows = tl.where(valid, safe_physical % PAGE_SIZE, -1)
    coordinate_offsets = layout * coordinates_group_stride + tokens * coordinates_token_stride
    flat_offsets = layout * flat_group_stride + tokens * flat_token_stride
    tl.store(flat_slots_ptr + flat_offsets, flat_slots, mask=mask)
    tl.store(coordinates_ptr + coordinate_offsets, pages, mask=mask)
    tl.store(coordinates_ptr + coordinate_offsets + 1, rows, mask=mask)


def build_a5_slot_mapping(
    slots: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_tokens: int,
    num_actual_reqs: int,
    num_actual_tokens: int,
    page_size: int,
    compress_ratio: int,
    *,
    skip_update: bool = False,
    coordinates_output: torch.Tensor,
    flat_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish coordinate and flat A5 cache addresses together.

    This replaces the eager GE/Add/Remainder/And/Div/Where/Copy chain used by
    every KV cache group. Ratio-2 completion and padded-row masking follow the
    existing metadata builder contract exactly.
    """
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")
    if num_tokens:
        block_size = 256
        _build_a5_slot_mapping_kernel[(triton.cdiv(num_tokens, block_size),)](
            slots,
            positions,
            query_start_loc,
            coordinates_output,
            flat_output,
            num_tokens,
            num_actual_reqs,
            num_actual_tokens,
            int(skip_update),
            slots.stride(0),
            positions.stride(0),
            query_start_loc.stride(0),
            PAGE_SIZE=page_size,
            COMPRESS_RATIO=compress_ratio,
            BLOCK_SIZE=block_size,
        )
    return coordinates_output, flat_output


def build_a5_slot_mapping_batch(
    slots: torch.Tensor,
    group_ids: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_tokens: int,
    num_actual_reqs: int,
    num_actual_tokens: int,
    page_size: int,
    compress_ratio: int,
    *,
    skip_update: bool = False,
    coordinates_output: torch.Tensor,
    flat_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one cache-address layout for several framework cache groups."""
    if slots.ndim != 2:
        raise ValueError("batched slot mappings must have shape [groups, tokens]")
    if group_ids.ndim != 1:
        raise ValueError("group_ids must be one-dimensional")
    num_layouts = group_ids.shape[0]
    if coordinates_output.shape[:2] != (num_layouts, num_tokens):
        raise ValueError("coordinates_output must have shape [layouts, tokens, 2]")
    if flat_output.shape != (num_layouts, num_tokens):
        raise ValueError("flat_output must have shape [layouts, tokens]")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")
    if num_tokens and num_layouts:
        block_size = 256
        _build_a5_slot_mapping_batch_kernel[(triton.cdiv(num_tokens, block_size), num_layouts)](
            slots,
            group_ids,
            positions,
            query_start_loc,
            coordinates_output,
            flat_output,
            num_tokens,
            num_actual_reqs,
            num_actual_tokens,
            int(skip_update),
            slots.stride(0),
            slots.stride(1),
            positions.stride(0),
            query_start_loc.stride(0),
            coordinates_output.stride(0),
            coordinates_output.stride(1),
            flat_output.stride(0),
            flat_output.stride(1),
            PAGE_SIZE=page_size,
            COMPRESS_RATIO=compress_ratio,
            BLOCK_SIZE=block_size,
        )
    return coordinates_output, flat_output
