# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Detached candidate: one byte-preserving, page-strided indexer K/scale store.

The caller owns valid, nonoverlapping logical slots, as in the original PA
scatter. Current cache pointers, strides, slot values and payloads are read on
every invocation. Importing this module changes no live backend or call site.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _write_indexer_c8(
    keys,
    scale_bits,
    values,
    value_scale_bits,
    slots,
    PAGE_ROWS: tl.constexpr,
    PAGES: tl.constexpr,
    KEY_PAGE_STRIDE: tl.constexpr,
    KEY_ROW_STRIDE: tl.constexpr,
    SCALE_PAGE_STRIDE: tl.constexpr,
    SCALE_ROW_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    logical = tl.load(slots + row).to(tl.int64)
    valid = (logical >= 0) & (logical < PAGES * PAGE_ROWS)
    # Widen before multiplication: scale addresses can exceed INT32_MAX.
    safe = tl.maximum(logical, 0)
    page, offset = safe // PAGE_ROWS, safe % PAGE_ROWS
    column = tl.arange(0, TILE)
    payload = tl.load(values + row * WIDTH + column, mask=valid & (column < WIDTH), other=0)
    tl.store(keys + page * KEY_PAGE_STRIDE + offset * KEY_ROW_STRIDE + column, payload, mask=valid & (column < WIDTH))
    scale = tl.load(value_scale_bits + row, mask=valid, other=0)
    tl.store(scale_bits + page * SCALE_PAGE_STRIDE + offset * SCALE_ROW_STRIDE, scale, mask=valid)


def try_write(key, scale, values, scales, slots) -> bool:
    """Use only the qualified C8 layout, otherwise retain the original path."""
    if key.device.type != "npu" or key.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
        return False
    if key.ndim != 4 or scale.ndim != 4:
        return False
    pages, block, heads, width = key.shape
    if (
        pages <= 0
        or block <= 0
        or heads != 1
        or width != 128
        or scale.shape != (pages, block, 1, 1)
        or key.stride()[1:] != (width, width, 1)
        or scale.stride()[1:] != (1, 1, 1)
        or key.stride(0) < block * width
        or scale.stride(0) < block
    ):
        return False
    if (
        slots.ndim != 1
        or slots.dtype not in (torch.int32, torch.int64)
        or not slots.is_contiguous()
        or not values.is_contiguous()
        or not scales.is_contiguous()
        or values.dtype != key.dtype
        or scales.dtype != scale.dtype
        or any(t.device != key.device for t in (scale, values, scales, slots))
        or values.numel() != slots.numel() * width
        or scales.numel() != slots.numel()
    ):
        return False
    if slots.numel() == 0:
        return True
    # Views preserve every FP8 bit, FP32 NaN payload and signed zero.
    _write_indexer_c8[(slots.numel(),)](
        key.view(torch.uint8),
        scale.view(torch.int32),
        values.view(torch.uint8),
        scales.view(torch.int32),
        slots,
        block,
        pages,
        key.stride(0),
        key.stride(1),
        scale.stride(0),
        scale.stride(1),
        width,
        triton.next_power_of_2(width),
    )
    return True
