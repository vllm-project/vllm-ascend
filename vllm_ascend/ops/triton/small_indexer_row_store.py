# SPDX-License-Identifier: Apache-2.0
"""Copy a small batch of unquantized SFA indexer keys into contiguous paged storage."""

import torch
from vllm.triton_utils import tl, triton

_MAX_ROWS = 32
_MAX_WIDTH = 2048
_ROW_ALIGNMENT_BYTES = 32
_MAX_ELEMENTS = 1 << 31


@triton.jit
def _store_small_indexer_rows_kernel(
    cache_ptr,
    updates_ptr,
    slots_ptr,
    CAPACITY: tl.constexpr,
    WIDTH: tl.constexpr,
    UPDATE_STRIDE: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row)
    if (slot >= 0) & (slot < CAPACITY):
        cols = tl.arange(0, BLOCK_WIDTH)
        values = tl.load(updates_ptr + row * UPDATE_STRIDE + cols, cols < WIDTH, other=0)
        tl.store(cache_ptr + slot.to(tl.int32) * WIDTH + cols, values, cols < WIDTH)


def _supports_layout(cache: torch.Tensor, updates: torch.Tensor, slots: torch.Tensor) -> bool:
    """Check host tensor descriptors only. Bound both address calculations before narrowing."""
    if cache.ndim != 4 or cache.shape[2] != 1 or updates.ndim != 2 or slots.ndim != 1:
        return False
    rows, width = updates.shape
    return (
        0 < rows == slots.numel() <= _MAX_ROWS
        and 0 < width == cache.shape[-1] <= _MAX_WIDTH
        and 0 < cache.numel() <= _MAX_ELEMENTS
        and cache.dtype == updates.dtype
        and cache.dtype in (torch.float16, torch.bfloat16)
        and slots.dtype in (torch.int32, torch.int64)
        and cache.device == updates.device == slots.device
        and cache.is_contiguous()
        and slots.is_contiguous()
        and cache.storage_offset() == updates.storage_offset() == slots.storage_offset() == 0
        and updates.stride(1) == 1
        and updates.stride(0) >= width
        and (rows - 1) * updates.stride(0) + width <= _MAX_ELEMENTS
        and width * cache.element_size() % _ROW_ALIGNMENT_BYTES == 0
    )


def try_store_small_indexer_rows(cache: torch.Tensor, updates: torch.Tensor, slots: torch.Tensor) -> bool:
    """Write distinct live slots on the current stream, preserving the caller's fallback on a miss."""
    if cache.device.type != "npu" or not _supports_layout(cache, updates, slots):
        return False
    rows, width = updates.shape
    _store_small_indexer_rows_kernel[(rows,)](
        cache, updates, slots, cache.numel() // width, width, updates.stride(0), triton.next_power_of_2(width)
    )
    return True
