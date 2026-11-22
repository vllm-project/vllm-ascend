# SPDX-License-Identifier: Apache-2.0
"""Bounded tree KV transfers over caches with padded physical page strides."""

import math

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_tree_cache_rows_kernel(
    cache_ptr,
    indices_ptr,
    output_ptr,
    index_stride_row,
    index_stride_axis,
    CACHE_STRIDES: tl.constexpr,
    CACHE_SIZES: tl.constexpr,
    ROW_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base = tl.full((), 0, tl.int64)
    valid = tl.full((), True, tl.int1)
    for axis in tl.static_range(len(CACHE_STRIDES)):
        index = tl.load(indices_ptr + row * index_stride_row + axis * index_stride_axis).to(tl.int64)
        valid = valid & (index >= 0) & (index < CACHE_SIZES[axis])
        base += index * CACHE_STRIDES[axis]
    values = tl.load(cache_ptr + base + offsets, mask=valid & (offsets < ROW_ELEMENTS), other=0)
    tl.store(output_ptr + row * ROW_ELEMENTS + offsets, values, mask=offsets < ROW_ELEMENTS)


@triton.jit
def _scatter_tree_cache_rows_kernel(
    cache_ptr,
    indices_ptr,
    source_ptr,
    index_stride_row,
    index_stride_axis,
    CACHE_STRIDES: tl.constexpr,
    CACHE_SIZES: tl.constexpr,
    ROW_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base = tl.full((), 0, tl.int64)
    valid = tl.full((), True, tl.int1)
    for axis in tl.static_range(len(CACHE_STRIDES)):
        index = tl.load(indices_ptr + row * index_stride_row + axis * index_stride_axis).to(tl.int64)
        valid = valid & (index >= 0) & (index < CACHE_SIZES[axis])
        base += index * CACHE_STRIDES[axis]
    values = tl.load(source_ptr + row * ROW_ELEMENTS + offsets, mask=offsets < ROW_ELEMENTS, other=0)
    tl.store(cache_ptr + base + offsets, values, mask=valid & (offsets < ROW_ELEMENTS))


def _validate_tree_cache_rows(cache, indices, rows):
    if indices.ndim != 2 or not 0 < indices.shape[1] < cache.ndim:
        raise ValueError("indices must have shape [rows, indexed cache dimensions]")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must have int32 or int64 dtype")
    dimensions = indices.shape[1]
    expected = (indices.shape[0], *cache.shape[dimensions:])
    if rows.shape != expected:
        raise ValueError(f"rows must have shape {expected}, got {tuple(rows.shape)}")
    if rows.dtype != cache.dtype or rows.device != cache.device or indices.device != cache.device:
        raise ValueError("cache, indices and rows must have matching devices and payload dtypes")
    if not rows.is_contiguous() or not cache[(0,) * dimensions].is_contiguous():
        raise ValueError("cache inner dimensions and transfer rows must be contiguous")
    return dimensions, math.prod(cache.shape[dimensions:])


def gather_tree_cache_rows(cache: torch.Tensor, indices: torch.Tensor, out: torch.Tensor) -> None:
    """Read selected rows directly, without materializing the full cache.

    Only indexed outer dimensions may have physical padding. Invalid indices
    produce zero rows; no invalid address is read. The caller owns dense output
    storage so graph capture and parameter updates can retain its address.
    """
    dimensions, row_elements = _validate_tree_cache_rows(cache, indices, out)
    if indices.shape[0] == 0 or row_elements == 0:
        return
    block_size = 1024
    _gather_tree_cache_rows_kernel[(triton.cdiv(row_elements, block_size), indices.shape[0])](
        cache,
        indices,
        out,
        indices.stride(0),
        indices.stride(1),
        CACHE_STRIDES=tuple(cache.stride()[:dimensions]),
        CACHE_SIZES=tuple(cache.shape[:dimensions]),
        ROW_ELEMENTS=row_elements,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )


def scatter_tree_cache_rows_(cache: torch.Tensor, indices: torch.Tensor, source: torch.Tensor) -> None:
    """Write selected rows directly, leaving physical-page padding untouched.

    Invalid destinations are skipped. Valid destinations must be unique; the
    shared tree transport resolves duplicate ownership before calling this.
    """
    dimensions, row_elements = _validate_tree_cache_rows(cache, indices, source)
    if indices.shape[0] == 0 or row_elements == 0:
        return
    block_size = 1024
    _scatter_tree_cache_rows_kernel[(triton.cdiv(row_elements, block_size), indices.shape[0])](
        cache,
        indices,
        source,
        indices.stride(0),
        indices.stride(1),
        CACHE_STRIDES=tuple(cache.stride()[:dimensions]),
        CACHE_SIZES=tuple(cache.shape[:dimensions]),
        ROW_ELEMENTS=row_elements,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
