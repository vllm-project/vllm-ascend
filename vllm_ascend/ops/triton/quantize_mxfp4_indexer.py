# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MXFP4 query quantization for the DeepSeek V4.1 A5 indexer."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

_INDEXER_WIDTH = 128
_GROUP_SIZE = 32
_GROUP_COUNT = _INDEXER_WIDTH // _GROUP_SIZE
_PACKED_WIDTH = _INDEXER_WIDTH // 2


@triton.jit
def _select_group_value(group, value0, value1, value2, value3):
    return tl.where(
        group == 0,
        value0,
        tl.where(group == 1, value1, tl.where(group == 2, value2, value3)),
    )


@triton.jit
def _ceil_e8m0_scale_byte(scale):
    """Encode ceil(log2(scale)) as a finite E8M0 exponent byte."""
    bits = scale.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exponent += mantissa != 0
    return tl.minimum(tl.maximum(exponent, 1), 254)


@triton.jit
def _fp4_e2m1_code(value):
    """Match the reference's lower-code tie break and signed-zero encoding."""
    magnitude = tl.minimum(tl.abs(value), 6.0)
    code = (magnitude > 0.25).to(tl.uint8)
    code += (magnitude > 0.75).to(tl.uint8)
    code += (magnitude > 1.25).to(tl.uint8)
    code += (magnitude > 1.75).to(tl.uint8)
    code += (magnitude > 2.5).to(tl.uint8)
    code += (magnitude > 3.5).to(tl.uint8)
    code += (magnitude > 5.0).to(tl.uint8)
    # Keep a negative-zero code for negative values quantized to zero, exactly
    # like ``_pack_fp4``. IEEE negative zero itself compares equal to zero.
    sign = (value < 0).to(tl.uint8)
    return code | (sign << 3)


@triton.jit
def _quantize_mxfp4_indexer_kernel(
    input_ptr,
    data_ptr,
    scale_ptr,
    BLOCK_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_WIDTH)
    values = tl.load(input_ptr + row * BLOCK_WIDTH + offsets).to(tl.float32)
    magnitudes = tl.abs(values)

    amax0 = tl.max(tl.where(offsets < GROUP_SIZE, magnitudes, 0.0), axis=0)
    amax1 = tl.max(
        tl.where((offsets >= GROUP_SIZE) & (offsets < 2 * GROUP_SIZE), magnitudes, 0.0),
        axis=0,
    )
    amax2 = tl.max(
        tl.where((offsets >= 2 * GROUP_SIZE) & (offsets < 3 * GROUP_SIZE), magnitudes, 0.0),
        axis=0,
    )
    amax3 = tl.max(tl.where(offsets >= 3 * GROUP_SIZE, magnitudes, 0.0), axis=0)

    # The eager golden computes ceil(log2(max(amax, 6*2^-126) / 6)).
    # Dividing the minimum by six yields the smallest finite E8M0 scale.
    min_amax = 6.0 * 2.0**-126
    scale0 = tl.maximum(amax0, min_amax) / 6.0
    scale1 = tl.maximum(amax1, min_amax) / 6.0
    scale2 = tl.maximum(amax2, min_amax) / 6.0
    scale3 = tl.maximum(amax3, min_amax) / 6.0

    exponent0 = _ceil_e8m0_scale_byte(scale0)
    exponent1 = _ceil_e8m0_scale_byte(scale1)
    exponent2 = _ceil_e8m0_scale_byte(scale2)
    exponent3 = _ceil_e8m0_scale_byte(scale3)
    packed_scale = exponent0 | (exponent1 << 8) | (exponent2 << 16) | (exponent3 << 24)
    tl.store(scale_ptr + row, packed_scale)

    pair_offsets = tl.arange(0, BLOCK_WIDTH // 2)
    even_offsets = pair_offsets * 2
    odd_offsets = even_offsets + 1
    even_group = even_offsets // GROUP_SIZE
    odd_group = odd_offsets // GROUP_SIZE
    even_exponent = _select_group_value(even_group, exponent0, exponent1, exponent2, exponent3)
    odd_exponent = _select_group_value(odd_group, exponent0, exponent1, exponent2, exponent3)
    even_scale = (even_exponent << 23).to(tl.float32, bitcast=True)
    odd_scale = (odd_exponent << 23).to(tl.float32, bitcast=True)
    even = tl.load(input_ptr + row * BLOCK_WIDTH + even_offsets).to(tl.float32) / even_scale
    odd = tl.load(input_ptr + row * BLOCK_WIDTH + odd_offsets).to(tl.float32) / odd_scale
    even_code = _fp4_e2m1_code(even)
    odd_code = _fp4_e2m1_code(odd)
    packed = (even_code & 0x0F) | ((odd_code & 0x0F) << 4)
    tl.store(data_ptr + row * (BLOCK_WIDTH // 2) + pair_offsets, packed)


@triton.jit
def _write_mxfp4_indexer_cache_kernel(
    input_ptr,
    slots_ptr,
    data_cache_ptr,
    scale_cache_ptr,
    DATA_PAGE_STRIDE,
    DATA_ROW_STRIDE,
    SCALE_PAGE_STRIDE,
    SCALE_ROW_STRIDE,
    BLOCK_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
):
    row = tl.program_id(0)
    # Cache page strides can make the byte offset exceed INT32_MAX even when
    # the page index itself fits in int32 (for example, page 16384 with a
    # 128-KiB page starts at byte 2**31).  Widen before multiplying by the
    # strides so Triton performs the address calculation in 64 bits.
    page = tl.load(slots_ptr + row * 2).to(tl.int64)
    slot = tl.load(slots_ptr + row * 2 + 1).to(tl.int64)
    valid = (page >= 0) & (slot >= 0)
    offsets = tl.arange(0, BLOCK_WIDTH)
    values = tl.load(input_ptr + row * BLOCK_WIDTH + offsets).to(tl.float32)
    magnitudes = tl.abs(values)

    amax0 = tl.max(tl.where(offsets < GROUP_SIZE, magnitudes, 0.0), axis=0)
    amax1 = tl.max(
        tl.where((offsets >= GROUP_SIZE) & (offsets < 2 * GROUP_SIZE), magnitudes, 0.0),
        axis=0,
    )
    amax2 = tl.max(
        tl.where((offsets >= 2 * GROUP_SIZE) & (offsets < 3 * GROUP_SIZE), magnitudes, 0.0),
        axis=0,
    )
    amax3 = tl.max(tl.where(offsets >= 3 * GROUP_SIZE, magnitudes, 0.0), axis=0)
    min_amax = 6.0 * 2.0**-126
    exponent0 = _ceil_e8m0_scale_byte(tl.maximum(amax0, min_amax) / 6.0)
    exponent1 = _ceil_e8m0_scale_byte(tl.maximum(amax1, min_amax) / 6.0)
    exponent2 = _ceil_e8m0_scale_byte(tl.maximum(amax2, min_amax) / 6.0)
    exponent3 = _ceil_e8m0_scale_byte(tl.maximum(amax3, min_amax) / 6.0)

    group_offsets = tl.arange(0, GROUP_COUNT)
    exponents = _select_group_value(
        group_offsets,
        exponent0,
        exponent1,
        exponent2,
        exponent3,
    )
    scale_base = page * SCALE_PAGE_STRIDE + slot * SCALE_ROW_STRIDE
    tl.store(scale_cache_ptr + scale_base + group_offsets, exponents, mask=valid)

    pair_offsets = tl.arange(0, BLOCK_WIDTH // 2)
    even_offsets = pair_offsets * 2
    odd_offsets = even_offsets + 1
    even_exponent = _select_group_value(
        even_offsets // GROUP_SIZE,
        exponent0,
        exponent1,
        exponent2,
        exponent3,
    )
    odd_exponent = _select_group_value(
        odd_offsets // GROUP_SIZE,
        exponent0,
        exponent1,
        exponent2,
        exponent3,
    )
    even_scale = (even_exponent << 23).to(tl.float32, bitcast=True)
    odd_scale = (odd_exponent << 23).to(tl.float32, bitcast=True)
    even = tl.load(input_ptr + row * BLOCK_WIDTH + even_offsets).to(tl.float32) / even_scale
    odd = tl.load(input_ptr + row * BLOCK_WIDTH + odd_offsets).to(tl.float32) / odd_scale
    packed = (_fp4_e2m1_code(even) & 0x0F) | ((_fp4_e2m1_code(odd) & 0x0F) << 4)
    data_base = page * DATA_PAGE_STRIDE + slot * DATA_ROW_STRIDE
    tl.store(data_cache_ptr + data_base + pair_offsets, packed, mask=valid)


def quantize_mxfp4_indexer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize contiguous 128-wide rows to packed E2M1 and E8M0 scales.

    The result preserves the caller's leading dimensions: data has a final
    width of 64 bytes and scale has a final width of four bytes.
    """
    if x.shape[-1] != _INDEXER_WIDTH:
        raise ValueError(f"MXFP4 indexer width must be {_INDEXER_WIDTH}, got {x.shape[-1]}")
    contiguous = x.contiguous()
    rows = contiguous.numel() // _INDEXER_WIDTH
    data_i8 = torch.empty((rows, _PACKED_WIDTH), dtype=torch.int8, device=x.device)
    packed_scale = torch.empty((rows,), dtype=torch.int32, device=x.device)
    if rows:
        _quantize_mxfp4_indexer_kernel[(rows,)](
            contiguous,
            data_i8,
            packed_scale,
            BLOCK_WIDTH=_INDEXER_WIDTH,
            GROUP_SIZE=_GROUP_SIZE,
        )
    leading_shape = tuple(x.shape[:-1])
    data = data_i8.view(torch.uint8).reshape(*leading_shape, _PACKED_WIDTH)
    scale = packed_scale.view(torch.uint8).reshape(*leading_shape, _GROUP_COUNT)
    return data, scale


def write_mxfp4_indexer_cache(
    x: torch.Tensor,
    slots: torch.Tensor,
    data_cache: torch.Tensor,
    scale_cache: torch.Tensor,
) -> None:
    """Quantize ``x[T,128]`` and write strided A5 cache planes in one launch."""
    if x.ndim != 2 or x.shape[1] != _INDEXER_WIDTH:
        raise ValueError(f"index K must be [T,{_INDEXER_WIDTH}], got {tuple(x.shape)}")
    if slots.shape != (x.shape[0], 2) or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"slots must be INT32/64[T,2], got {slots.dtype}{tuple(slots.shape)}")
    expected_data_tail = (1, _PACKED_WIDTH)
    expected_scale_tail = (1, _GROUP_COUNT)
    if data_cache.ndim != 4 or tuple(data_cache.shape[2:]) != expected_data_tail:
        raise ValueError(f"data cache must end in {expected_data_tail}, got {tuple(data_cache.shape)}")
    if scale_cache.ndim != 4 or tuple(scale_cache.shape[2:]) != expected_scale_tail:
        raise ValueError(f"scale cache must end in {expected_scale_tail}, got {tuple(scale_cache.shape)}")
    if data_cache.dtype != torch.uint8 or scale_cache.dtype != torch.uint8:
        raise ValueError("A5 index cache data and scale planes must be uint8")
    if x.shape[0]:
        _write_mxfp4_indexer_cache_kernel[(x.shape[0],)](
            x.contiguous(),
            slots.contiguous(),
            data_cache,
            scale_cache,
            data_cache.stride(0),
            data_cache.stride(1),
            scale_cache.stride(0),
            scale_cache.stride(1),
            BLOCK_WIDTH=_INDEXER_WIDTH,
            GROUP_SIZE=_GROUP_SIZE,
            GROUP_COUNT=_GROUP_COUNT,
        )
