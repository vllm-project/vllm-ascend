# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Byte-preserving C8 copies with invocation-owned output and current metadata."""

import torch
from vllm.triton_utils import tl, triton

_NOPE_BYTES = 512
_ROPE_BYTES = 128
_SCALE_BYTES = 16
_ROW_BYTES = _NOPE_BYTES + _ROPE_BYTES + _SCALE_BYTES


@triton.jit
def _scatter_components(
    nope,
    rope,
    scales,
    cache,
    slots,
    valid_rows,
    CACHE_ROWS: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, TILE)
    slot = tl.load(slots + row).to(tl.int64)
    valid = (row < tl.load(valid_rows)) & (slot >= 0) & (slot < CACHE_ROWS)
    a = tl.load(nope + row.to(tl.int64) * 512 + column, mask=valid & (column < 512), other=0)
    b = tl.load(rope + row.to(tl.int64) * 128 + column - 512, mask=valid & (column >= 512) & (column < 640), other=0)
    c = tl.load(scales + row.to(tl.int64) * 16 + column - 640, mask=valid & (column >= 640) & (column < 656), other=0)
    destination = slot // CACHE_BLOCK * CACHE_STRIDE + slot % CACHE_BLOCK * 656
    tl.store(
        cache + destination + column,
        tl.where(column < 512, a, tl.where(column < 640, b, c)),
        mask=valid & (column < 656),
    )


def scatter_c8_components(nope, rope, scales, cache, slots, valid_rows):
    """Copy opaque NoPE/RoPE/scale bytes directly into the current mapped rows.

    The caller retains the original normalization, RoPE and quantization ops.
    Slots follow the existing scatter contract: writable valid slots are unique.
    Padding, invalid slots, page gaps and neighboring layers are never written.
    """
    if nope.dtype != torch.float8_e4m3fn:
        raise ValueError("C8 requires E4M3 NoPE")
    components = (nope, rope, scales)
    if any(t.device != nope.device or not t.is_contiguous() for t in components):
        raise ValueError("C8 components must be contiguous on one device")
    rows = nope.numel() // _NOPE_BYTES
    if nope.numel() != rows * _NOPE_BYTES:
        raise ValueError("C8 requires complete NoPE rows")
    if rope.numel() * rope.element_size() != rows * _ROPE_BYTES:
        raise ValueError("C8 requires 128 RoPE bytes per row")
    if scales.numel() * scales.element_size() != rows * _SCALE_BYTES:
        raise ValueError("C8 requires 16 scale bytes per row")
    if cache.ndim != 3 or cache.shape[-1] != _ROW_BYTES or cache.element_size() != 1:
        raise ValueError("C8 destination must contain packed 656-byte rows")
    pages, block, _ = cache.shape
    if pages <= 0 or block <= 0 or cache.stride(1) != _ROW_BYTES or cache.stride(2) != 1:
        raise ValueError("C8 destination requires dense token rows")
    if cache.stride(0) < block * _ROW_BYTES:
        raise ValueError("C8 pages cannot overlap")
    if (
        slots.ndim != 1
        or slots.numel() != rows
        or slots.stride(0) != 1
        or slots.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("C8 scatter requires contiguous integer slots for each row")
    if valid_rows.numel() != 1 or valid_rows.dtype not in (torch.int32, torch.int64):
        raise ValueError("C8 scatter requires the current device valid-row count")
    if nope.device.type != "npu" or any(t.device != nope.device for t in (cache, slots, valid_rows)):
        raise ValueError("C8 scatter inputs must share one NPU")
    if rows == 0:
        return
    _scatter_components[(rows,)](
        *(t.view(torch.uint8).reshape(rows, width) for t, width in zip(components, (512, 128, 16))),
        cache.view(torch.uint8),
        slots,
        valid_rows,
        pages * block,
        block,
        cache.stride(0),
        1024,
    )


@triton.jit
def _pack_query_lengths(
    query,
    rope,
    packed,
    query_end,
    key_length,
    rows,
    key_rows,
    HEADS: tl.constexpr,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    Q_COL: tl.constexpr,
    R_ROW: tl.constexpr,
    R_HEAD: tl.constexpr,
    R_COL: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    column = tl.arange(0, TILE)
    a = tl.load(query + row.to(tl.int64) * Q_ROW + head * Q_HEAD + column * Q_COL, mask=column < 512, other=0)
    b = tl.load(
        rope + row.to(tl.int64) * R_ROW + head * R_HEAD + (column - 512) * R_COL,
        mask=(column >= 512) & (column < 576),
        other=0,
    )
    tl.store(packed + (row.to(tl.int64) * HEADS + head) * 576 + column, tl.where(column < 512, a, b), mask=column < 576)
    if (row == 0) & (head == 0):
        tl.store(query_end, rows)
        tl.store(key_length, key_rows)


def pack_c8_query_lengths(query, rope, query_end, key_length, key_rows):
    """Pack BF16 as raw bits and write both lengths on the consumer stream.

    Length values are restored by every graph replay. Metadata is owned by the
    caller's current workspace lane; no cross-layer or cross-round cache exists.
    """
    if query.ndim != 3 or query.dtype != torch.bfloat16 or query.shape[-1] != 512:
        raise ValueError("C8 query requires BF16 [rows,heads,512]")
    rows, heads, _ = query.shape
    if min(rows, heads) <= 0 or rope.shape != (rows, heads, 64) or rope.dtype != query.dtype:
        raise ValueError("C8 query requires matching nonempty RoPE rows")
    if any(t.shape != (1,) or t.dtype != torch.int32 for t in (query_end, key_length)):
        raise ValueError("C8 lengths require int32 scalar tensors")
    if query.device.type != "npu" or any(t.device != query.device for t in (rope, query_end, key_length)):
        raise ValueError("C8 query inputs must share one NPU")
    if not 0 < key_rows <= 2**31 - 1:
        raise ValueError("C8 physical capacity must fit int32")
    packed = torch.empty((rows, heads, 576), dtype=query.dtype, device=query.device)
    _pack_query_lengths[(rows, heads)](
        query.view(torch.uint16),
        rope.view(torch.uint16),
        packed.view(torch.uint16),
        query_end,
        key_length,
        rows,
        key_rows,
        heads,
        *query.stride(),
        *rope.stride(),
        1024,
    )
    return packed


@triton.jit
def _zero_empty_rows(
    source,
    indices,
    output,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    TOP_K: tl.constexpr,
    S_ROW: tl.constexpr,
    S_HEAD: tl.constexpr,
    S_COL: tl.constexpr,
    I_ROW: tl.constexpr,
    I_COL: tl.constexpr,
    INDEX_TILE: tl.constexpr,
    VALUE_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, INDEX_TILE)
    selected = tl.load(indices + row.to(tl.int64) * I_ROW + column * I_COL, mask=column < TOP_K, other=-1)
    valid = tl.sum((selected >= 0).to(tl.int32), 0) > 0
    element = tl.arange(0, VALUE_TILE)
    value = tl.load(
        source + row.to(tl.int64) * S_ROW + element // WIDTH * S_HEAD + element % WIDTH * S_COL,
        mask=element < HEADS * WIDTH,
        other=0,
    )
    tl.store(
        output + row.to(tl.int64) * HEADS * WIDTH + element, tl.where(valid, value, 0), mask=element < HEADS * WIDTH
    )


def zero_empty_c8_rows(output, indices):
    """Preserve all valid-row bits; replace every empty row with positive zero."""
    if output.ndim != 3 or output.dtype != torch.bfloat16 or min(output.shape) <= 0:
        raise ValueError("C8 output requires nonempty BF16 [rows,heads,width]")
    if indices.ndim != 2 or indices.shape[0] != output.shape[0] or indices.dtype != torch.int32:
        raise ValueError("C8 output requires current int32 selection rows")
    if not 0 < indices.shape[1] <= 4096 or output.shape[1] * output.shape[2] > 32768:
        raise ValueError("C8 cleanup shape exceeds supported vector tile")
    if output.device.type != "npu" or indices.device != output.device:
        raise ValueError("C8 output and indices must share one NPU")
    result = torch.empty(output.shape, dtype=output.dtype, device=output.device)
    _zero_empty_rows[(output.shape[0],)](
        output.view(torch.uint16),
        indices,
        result.view(torch.uint16),
        output.shape[1],
        output.shape[2],
        indices.shape[1],
        *output.stride(),
        *indices.stride(),
        triton.next_power_of_2(indices.shape[1]),
        triton.next_power_of_2(output.shape[1] * output.shape[2]),
    )
    return result
