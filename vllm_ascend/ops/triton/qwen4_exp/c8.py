# SPDX-License-Identifier: Apache-2.0
"""Strided C8 cache writes and bounded INT8 packing for native GBSA."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.models.qwen4_exp.quantization import C8_QUANT_MAX, C8_SCALE_MIN, qsa_c8_cache_views


@triton.jit
def _store_c8_rows(
    rows,
    slots,
    codes,
    scales,
    static_scale,
    row_stride,
    head_stride,
    dim_stride,
    code_page_stride,
    code_token_stride,
    code_head_stride,
    scale_page_stride,
    scale_token_stride,
    scale_head_stride,
    CAPACITY: tl.constexpr,
    PAGE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    SCALE_MIN: tl.constexpr,
    STATIC_V: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dim = tl.arange(0, BLOCK)
    slot = tl.load(slots + row).to(tl.int64)
    valid = (slot >= 0) & (slot < CAPACITY)
    # Ascend may validate addresses before evaluating masks, including graph
    # padding slots. Clamp pointers as well as masking the actual writes.
    safe_slot = tl.minimum(tl.maximum(slot, 0), CAPACITY - 1)
    page, token = safe_slot // PAGE, safe_slot % PAGE
    values = tl.load(rows + row * row_stride + head * head_stride + dim * dim_stride, dim < DIM, other=0).to(tl.float32)
    if STATIC_V:
        scale = tl.load(static_scale + head * DIM + dim, dim < DIM, other=1)
    else:
        scale = tl.maximum(tl.max(tl.abs(values), 0) / QUANT_MAX, SCALE_MIN)
    quantized = tl.extra.cann.libdevice.nearbyint(values / scale)
    quantized = tl.minimum(tl.maximum(quantized, -128 if STATIC_V else -QUANT_MAX), QUANT_MAX).to(tl.int8)
    tl.store(
        codes + page * code_page_stride + token * code_token_stride + head * code_head_stride + dim,
        quantized,
        valid & (dim < DIM),
    )
    if not STATIC_V:
        tl.store(
            scales + page * scale_page_stride + token * scale_token_stride + head * scale_head_stride, scale, valid
        )


def store_qsa_c8_rows(
    cache: torch.Tensor,
    slots: torch.Tensor,
    rows: torch.Tensor,
    *,
    indexer: bool = False,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Quantize normalized/rotated rows directly into a scheduler-owned page."""
    codes, scales = qsa_c8_cache_views(cache, indexer=indexer)
    if rows.ndim != 3 or rows.shape[1:] != codes.shape[2:]:
        raise ValueError("QSA C8 rows must match cache heads and logical head dimension")
    if cache.device.type != "npu" or rows.device != cache.device or slots.device != cache.device:
        raise ValueError("QSA C8 store requires rows, slots and cache on the same NPU")
    if rows.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("QSA C8 store requires BF16/FP16/FP32 input")
    if slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("QSA C8 store requires integer slots")
    if v_scale is not None and (
        indexer
        or v_scale.shape != rows.shape[1:]
        or v_scale.dtype != torch.float32
        or v_scale.device != cache.device
        or not v_scale.is_contiguous()
    ):
        raise ValueError("QSA V store requires dense FP32 [KV heads,dim] calibrated scales")
    slots = slots.reshape(-1)
    updates = min(slots.shape[0], rows.shape[0])
    if not updates:
        return
    _store_c8_rows[(updates, rows.shape[1])](
        rows,
        slots,
        codes,
        scales,
        scales if v_scale is None else v_scale,
        *rows.stride(),
        *codes.stride()[:3],
        *scales.stride(),
        CAPACITY=codes.shape[0] * codes.shape[1],
        PAGE=codes.shape[1],
        HEADS=codes.shape[2],
        DIM=codes.shape[-1],
        BLOCK=triton.next_power_of_2(codes.shape[-1]),
        QUANT_MAX=C8_QUANT_MAX,
        SCALE_MIN=C8_SCALE_MIN,
        STATIC_V=v_scale is not None,
        num_warps=4,
    )


@triton.jit
def _pack_c8_rows(
    codes,
    scales,
    groups,
    counts,
    visible,
    table,
    output,
    output_scales,
    code_page_stride,
    code_token_stride,
    code_head_stride,
    scale_page_stride,
    scale_token_stride,
    scale_head_stride,
    group_head_stride,
    group_row_stride,
    count_head_stride,
    count_row_stride,
    table_row_stride,
    PAGES: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    GROUPS: tl.constexpr,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COPY_SCALES: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.program_id(1)
    head = tl.program_id(2)
    dim = tl.arange(0, BLOCK_D)
    group_slot = token // 4
    count = tl.load(counts + head * count_head_stride + row * count_row_stride)
    visible_length = tl.load(visible + row)
    group = tl.load(groups + head * group_head_stride + row * group_row_stride + tl.minimum(group_slot, GROUPS - 1))
    logical = group.to(tl.int64) * 4 + token % 4
    column = tl.minimum(tl.maximum(logical // 128, 0), TABLE_COLS - 1)
    page = tl.load(table + row * table_row_stride + column).to(tl.int64)
    valid = (group_slot < count) & (group_slot < GROUPS) & (group >= 0) & (logical >= 0) & (logical < visible_length)
    valid = valid & (logical // 128 < TABLE_COLS) & (page >= 0) & (page < PAGES)
    safe_page = tl.minimum(tl.maximum(page, 0), PAGES - 1)
    offset = tl.maximum(logical, 0) % 128
    values = tl.load(
        codes + safe_page * code_page_stride + offset * code_token_stride + head * code_head_stride + dim,
        valid & (dim < DIM),
        other=0,
    )
    tl.store(output + ((row * TOKENS + token) * HEADS + head) * DIM + dim, values, dim < DIM)
    if COPY_SCALES:
        scale = tl.load(
            scales + safe_page * scale_page_stride + offset * scale_token_stride + head * scale_head_stride,
            valid,
            other=0,
        )
        # GBSA's paged K scale is [physical_page,head,128], irrespective of KV layout.
        scale_offset = ((row * (TOKENS // 128) + token // 128) * HEADS + head) * 128 + token % 128
        tl.store(output_scales + scale_offset, scale)


def pack_qsa_c8_rows(
    cache: torch.Tensor,
    groups: torch.Tensor,
    counts: torch.Tensor,
    visible: torch.Tensor,
    table: torch.Tensor,
    output: torch.Tensor,
    output_scales: torch.Tensor | None = None,
) -> None:
    """Copy selected INT8 codes and optional K scales without dequantization."""
    codes, scales = qsa_c8_cache_views(cache)
    if cache.device.type != "npu" or any(
        tensor.device != cache.device for tensor in (groups, counts, visible, table, output)
    ):
        raise ValueError("QSA C8 pack requires all tensors on the same NPU")
    if cache.shape[1] != 128 or output.ndim != 4 or output.shape[2:] != codes.shape[2:]:
        raise ValueError("QSA C8 pack requires 128-token pages and [rows,tokens,heads,dim] output")
    rows, tokens, heads, dim = output.shape
    if groups.shape[:2] != (heads, rows) or counts.shape != (heads, rows):
        raise ValueError("QSA C8 pack group metadata does not match output")
    if table.ndim != 2 or table.shape[0] != rows or not table.shape[1] or visible.shape != (rows,):
        raise ValueError("QSA C8 pack requires per-row page tables and visible lengths")
    if tokens % 128 or tokens < groups.shape[2] * 4 or not output.is_contiguous():
        raise ValueError("QSA C8 pack output must be dense and cover whole selected pages")
    if output.dtype != torch.int8 or table.stride(1) != 1:
        raise ValueError("QSA C8 pack requires INT8 output and dense page-table columns")
    if output_scales is not None and (
        output_scales.shape != (rows * tokens // 128, heads, 128)
        or output_scales.dtype != torch.float32
        or not output_scales.is_contiguous()
        or output_scales.device != cache.device
    ):
        raise ValueError("QSA C8 pack requires dense FP32 [pages,heads,128] K scales")
    if any(tensor.dtype != torch.int32 for tensor in (groups, counts, visible, table)):
        raise ValueError("QSA C8 pack requires INT32 sparse metadata")
    if rows:
        _pack_c8_rows[(rows, tokens, heads)](
            codes,
            scales,
            groups,
            counts,
            visible,
            table,
            output,
            scales if output_scales is None else output_scales,
            *codes.stride()[:3],
            *scales.stride(),
            groups.stride(0),
            groups.stride(1),
            *counts.stride(),
            table.stride(0),
            PAGES=codes.shape[0],
            TABLE_COLS=table.shape[1],
            GROUPS=groups.shape[2],
            TOKENS=tokens,
            HEADS=heads,
            DIM=dim,
            BLOCK_D=triton.next_power_of_2(dim),
            COPY_SCALES=output_scales is not None,
            num_warps=4,
        )
