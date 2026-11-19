# SPDX-License-Identifier: Apache-2.0
"""C8 query preparation and strided cache writes for QSA and its indexer."""

import math

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.models.qwen4_exp.quantization import (
    C8_QUANT_MAX,
    C8_SCALE_MIN,
    qsa_c8_indexer_cache_views,
    quantize_qsa_rows,
)
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

QLI_V2_QUERY_HEADS = 64
_QLI_V2_FUSED_MIN_TOKENS = 4


@triton.jit
def _quantize_c8_query(
    query,
    codes,
    scales,
    num_rows,
    blocks_per_core,
    DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    SCALE_MIN: tl.constexpr,
):
    channels = tl.arange(0, DIM)
    for block in range(blocks_per_core):
        rows = (tl.program_id(0) * blocks_per_core + block) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        valid = rows < num_rows
        offsets = rows[:, None] * DIM + channels[None, :]
        values = tl.load(query + offsets, valid[:, None], other=0).to(tl.float32)
        # Match the PyTorch FP32 scale and division so halfway values reach
        # nearbyint with the same round-to-nearest-even semantics.
        scale = tl.maximum(tl.div_rn(tl.max(tl.abs(values), axis=1), QUANT_MAX), SCALE_MIN)
        rounded = tl.extra.cann.libdevice.nearbyint(tl.div_rn(values, scale[:, None]))
        codes_i8 = tl.minimum(tl.maximum(rounded, -QUANT_MAX), QUANT_MAX).to(tl.int8)
        tl.store(codes + offsets, codes_i8, valid[:, None])
        tl.store(scales + rows, scale, valid)


def quantize_qsa_query(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse C8 query quantization with the reference FP32 division and rounding."""
    if (
        query.device.type != "npu"
        or query.ndim != 3
        or query.dtype != torch.bfloat16
        or query.shape[-1] not in (128, 256)
        or not query.numel()
    ):
        return quantize_qsa_rows(query)
    tokens, heads, dim = query.shape
    codes = torch.empty(query.shape, dtype=torch.int8, device=query.device)
    scales = torch.empty((tokens, heads), dtype=torch.float32, device=query.device)
    rows = tokens * heads
    block_rows = 16
    blocks = triton.cdiv(rows, block_rows)
    cores = min(blocks, get_vectorcore_num())
    _quantize_c8_query[(cores,)](
        query.contiguous(),
        codes,
        scales,
        rows,
        triton.cdiv(blocks, cores),
        DIM=dim,
        BLOCK_ROWS=block_rows,
        QUANT_MAX=float(C8_QUANT_MAX),
        SCALE_MIN=C8_SCALE_MIN,
    )
    return codes, scales


@triton.jit
def _pad_qli_query(
    source_codes,
    source_scales,
    padded_codes,
    padded_scales,
    weights,
    num_tokens,
    rows_per_core,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    HEAD_WEIGHT: tl.constexpr,
):
    heads = tl.arange(0, PADDED_HEADS)
    channels = tl.arange(0, DIM)
    for offset in range(rows_per_core):
        token = tl.program_id(0) + offset * tl.num_programs(0)
        if token < num_tokens:
            active = heads < HEADS
            source = token * HEADS + tl.minimum(heads, HEADS - 1)
            codes = tl.load(source_codes + source[:, None] * DIM + channels[None, :], active[:, None], other=0)
            scales = tl.load(source_scales + source, active, other=1.0)
            dest = token * PADDED_HEADS + heads
            tl.store(padded_codes + dest[:, None] * DIM + channels[None, :], codes)
            tl.store(padded_scales + dest, scales)
            tl.store(weights + dest, tl.where(active, HEAD_WEIGHT, 0.0))


def prepare_qli_query(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize the dense 64-head INT8 input and FP16 QLI v2 scales and weights."""
    tokens, heads, dim = query.shape
    codes, scales = quantize_qsa_query(query)
    shape = (tokens, QLI_V2_QUERY_HEADS)
    weight = 1.0 / math.sqrt(dim)
    # A2/A3 QLI v2 requires 64 heads. Inactive heads have zero weight,
    # so the padding cannot change the score. Keep the exact same FP32
    # quantizer and convert scales only when writing the FP16 input.
    if query.device.type == "npu" and tokens >= _QLI_V2_FUSED_MIN_TOKENS:
        padded_codes = torch.empty((*shape, dim), dtype=torch.int8, device=query.device)
        padded_scales = torch.empty(shape, dtype=torch.float16, device=query.device)
        weights = torch.empty_like(padded_scales)
        cores = min(tokens, get_vectorcore_num())
        _pad_qli_query[(cores,)](
            codes,
            scales,
            padded_codes,
            padded_scales,
            weights,
            tokens,
            triton.cdiv(tokens, cores),
            HEADS=heads,
            DIM=dim,
            PADDED_HEADS=QLI_V2_QUERY_HEADS,
            HEAD_WEIGHT=weight,
        )
        return padded_codes, padded_scales, weights
    padded_codes = torch.zeros((*shape, dim), dtype=torch.int8, device=query.device)
    padded_codes[:, :heads].copy_(codes)
    padded_scales = torch.ones(shape, dtype=torch.float16, device=query.device)
    padded_scales[:, :heads].copy_(scales)
    weights = torch.zeros_like(padded_scales)
    weights[:, :heads] = weight
    return padded_codes, padded_scales, weights


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
    key_scales: torch.Tensor | None = None,
) -> None:
    """Quantize normalized/rotated rows directly into a scheduler-owned page."""
    if cache.ndim != 4:
        raise ValueError("QSA C8 storage requires a 4-D cache")
    if indexer:
        if key_scales is not None or v_scale is not None:
            raise ValueError("QSA C8 indexer stores its scales in the key page")
        codes, scales = qsa_c8_indexer_cache_views(cache)
    elif v_scale is None:
        if (
            key_scales is None
            or key_scales.shape != (cache.shape[0], cache.shape[2], cache.shape[1])
            or key_scales.dtype != torch.float32
            or key_scales.device != cache.device
            or not key_scales.is_contiguous()
        ):
            raise ValueError("QSA C8 K scales must match the physical cache pages and heads")
        codes, scales = cache, key_scales.transpose(1, 2)
    else:
        if key_scales is not None:
            raise ValueError("QSA C8 V store takes only calibrated V scales")
        codes, scales = cache, None
    if codes.ndim != 4 or codes.dtype != torch.int8 or rows.ndim != 3 or rows.shape[1:] != codes.shape[2:]:
        raise ValueError("QSA C8 rows must match cache heads and logical head dimension")
    if cache.device.type != "npu" or rows.device != cache.device or slots.device != cache.device:
        raise ValueError("QSA C8 store requires rows, slots and cache on the same NPU")
    if rows.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("QSA C8 store requires BF16/FP16/FP32 input")
    if slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("QSA C8 store requires integer slots")
    if v_scale is not None and (
        v_scale.shape != rows.shape[1:]
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
        scales if scales is not None else codes,
        scales if v_scale is None else v_scale,
        *rows.stride(),
        *codes.stride()[:3],
        *(scales.stride() if scales is not None else codes.stride()[:3]),
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
