# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.hy_v4_flash.reference import FP8_MAX, INDEX_DIM
from vllm.triton_utils import tl, triton

_MIN_AMAX = 1e-4
_MIN_NORMAL = 2.0**-6
_FP32_FRACTION_BITS = 23
_FP32_EXPONENT_BIAS = 127
_E4M3_FRACTION_BITS = 3


def encode_indexpool_fp8(vectors: torch.Tensor) -> torch.Tensor:
    """Return FP32 Hadamard rows rounded to the original E4M3 bins."""
    if not vectors.numel():
        return vectors.float()
    width = vectors.shape[-1]
    values = vectors.float()
    stride = 1
    while stride < width:
        left, right = values.reshape(-1, 2, stride).unbind(1)
        values = torch.stack((left + right, left - right), dim=1).reshape(vectors.shape)
        stride *= 2
    values = values * width**-0.5
    magnitude = values.abs()
    maxima = magnitude.amax(dim=-1, keepdim=True)
    maxima = maxima.masked_fill(~(magnitude < torch.inf).all(dim=-1, keepdim=True), torch.nan)
    scale = torch.exp2(torch.ceil(torch.log2(maxima.clamp_min(_MIN_AMAX) / FP8_MAX)))
    magnitude = (values / scale).abs().clamp_max(FP8_MAX)
    bits = magnitude.clamp_min(_MIN_NORMAL).contiguous().view(torch.int32)
    exponent = (bits >> _FP32_FRACTION_BITS) - _FP32_EXPONENT_BIAS - _E4M3_FRACTION_BITS
    unit = ((exponent + _FP32_EXPONENT_BIAS) << _FP32_FRACTION_BITS).view(torch.float32)
    inverse_unit = ((_FP32_EXPONENT_BIAS - exponent) << _FP32_FRACTION_BITS).view(torch.float32)
    sign = (values > 0).float() - (values < 0).float()
    result = (magnitude * inverse_unit).round() * unit * sign * scale
    return result.masked_fill(scale.isnan(), torch.nan)


@triton.jit(do_not_specialize=["rows"])
def _round_prefill_fp8_rows(
    transformed,
    output,
    rows,
    TILE_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    NORMALIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MIN_AMAX: tl.constexpr,
):
    row = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    lane = tl.arange(0, WIDTH)
    offset = row[:, None] * WIDTH + lane[None, :]
    data = tl.load(transformed + offset, row[:, None] < rows, other=0) * NORMALIZE
    magnitude = tl.abs(data)
    valid = tl.sum((~(magnitude < float("inf"))).to(tl.int32), axis=1) == 0
    amax = tl.maximum(tl.max(magnitude, axis=1), MIN_AMAX)
    exponent = tl.ceil(tl.log2(tl.div_rn(amax, FP8_MAX))).to(tl.int32)
    scale = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
    inverse_scale = ((127 - exponent) << 23).to(tl.float32, bitcast=True)

    scaled = tl.minimum(magnitude * inverse_scale[:, None], FP8_MAX)
    # E4M3 uses three fraction bits and a fixed step below the smallest normal.
    normal = tl.maximum(scaled, 2.0**-6).to(tl.int32, bitcast=True)
    unit_exponent = (normal >> 23) - 127 - 3
    unit = ((unit_exponent + 127) << 23).to(tl.float32, bitcast=True)
    inverse_unit = ((127 - unit_exponent) << 23).to(tl.float32, bitcast=True)
    rounded = tl.extra.cann.libdevice.nearbyint(scaled * inverse_unit)
    restored = rounded * unit * scale[:, None]
    sign = data.to(tl.int32, bitcast=True) & -2147483648
    signed = (restored.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)
    result = tl.where(valid[:, None], signed, float("nan"))
    tl.store(output + offset, result, row[:, None] < rows)


def round_prefill_fp8_rows(transformed: torch.Tensor, *, output_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Round the FP32 Hadamard output to the IndexPool E4M3 grid."""
    rows = transformed.shape[0]
    result = torch.empty_like(transformed, dtype=output_dtype)
    tile_rows = 32
    _round_prefill_fp8_rows[(triton.cdiv(rows, tile_rows),)](
        transformed,
        result,
        rows,
        tile_rows,
        INDEX_DIM,
        INDEX_DIM**-0.5,
        FP8_MAX,
        _MIN_AMAX,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return result
