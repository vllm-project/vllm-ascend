# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.hy_v4_flash.reference import FP8_MAX, INDEX_DIM
from vllm.triton_utils import tl, triton

_MIN_AMAX = 1e-4
_E4M3_MIN_NORMAL = tl.constexpr(2.0**-6)
_E4M3_FRACTION_BITS = tl.constexpr(3)
_FP32_FRACTION_BITS = tl.constexpr(23)
_FP32_EXPONENT_BIAS = tl.constexpr(127)


@triton.jit
def _indexpool_hadamard(
    vectors,
    transformed,
    maxima,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, WIDTH)
    offset = row * WIDTH + lane
    values = tl.load(vectors + offset).to(tl.float32)
    for level in tl.static_range(LEVELS):
        stride = 1 << level
        partner = tl.gather(values, lane ^ stride, 0)
        values = tl.where((lane & stride) == 0, values + partner, partner - values)
    values = values * NORMALIZE

    magnitude = tl.abs(values)
    nonfinite = tl.max(tl.where(magnitude < float("inf"), 0, 1), 0) != 0
    amax = tl.where(nonfinite, float("nan"), tl.max(magnitude, 0))
    tl.store(transformed + offset, values)
    tl.store(maxima + row, amax)


@triton.jit
def _indexpool_round_e4m3(
    transformed,
    normalized,
    scales,
    WIDTH: tl.constexpr,
    QUANT_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    offset = row * WIDTH + tl.arange(0, WIDTH)
    values = tl.load(transformed + offset)
    scale = tl.load(scales + row)
    magnitude = tl.minimum(tl.abs(tl.load(normalized + offset)), QUANT_MAX)
    # The FP32 exponent gives the normal E4M3 spacing. Clamping at the
    # smallest normal magnitude also gives the fixed subnormal spacing.
    magnitude_bits = tl.maximum(magnitude, _E4M3_MIN_NORMAL).to(tl.int32, bitcast=True)
    exponent = (magnitude_bits >> _FP32_FRACTION_BITS) - _FP32_EXPONENT_BIAS - _E4M3_FRACTION_BITS
    unit = ((exponent + _FP32_EXPONENT_BIAS) << _FP32_FRACTION_BITS).to(tl.float32, bitcast=True)
    inverse_unit = ((_FP32_EXPONENT_BIAS - exponent) << _FP32_FRACTION_BITS).to(tl.float32, bitcast=True)
    rounded = tl.extra.cann.libdevice.nearbyint(magnitude * inverse_unit)
    sign = tl.where(values > 0, 1.0, tl.where(values < 0, -1.0, 0.0))
    result = tl.where(scale == scale, rounded * unit * sign * scale, float("nan"))
    tl.store(normalized + offset, result)


def encode_indexpool_fp8(vectors: torch.Tensor) -> torch.Tensor:
    """Return the FP32, E4M3-rounded IndexPool rows used by the scorer."""
    rows = vectors.numel() // INDEX_DIM
    transformed = torch.empty_like(vectors, dtype=torch.float32)
    maxima = vectors.new_empty((*vectors.shape[:-1], 1), dtype=torch.float32)
    _indexpool_hadamard[(rows,)](
        vectors,
        transformed,
        maxima,
        INDEX_DIM,
        INDEX_DIM.bit_length() - 1,
        INDEX_DIM**-0.5,
        num_warps=1,
        enable_fp_fusion=False,
    )
    # Keep CANN's scale and division rounding at exponent and E4M3 bin boundaries.
    scale = torch.exp2(torch.ceil(torch.log2(maxima.clamp_min(_MIN_AMAX) / FP8_MAX)))
    normalized = transformed / scale
    _indexpool_round_e4m3[(rows,)](
        transformed, normalized, scale, INDEX_DIM, FP8_MAX, num_warps=1, enable_fp_fusion=False
    )
    return normalized
