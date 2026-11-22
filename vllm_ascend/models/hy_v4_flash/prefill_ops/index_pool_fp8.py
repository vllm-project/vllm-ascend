# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.hy_v4_flash.reference import FP8_MAX

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
