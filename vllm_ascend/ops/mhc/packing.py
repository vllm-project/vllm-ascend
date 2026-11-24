# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Host-only second-order Ozaki dual-BF16 weight packing and C4 gather layout."""

from __future__ import annotations

import torch

from .contracts import WeightFormat

C4_COEFFICIENTS = 24
NZ_TILE_WIDTH = 16
OZAKI_FORMATS = frozenset((WeightFormat.OZAKI_BF16_NZ_V1, WeightFormat.OZAKI_BF16_ND_V1))


def ozaki_shape(k: int, weight_format: WeightFormat) -> tuple[int, ...]:
    if k <= 0:
        raise ValueError(f"Ozaki K must be positive, got {k}")
    if weight_format == WeightFormat.OZAKI_BF16_ND_V1:
        return (2 * C4_COEFFICIENTS, k)
    if weight_format == WeightFormat.OZAKI_BF16_NZ_V1 and k % NZ_TILE_WIDTH == 0:
        return (k // NZ_TILE_WIDTH, 2 * C4_COEFFICIENTS, NZ_TILE_WIDTH)
    raise ValueError(f"Invalid Ozaki format or K for NZ packing: {weight_format}, K={k}")


def c4_weight_nbytes(k: int, weight_format: WeightFormat) -> int:
    """Return storage bytes for one canonical FP32 or Ozaki C4 weight tensor."""
    if weight_format == WeightFormat.FP32_CANONICAL:
        if k <= 0:
            raise ValueError(f"Canonical K must be positive, got {k}")
    else:
        ozaki_shape(k, weight_format)
    return C4_COEFFICIENTS * k * 4


@torch.no_grad()
def pack_ozaki_weights(
    weight_fp32: torch.Tensor,
    target_format: WeightFormat = WeightFormat.OZAKI_BF16_NZ_V1,
) -> torch.Tensor:
    """Pack hi=BF16(w) and lo=BF16(w - FP32(hi)) on CPU.

    ND stores all 24 high rows followed by all 24 low rows.
    FRACTAL_NZ stores NZ[k // 16, row, k % 16] = ND[row, k].
    """
    if weight_fp32.device.type != "cpu" or weight_fp32.dtype != torch.float32:
        raise ValueError("Ozaki packing requires a CPU float32 tensor")
    if weight_fp32.ndim != 2 or weight_fp32.shape[0] != C4_COEFFICIENTS:
        raise ValueError(f"Ozaki packing expects [24, K], got {tuple(weight_fp32.shape)}")
    k = weight_fp32.shape[1]
    packed = torch.empty(ozaki_shape(k, target_format), dtype=torch.bfloat16, device="cpu")
    if target_format == WeightFormat.OZAKI_BF16_NZ_V1:
        source = weight_fp32.unflatten(1, (k // NZ_TILE_WIDTH, NZ_TILE_WIDTH)).transpose(0, 1)
        hi, lo = packed.split(C4_COEFFICIENTS, dim=1)
    else:
        source = weight_fp32
        hi, lo = packed.split(C4_COEFFICIENTS, dim=0)
    hi.copy_(source)
    lo.copy_(source - hi)
    return packed


def unpack_ozaki_weights(
    packed: torch.Tensor,
    weight_format: WeightFormat = WeightFormat.OZAKI_BF16_NZ_V1,
) -> torch.Tensor:
    """Return the CPU FP32 reconstruction hi.float() + lo.float()."""
    if packed.device.type != "cpu" or packed.dtype != torch.bfloat16:
        raise ValueError("Ozaki unpacking requires a CPU bfloat16 tensor")
    if weight_format == WeightFormat.OZAKI_BF16_NZ_V1 and packed.ndim == 3:
        k = packed.shape[0] * NZ_TILE_WIDTH
    elif weight_format == WeightFormat.OZAKI_BF16_ND_V1 and packed.ndim == 2:
        k = packed.shape[1]
    else:
        raise ValueError(f"Invalid Ozaki shape {tuple(packed.shape)} for {weight_format}")
    if packed.shape != ozaki_shape(k, weight_format) or not packed.is_contiguous():
        raise ValueError(f"Invalid contiguous Ozaki layout {tuple(packed.shape)} for {weight_format}")
    if weight_format == WeightFormat.OZAKI_BF16_NZ_V1:
        packed = packed.transpose(0, 1).reshape(2 * C4_COEFFICIENTS, k)
    hi, lo = packed.split(C4_COEFFICIENTS, dim=0)
    return hi.float() + lo.float()


def build_c4_index_map() -> torch.Tensor:
    """Build byte-offset gather table for C4 reduction, logits, scales, and post."""
    reduction_maps = [(i ^ mask) * 4 for mask in (1, 2, 4, 8) for i in range(64)]
    sigmoid = [(row * 64 + j) * 4 for row in range(4) for j in range(8)]
    comb = [(row * 64 + 8 + j) * 4 for row in range(4) for j in range(16)]
    sigmoid_scale = [0] * 4 + [4] * 4
    comb_scale = [8] * 16
    post = [(row * 8 + 4 + j % 4) * 4 for row in range(4) for j in range(8)]
    sections = (reduction_maps, sigmoid, comb, sigmoid_scale, comb_scale, post)
    for offsets, source_words in zip(sections, (64, 256, 256, 8, 8, 32), strict=True):
        assert all(0 <= offset < source_words * 4 and offset % 4 == 0 for offset in offsets)
    values = [offset for section in sections for offset in section]
    assert len(values) == 408
    return torch.tensor(values, dtype=torch.int32, device="cpu")
