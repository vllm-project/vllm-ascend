# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Logical mHC seam contracts shared by native, fused, and decoupled backends.

The seam applies HcPost, rounds the updated residual to the activation dtype,
collapses it with the incoming delayed pre_mix, and rounds the collapsed tensor
before RMSNorm. Routing coefficients for the next seam are computed from the
rounded updated residual and are independent of the collapse and RMSNorm branch.

When requested, v_fp32 is BF16(v).float() with the low 16 FP32 bits zero.
When omitted, the optional output is None.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import torch


class WeightFormat(IntEnum):
    FP32_CANONICAL = 0
    OZAKI_BF16_NZ_V1 = 1
    OZAKI_BF16_ND_V1 = 2


class NumericsProfile(IntEnum):
    STANDARD_NR1 = 0
    HIGH_PRECISION_NR2 = 1


@dataclass(frozen=True, slots=True)
class MhcHardwareCapability:
    """Capabilities resolved by the runner for the target NPU device.

    has_a2_overlap_op attests to the 910B overlap operators and runner
    prerequisites. has_a5_native_op attests to a native arch35 adapter for this
    semantic ABI. graph_mode is the runner mode name in lowercase, or
    eager/aclgraph.
    """

    soc_name: str
    arch_gen: str
    aic_count: int
    dtype_support: frozenset[torch.dtype]
    has_a3_fused_op: bool = False
    has_a3_decoupled_op: bool = False
    has_a2_overlap_op: bool = False
    has_a5_native_op: bool = False
    graph_mode: str = "eager"
    superkernel_enabled: bool = False


@dataclass(frozen=True, slots=True)
class MhcSeamSpec:
    """Static seam metadata evaluated before graph capture. T is local rows."""

    C: int
    D: int
    dtype: torch.dtype
    return_v_fp32: bool = False
    weight_format: WeightFormat = WeightFormat.OZAKI_BF16_NZ_V1
    numerics_profile: NumericsProfile = NumericsProfile.STANDARD_NR1
    is_layer0_entry: bool = False
    is_write_kv_path: bool = False
    sp_enabled: bool = False
    T: int | None = None
    write_kv_migrated: bool = False
    post_mult: float = 2.0
    prefer_side_stream: bool = False


@dataclass(frozen=True, slots=True)
class MhcSeamInputs:
    """Full seam inputs with contiguous tensors on one device.

    y and gamma use the activation dtype with shapes [T, D] and [D].
    residual has shape [T, C, D]. post and delayed pre_mix are FP32 [T, C],
    and comb is FP32 [T, C, C] in (source, destination) order.
    hc_scale is FP32 [3] and hc_base is FP32 [C * (C + 2)].
    """

    y: torch.Tensor
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    pre_mix: torch.Tensor
    gamma: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.y.device


@dataclass(frozen=True, slots=True)
class MhcSeamOutputs:
    """Rounded residual and normalized activations, followed by next coefficients."""

    residual: torch.Tensor
    v: torch.Tensor
    v_fp32: torch.Tensor | None
    post: torch.Tensor
    comb: torch.Tensor
    pre: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.residual.device


@dataclass(frozen=True, slots=True)
class PreparedMhcWeights:
    """One device weight representation tied to a loader version.

    source_released records release of the canonical FP32 device tensor by the
    lifecycle owner. index_map_ready records preparation of the A3 device
    constant table.
    """

    tensor: torch.Tensor
    device: torch.device
    version_tag: str
    weight_format: WeightFormat
    source_released: bool = False
    index_map_ready: bool = False
    native_adapter: Callable[[MhcSeamInputs, torch.Tensor, dict], MhcSeamOutputs] | None = None
