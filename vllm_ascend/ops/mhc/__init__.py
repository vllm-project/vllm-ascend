# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Reusable Manifold-Constrained Hyper-Connections (mHC) seam dispatch and lifecycle utilities."""

from .contracts import (
    MhcHardwareCapability,
    MhcSeamInputs,
    MhcSeamOutputs,
    MhcSeamSpec,
    NumericsProfile,
    PreparedMhcWeights,
    WeightFormat,
)
from .lifecycle import MhcWeightLifecycleManager
from .packing import (
    C4_COEFFICIENTS,
    OZAKI_FORMATS,
    build_c4_index_map,
    c4_weight_nbytes,
    ozaki_shape,
    pack_ozaki_weights,
    unpack_ozaki_weights,
)
from .registry import (
    A2OverlapMhcBackend,
    A3DecoupledMhcBackend,
    A3MegaMhcBackend,
    A5NativeMhcBackend,
    MhcBackend,
    NativeFallbackMhcBackend,
    select_backend,
    should_prefer_side_stream,
)

__all__ = [
    "A2OverlapMhcBackend",
    "A3DecoupledMhcBackend",
    "A3MegaMhcBackend",
    "A5NativeMhcBackend",
    "C4_COEFFICIENTS",
    "MhcBackend",
    "MhcHardwareCapability",
    "MhcSeamInputs",
    "MhcSeamOutputs",
    "MhcSeamSpec",
    "MhcWeightLifecycleManager",
    "NativeFallbackMhcBackend",
    "NumericsProfile",
    "OZAKI_FORMATS",
    "PreparedMhcWeights",
    "WeightFormat",
    "build_c4_index_map",
    "c4_weight_nbytes",
    "ozaki_shape",
    "pack_ozaki_weights",
    "select_backend",
    "should_prefer_side_stream",
    "unpack_ozaki_weights",
]
