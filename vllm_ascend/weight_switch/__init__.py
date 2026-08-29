# SPDX-License-Identifier: Apache-2.0
"""Generic full-weight switching primitives."""

from .linear import (
    WeightLoadPartition,
    WeightSwitchConfig,
    WeightSwitchGatherPart,
    WeightSwitchGatherSpec,
    WeightSwitchLoadState,
    WeightSwitchMixin,
    WeightSwitchRepeatPart,
    WeightSwitchRepeatSpec,
    WeightSwitchState,
)

__all__ = [
    "WeightLoadPartition",
    "WeightSwitchConfig",
    "WeightSwitchGatherPart",
    "WeightSwitchGatherSpec",
    "WeightSwitchLoadState",
    "WeightSwitchMixin",
    "WeightSwitchRepeatPart",
    "WeightSwitchRepeatSpec",
    "WeightSwitchState",
]
