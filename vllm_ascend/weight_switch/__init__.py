# SPDX-License-Identifier: Apache-2.0
"""Generic full-weight switching primitives and backend controllers."""

from .controller import (
    WeightSwitchControllerMixin,
    WeightSwitchHandle,
    WeightSwitchTarget,
)
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

# Temporary TP-only aliases keep existing backends source-compatible while
# all new integrations use the parallel-domain-agnostic names above.
TPWeightGatherPart = WeightSwitchGatherPart
TPWeightGatherSpec = WeightSwitchGatherSpec
TPWeightRepeatPart = WeightSwitchRepeatPart
TPWeightRepeatSpec = WeightSwitchRepeatSpec
TPWeightSwitchMixin = WeightSwitchMixin
TPWeightSwitchState = WeightSwitchState

__all__ = [
    "TPWeightGatherPart",
    "TPWeightGatherSpec",
    "TPWeightRepeatPart",
    "TPWeightRepeatSpec",
    "TPWeightSwitchMixin",
    "TPWeightSwitchState",
    "WeightLoadPartition",
    "WeightSwitchConfig",
    "WeightSwitchControllerMixin",
    "WeightSwitchGatherPart",
    "WeightSwitchGatherSpec",
    "WeightSwitchHandle",
    "WeightSwitchLoadState",
    "WeightSwitchMixin",
    "WeightSwitchRepeatPart",
    "WeightSwitchRepeatSpec",
    "WeightSwitchState",
    "WeightSwitchTarget",
]
