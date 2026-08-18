# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hardware-aware dynamic speculative decoding helpers.

The public entry points in this package are deliberately small.  The proposer
still owns model-specific confidence estimation; this package owns calibration
and the hardware-dependent allocation of verification prefixes.
"""

from vllm_ascend.spec_decode.dynamic.calibration import SequentialTemperatureScaler
from vllm_ascend.spec_decode.dynamic.cost_model import HardwareCostModel
from vllm_ascend.spec_decode.dynamic.draft_k_controller import AdaptiveDraftKController
from vllm_ascend.spec_decode.dynamic.policy import HardwareAwarePrefixPolicy
from vllm_ascend.spec_decode.dynamic.proposal_gate import ProposalGate

__all__ = [
    "HardwareAwarePrefixPolicy",
    "HardwareCostModel",
    "SequentialTemperatureScaler",
    "ProposalGate",
    "AdaptiveDraftKController",
]
