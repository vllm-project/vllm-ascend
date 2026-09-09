# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Physical draft K and optional batch gating.

Confidence estimation, cost-based logical budgets, and prefix allocation belong
to the upstream adaptive-verification manager.
"""

from vllm_ascend.spec_decode.dynamic.draft_k_controller import AdaptiveDraftKController
from vllm_ascend.spec_decode.dynamic.proposal_gate import ProposalGate

__all__ = ["AdaptiveDraftKController", "ProposalGate"]
