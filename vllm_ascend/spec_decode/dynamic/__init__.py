# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Physical draft K and optional batch gating.

Confidence estimation, cost-based logical budgets, and prefix allocation belong
to the upstream adaptive-verification manager.
"""

from vllm_ascend.spec_decode.dynamic.policy import AdaptiveDraftKController, ProposalGate

__all__ = ["AdaptiveDraftKController", "ProposalGate"]
