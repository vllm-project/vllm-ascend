# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduced GLM model cases for weight-transfer tests."""

from tests.e2e.pull_request.two_card.rlhf.common.weight_transfer.models.base import WeightUpdateModelCase

GLM51_CASE = WeightUpdateModelCase(
    id="glm-5.1-sfa-derived-kv",
    model="zai-org/GLM-5.1",
    hf_overrides={
        "num_hidden_layers": 4,
        # Keep the trainer payload and TP workers within the test memory budget.
        "n_routed_experts": 8,
    },
)

GLM_CASES = (GLM51_CASE,)
