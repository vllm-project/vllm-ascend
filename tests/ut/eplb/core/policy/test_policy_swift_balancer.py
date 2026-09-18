# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np
import pytest

from vllm_ascend.eplb.core.policy.policy_swift_balancer import SwiftBalanceEplb


@pytest.mark.parametrize(
    ("num_redundant_experts", "expected_replica_weight"),
    [
        (1, 60.0),
        (2, 40.0),
    ],
)
def test_compute_redundant_assignments_preserves_total_expert_load(
    num_redundant_experts: int,
    expected_replica_weight: float,
) -> None:
    policy = SwiftBalanceEplb.__new__(SwiftBalanceEplb)
    policy.num_original_experts = 1

    redundant_experts, updated_weights = policy.compute_redundant_assignments(
        initial_weights=[(0, 120.0)],
        num_redundant_experts=num_redundant_experts,
        num_ranks=4,
    )

    assert redundant_experts == [(0, expected_replica_weight)] * num_redundant_experts
    np.testing.assert_allclose(updated_weights, [expected_replica_weight])
    assert (num_redundant_experts + 1) * updated_weights[0] == pytest.approx(120.0)
