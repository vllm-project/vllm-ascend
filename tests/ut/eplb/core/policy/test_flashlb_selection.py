# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from vllm_ascend.eplb.core.policy.policy_flashlb import FlashLB


@pytest.mark.parametrize(
    "loads,limit,expected_layers",
    [
        ([[1, 1, 1, 1]], -1, []),
        ([[10, 9, 1, 1], [10, 8, 2, 1]], 1, [0]),
        ([[10, 9, 1, 1]], -1, [0]),
    ],
)
def test_only_selected_layers_redeploy(loads, limit, expected_layers):
    current = np.tile(np.array([[[0, 1], [2, 3]]], dtype=np.int32), (len(loads), 1, 1))
    logical = np.asarray(loads, dtype=np.float32)
    workload = np.stack([logical[layer][current[layer]] for layer in range(len(loads))])[None]
    policy = FlashLB()
    policy.true_update = True  # Keep the 32-rank thresholds without initializing a group.
    policy.update_layers_upper_bound = limit
    changed, priority, returned = policy.rebalance_experts(torch.from_numpy(current), torch.from_numpy(workload))
    assert priority.tolist() == expected_layers
    assert changed == bool(expected_layers)
    assert sorted(policy.average_to_peak_history) == expected_layers
    for layer in range(len(loads)):
        if layer not in expected_layers:
            np.testing.assert_array_equal(returned[layer], current[layer])
        else:
            assert not np.array_equal(returned[layer], current[layer])
