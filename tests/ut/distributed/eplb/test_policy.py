# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.eplb.policy import AscendV2EplbPolicy
from vllm_ascend.eplb.core.policy.policy_swift_balancer import (
    SwiftBalanceEplb,
)


def make_policy(*, changed, new_deployment, per_layer_priority=None, max_layers=2):
    policy = AscendV2EplbPolicy.__new__(AscendV2EplbPolicy)
    policy.max_rebalanced_layers_per_cycle = max_layers
    policy.ep_rank = 1
    policy._bootstrap_rebalance_active = True
    policy._policy = MagicMock()
    policy._policy.imbalance_threshold = 1.01
    policy._policy.increment = 0.01
    policy._policy.rebalance_experts.return_value = (
        changed,
        per_layer_priority,
        new_deployment,
    )
    return policy


def test_uses_swift_balancer_policy(monkeypatch):
    generated_policy = object()
    generate_policy = MagicMock(return_value=generated_policy)
    monkeypatch.setattr(
        "vllm_ascend.eplb.core.policy.policy_factory.PolicyFactory.generate_policy",
        generate_policy,
    )

    policy = AscendV2EplbPolicy()

    assert policy._policy is generated_policy
    assert policy.max_rebalanced_layers_per_cycle == 10
    generate_policy.assert_called_once_with(2)


def test_rejects_non_positive_rebalanced_layer_limit(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.eplb.core.policy.policy_factory.PolicyFactory.generate_policy",
        MagicMock(),
    )

    with pytest.raises(ValueError, match="must be greater than 0"):
        AscendV2EplbPolicy(max_rebalanced_layers_per_cycle=0)


def test_build_physical_load_splits_replicated_expert_load():
    logical_load = torch.tensor([[100, 20]], dtype=torch.int32)
    mapping = torch.tensor([[0, 1, 0, 1]])

    physical_load = AscendV2EplbPolicy._build_physical_load(
        logical_load,
        mapping,
    )

    torch.testing.assert_close(
        physical_load,
        torch.tensor([[50.0, 10.0, 50.0, 10.0]]),
    )


def test_rebalance_converts_between_vllm_and_ascend_shapes():
    old_mapping = torch.tensor([[0, 1, 2, 3, 0, 1]])
    new_deployment = [[[0, 1, 2], [3, 0, 2]]]
    policy = make_policy(changed=True, new_deployment=new_deployment)

    result = policy.rebalance_experts(
        weight=torch.tensor([[100, 20, 10, 5]]),
        num_replicas=6,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=old_mapping,
    )

    current_table, workload_table = policy._policy.rebalance_experts.call_args.args
    torch.testing.assert_close(
        current_table,
        torch.tensor([[[0, 1, 2], [3, 0, 1]]]),
    )
    torch.testing.assert_close(
        workload_table,
        torch.tensor([[[50.0, 10.0, 10.0], [5.0, 50.0, 10.0]]]),
    )
    torch.testing.assert_close(result, torch.tensor([[0, 1, 2, 3, 0, 2]]))


def test_rebalance_uses_candidate_mapping_when_policy_declines_update():
    old_mapping = torch.tensor([[0, 1, 2, 3, 0, 1]])
    policy = make_policy(
        changed=False,
        new_deployment=[[[0, 1, 2], [3, 0, 2]]],
    )

    result = policy.rebalance_experts(
        weight=torch.tensor([[100, 20, 10, 5]]),
        num_replicas=6,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=old_mapping,
    )

    torch.testing.assert_close(result, torch.tensor([[0, 1, 2, 3, 0, 2]]))


def test_rebalance_limits_changed_layers_by_policy_priority():
    old_mapping = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1],
            [0, 1, 2, 3, 0, 1],
            [0, 1, 2, 3, 0, 1],
            [0, 1, 2, 3, 0, 1],
        ]
    )
    new_deployment = [
        [[0, 1, 2], [3, 0, 2]],
        [[0, 1, 3], [2, 0, 1]],
        [[0, 1, 2], [3, 0, 2]],
        [[0, 1, 3], [2, 0, 1]],
    ]
    policy = make_policy(
        changed=True,
        new_deployment=new_deployment,
        per_layer_priority=[3, 1, 0, 2],
        max_layers=2,
    )

    result = policy.rebalance_experts(
        weight=torch.tensor(
            [
                [100, 20, 10, 5],
                [100, 20, 10, 5],
                [100, 20, 10, 5],
                [100, 20, 10, 5],
            ]
        ),
        num_replicas=6,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=old_mapping,
    )

    torch.testing.assert_close(result[0], old_mapping[0])
    torch.testing.assert_close(result[1], torch.tensor([0, 1, 3, 2, 0, 1]))
    torch.testing.assert_close(result[2], old_mapping[2])
    torch.testing.assert_close(result[3], torch.tensor([0, 1, 3, 2, 0, 1]))


def test_rebalance_priority_skips_unchanged_layers():
    old_mapping = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1],
            [0, 1, 2, 3, 0, 1],
            [0, 1, 2, 3, 0, 1],
        ]
    )
    new_deployment = [
        [[0, 1, 2], [3, 0, 1]],
        [[0, 1, 3], [2, 0, 1]],
        [[0, 1, 2], [3, 0, 2]],
    ]
    policy = make_policy(
        changed=True,
        new_deployment=new_deployment,
        per_layer_priority=[0, 2, 1],
        max_layers=1,
    )

    result = policy.rebalance_experts(
        weight=torch.tensor([[100, 20, 10, 5]] * 3),
        num_replicas=6,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=old_mapping,
    )

    torch.testing.assert_close(result[0], old_mapping[0])
    torch.testing.assert_close(result[1], old_mapping[1])
    torch.testing.assert_close(result[2], torch.tensor([0, 1, 2, 3, 0, 2]))


def test_rejects_replicas_on_the_same_rank():
    with pytest.raises(ValueError, match="same logical expert"):
        AscendV2EplbPolicy._validate_replica_rank_placement(
            torch.tensor([[0, 0, 2, 3, 1, 2]]),
            num_ranks=2,
        )


def test_accepts_replicas_on_distinct_ranks():
    AscendV2EplbPolicy._validate_replica_rank_placement(
        torch.tensor([[0, 1, 2, 3, 0, 2]]),
        num_ranks=2,
    )


def test_bootstrap_disables_thresholds_and_restores_them():
    old_mapping = torch.tensor([[0, 1, 2, 3, 0, 1]])
    policy = make_policy(
        changed=False,
        new_deployment=[[[0, 1, 2], [3, 0, 1]]],
    )
    observed_thresholds = []

    def rebalance(current_table, workload_table):
        del workload_table
        observed_thresholds.append(
            (
                policy._policy.imbalance_threshold,
                policy._policy.increment,
            )
        )
        return False, [0], current_table.tolist()

    policy._policy.rebalance_experts.side_effect = rebalance
    policy.rebalance_experts(
        weight=torch.tensor([[100, 20, 10, 5]]),
        num_replicas=6,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=old_mapping,
    )

    assert observed_thresholds == [(1.0, 0.0)]
    assert policy._policy.imbalance_threshold == 1.01
    assert policy._policy.increment == 0.01
    assert not policy._bootstrap_rebalance_active


def test_swift_balancer_replicates_hot_experts_on_distinct_ranks():
    policy = AscendV2EplbPolicy.__new__(AscendV2EplbPolicy)
    policy.max_rebalanced_layers_per_cycle = 10
    policy.ep_rank = 1
    policy._bootstrap_rebalance_active = True
    policy._policy = SwiftBalanceEplb()
    old_mapping = torch.tensor([[0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 0]])

    result = policy.rebalance_experts(
        weight=torch.tensor([[1000, 800, 400, 300, 200, 150, 100, 50]]),
        num_replicas=12,
        num_groups=1,
        num_nodes=1,
        num_ranks=4,
        old_global_expert_indices=old_mapping,
    )

    old_counts = torch.bincount(old_mapping[0], minlength=8)
    new_counts = torch.bincount(result[0], minlength=8)
    assert new_counts[0] > old_counts[0]
    assert new_counts[1] > old_counts[1]
    assert new_counts[0] == new_counts.max()
    AscendV2EplbPolicy._validate_replica_rank_placement(
        result,
        num_ranks=4,
    )


def test_rebalance_rejects_elastic_slot_count_change():
    policy = make_policy(changed=False, new_deployment=[])

    with pytest.raises(ValueError, match="changing the number"):
        policy.rebalance_experts(
            weight=torch.tensor([[100, 20]]),
            num_replicas=6,
            num_groups=1,
            num_nodes=1,
            num_ranks=2,
            old_global_expert_indices=torch.tensor([[0, 1, 0, 1]]),
        )
