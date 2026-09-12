# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.eplb.policy import AscendV2EplbPolicy


def make_policy(*, changed, new_deployment):
    policy = AscendV2EplbPolicy.__new__(AscendV2EplbPolicy)
    policy.policy_name = "test"
    policy._policy = MagicMock()
    policy._policy.rebalance_experts.return_value = (
        changed,
        None,
        new_deployment,
    )
    return policy


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
    old_mapping = torch.tensor([[0, 1, 0, 1]])
    new_deployment = [[[0, 0], [1, 1]]]
    policy = make_policy(changed=True, new_deployment=new_deployment)

    result = policy.rebalance_experts(
        weight=torch.tensor([[100, 20]]),
        num_replicas=4,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=old_mapping,
    )

    current_table, workload_table = policy._policy.rebalance_experts.call_args.args
    torch.testing.assert_close(
        current_table,
        torch.tensor([[[0, 1], [0, 1]]]),
    )
    torch.testing.assert_close(
        workload_table,
        torch.tensor([[[50.0, 10.0], [50.0, 10.0]]]),
    )
    torch.testing.assert_close(result, torch.tensor([[0, 0, 1, 1]]))


def test_rebalance_keeps_old_mapping_when_policy_declines_update():
    old_mapping = torch.tensor([[0, 1, 0, 1]])
    policy = make_policy(
        changed=False,
        new_deployment=[[[0, 0], [1, 1]]],
    )

    result = policy.rebalance_experts(
        weight=torch.tensor([[100, 20]]),
        num_replicas=4,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=old_mapping,
    )

    torch.testing.assert_close(result, old_mapping)
    assert result is not old_mapping


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
