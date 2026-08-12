# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

import vllm_ascend.distributed.eplb_policy as eplb_policy
from vllm_ascend.distributed.eplb_policy import (
    StairEplbPolicyAdapter,
    _expand_logical_load_to_slots,
    _reject_invalid_placement_layers,
)
from vllm_ascend.eplb.core.policy.policy_stair import compute_balance_score


def test_expand_logical_load_to_slots_preserves_logical_load():
    logical_load = torch.tensor([[10, 20, 30, 40]], dtype=torch.int32)
    placement = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long)

    slot_load = _expand_logical_load_to_slots(logical_load, placement)
    reconstructed = torch.zeros_like(logical_load, dtype=slot_load.dtype)
    reconstructed.scatter_add_(1, placement, slot_load)

    torch.testing.assert_close(reconstructed, logical_load.to(torch.float64))


def test_reject_invalid_placement_layers_matches_transfer_constraints():
    old_placement = torch.tensor(
        [[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 0, 1]],
        dtype=torch.long,
    )
    proposed_placement = torch.tensor(
        [[0, 1, 3, 2, 0, 1], [0, 2, 1, 3, 0, 0]],
        dtype=torch.long,
    )

    rejected = _reject_invalid_placement_layers(
        old_placement,
        proposed_placement,
        num_ranks=2,
        num_logical_experts=4,
    )

    assert rejected == [1]
    torch.testing.assert_close(proposed_placement[0], torch.tensor([0, 1, 3, 2, 0, 1]))
    torch.testing.assert_close(proposed_placement[1], old_placement[1])


def test_stair_adapter_is_instance_owned_and_improves_balance():
    load_window = torch.tensor(
        [[[1, 1, 100, 1]], [[1, 1, 120, 1]], [[1, 1, 80, 1]], [[1, 1, 110, 1]]],
        dtype=torch.int32,
    )
    placement = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long)
    first = StairEplbPolicyAdapter()
    second = StairEplbPolicyAdapter()

    result = first.rebalance_experts(
        load_window,
        num_replicas=6,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=placement,
    )

    assert first.policy is not second.policy
    assert result.shape == placement.shape
    old_score = compute_balance_score(load_window[:, 0].numpy(), placement.reshape(2, 3).numpy())
    new_score = compute_balance_score(load_window[:, 0].numpy(), result.reshape(2, 3).numpy())
    assert new_score < old_score


def test_stair_adapter_filters_swift_candidates_with_full_time_series(monkeypatch):
    load_window = torch.tensor(
        [[[100, 90, 1, 1], [100, 1, 90, 1]], [[90, 100, 1, 1], [90, 1, 100, 1]]],
        dtype=torch.int32,
    )
    placement = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    candidate = torch.tensor([[0, 2, 1, 3], [0, 2, 1, 3]], dtype=torch.long)
    observed_load = None

    def fake_calculate(logical_load, old_placement, num_nodes, num_ranks):
        nonlocal observed_load
        observed_load = logical_load.clone()
        return candidate.clone()

    monkeypatch.setattr(eplb_policy, "_calculate_swift_placement", fake_calculate)
    result = StairEplbPolicyAdapter().rebalance_experts(
        load_window,
        num_replicas=4,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=placement,
    )

    torch.testing.assert_close(observed_load, load_window.sum(dim=0))
    torch.testing.assert_close(result[0], candidate[0])
    torch.testing.assert_close(result[1], placement[1])


def test_stair_adapter_commits_history_only_after_real_layer_commit(monkeypatch):
    adapter = StairEplbPolicyAdapter()
    load_window = torch.tensor([[[100, 100, 1, 1]], [[1, 1, 100, 100]]], dtype=torch.int32)
    placement = torch.tensor([0, 2, 1, 3], dtype=torch.long)

    adapter.commit_layer(load_window, 0, placement, num_ranks=2)

    assert set(adapter.policy.average_to_peak_history) == {0}


def test_stair_adapter_rejects_missing_or_invalid_placement():
    load_window = torch.ones((2, 1, 4), dtype=torch.int32)
    adapter = StairEplbPolicyAdapter()
    for placement, error in [
        (None, "requires the current"),
        (torch.tensor([[0, 1, 2, 4]]), "invalid logical expert"),
        (torch.tensor([[0, 0, 1, 2]]), "at least one physical replica"),
    ]:
        try:
            adapter.rebalance_experts(
                load_window,
                num_replicas=4,
                num_groups=1,
                num_nodes=1,
                num_ranks=2,
                old_global_expert_indices=placement,
            )
        except ValueError as exc:
            assert error in str(exc)
        else:
            raise AssertionError("Expected invalid STAIR placement to fail.")
