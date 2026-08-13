# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np
import pytest
import torch

from vllm_ascend.distributed.stair_policy import (
    StairEplbPolicy,
    _build_incremental_candidate,
    compute_balance_score,
)


def _rebalance(
    policy: StairEplbPolicy,
    load_window: torch.Tensor,
    placement: torch.Tensor,
    num_ranks: int = 2,
) -> torch.Tensor:
    return policy.rebalance_experts(
        load_window,
        num_replicas=placement.shape[1],
        num_groups=1,
        num_nodes=1,
        num_ranks=num_ranks,
        old_global_expert_indices=placement,
    )


def test_stair_builds_incremental_candidate_without_legacy_policy():
    logical_load = np.array([[1, 1, 100, 1]], dtype=np.float64)
    current = np.array([[0, 1, 2, 3, 0, 1]], dtype=np.int64)

    candidate = _build_incremental_candidate(logical_load, current, num_ranks=2)

    assert candidate.shape == current.shape
    assert not np.array_equal(candidate, current)
    assert compute_balance_score(logical_load, candidate.reshape(2, 3)) < compute_balance_score(
        logical_load,
        current.reshape(2, 3),
    )
    for rank, old_rank in zip(candidate.reshape(2, 3), current.reshape(2, 3)):
        assert np.unique(rank).size == rank.size
        for slot_idx, expert_id in enumerate(rank):
            if expert_id in old_rank:
                assert expert_id == old_rank[slot_idx]


def test_stair_candidate_never_worsens_aggregate_balance():
    random = np.random.default_rng(7)
    num_ranks = 4
    slots_per_rank = 3
    current = np.tile(np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]), (64, 1))
    logical_load = random.integers(0, 1000, size=(64, 8)).astype(np.float64)

    candidate = _build_incremental_candidate(logical_load, current, num_ranks)

    for layer_idx in range(logical_load.shape[0]):
        current_score = compute_balance_score(
            logical_load[layer_idx : layer_idx + 1],
            current[layer_idx].reshape(num_ranks, slots_per_rank),
        )
        candidate_score = compute_balance_score(
            logical_load[layer_idx : layer_idx + 1],
            candidate[layer_idx].reshape(num_ranks, slots_per_rank),
        )
        assert candidate_score <= current_score


def test_stair_is_instance_owned_and_improves_balance():
    load_window = torch.tensor(
        [[[1, 1, 100, 1]], [[1, 1, 120, 1]], [[1, 1, 80, 1]], [[1, 1, 110, 1]]],
        dtype=torch.int32,
    )
    placement = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long)
    first = StairEplbPolicy()
    second = StairEplbPolicy()

    result = _rebalance(first, load_window, placement)

    assert first.average_to_peak_history is not second.average_to_peak_history
    old_score = compute_balance_score(load_window[:, 0].numpy(), placement.reshape(2, 3).numpy())
    new_score = compute_balance_score(load_window[:, 0].numpy(), result.reshape(2, 3).numpy())
    assert new_score < old_score


def test_stair_uses_full_time_series_for_temporal_acceptance(monkeypatch):
    load_window = torch.tensor(
        [[[100, 90, 1, 1], [100, 1, 90, 1]], [[90, 100, 1, 1], [90, 1, 100, 1]]],
        dtype=torch.int32,
    )
    placement = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    candidate = np.array([[0, 2, 1, 3], [0, 2, 1, 3]], dtype=np.int64)
    observed_load = None

    def fake_candidate(logical_load, current_placement, num_ranks):
        nonlocal observed_load
        observed_load = logical_load.copy()
        return candidate.copy()

    monkeypatch.setattr(
        "vllm_ascend.distributed.stair_policy._build_incremental_candidate",
        fake_candidate,
    )
    result = _rebalance(StairEplbPolicy(), load_window, placement)

    np.testing.assert_array_equal(observed_load, load_window.sum(dim=0).numpy())
    torch.testing.assert_close(result[0], torch.from_numpy(candidate[0]))
    torch.testing.assert_close(result[1], placement[1])


def test_stair_keeps_zero_load_and_balanced_placement():
    placement = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    zero_result = _rebalance(
        StairEplbPolicy(),
        torch.zeros((4, 1, 4), dtype=torch.float64),
        placement,
    )
    balanced_result = _rebalance(
        StairEplbPolicy(),
        torch.ones((4, 1, 4), dtype=torch.float64),
        placement,
    )

    torch.testing.assert_close(zero_result, placement)
    torch.testing.assert_close(balanced_result, placement)


def test_stair_uses_temporal_hysteresis_and_absolute_thresholds():
    policy = StairEplbPolicy()
    policy.average_to_peak_history[0] = 1.0

    assert not policy._needs_temporal_update(0, current_score=1 / 0.96, num_ranks=2)
    assert policy._needs_temporal_update(0, current_score=1 / 0.94, num_ranks=2)

    policy.average_to_peak_history[0] = 0.92
    assert policy._needs_temporal_update(0, current_score=1 / 0.89, num_ranks=2)


def test_stair_commits_history_only_after_real_layer_commit():
    policy = StairEplbPolicy()
    load_window = torch.tensor([[[100, 100, 1, 1]], [[1, 1, 100, 100]]], dtype=torch.int32)
    placement = torch.tensor([0, 2, 1, 3], dtype=torch.long)

    policy.commit_layer(load_window, 0, placement, num_ranks=2)

    assert set(policy.average_to_peak_history) == {0}
    expected_score = compute_balance_score(load_window[:, 0].numpy(), placement.reshape(2, 2).numpy())
    assert policy.average_to_peak_history[0] == pytest.approx(1 / expected_score)


def test_stair_clears_history_when_committed_placement_is_not_observed():
    policy = StairEplbPolicy()
    load_window = torch.ones((2, 1, 4), dtype=torch.float64)
    current = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    committed = torch.tensor([0, 2, 1, 3], dtype=torch.long)
    policy.commit_layer(load_window, 0, committed, num_ranks=2)

    _rebalance(policy, load_window, current)

    assert policy.average_to_peak_history == {}


@pytest.mark.parametrize(
    ("load", "placement", "error"),
    [
        (
            torch.ones((2, 1, 4)),
            None,
            "requires the current",
        ),
        (
            torch.ones((2, 1, 4)),
            torch.tensor([[0, 1, 2, 4]]),
            "invalid logical expert",
        ),
        (
            torch.ones((2, 1, 4)),
            torch.tensor([[0, 0, 1, 2]]),
            "at least one physical replica",
        ),
        (
            torch.tensor([[[1.0, -1.0, 1.0, 1.0]]]),
            torch.tensor([[0, 1, 2, 3]]),
            "finite, non-negative",
        ),
    ],
)
def test_stair_rejects_invalid_inputs(
    load: torch.Tensor,
    placement: torch.Tensor | None,
    error: str,
):
    with pytest.raises(ValueError, match=error):
        StairEplbPolicy().rebalance_experts(
            load,
            num_replicas=4,
            num_groups=1,
            num_nodes=1,
            num_ranks=2,
            old_global_expert_indices=placement,
        )
