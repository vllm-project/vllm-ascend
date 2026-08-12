# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np
import pytest

from vllm_ascend.eplb.core.policy.policy_stair import StairEplbPolicy, compute_balance_score


def test_compute_balance_score_uses_peak_to_average_ratio():
    expert_load = np.array([[100, 1, 1, 1]], dtype=np.float64)
    placement = np.array([[0, 1], [2, 3]], dtype=np.int64)

    score = compute_balance_score(expert_load, placement)

    assert score == pytest.approx(101 / 51.5)


def test_stair_accepts_only_swift_candidates_that_improve_the_time_series():
    current = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
        dtype=np.int64,
    )
    candidate = np.array(
        [
            [0, 2, 1, 3],
            [0, 2, 1, 3],
        ],
        dtype=np.int64,
    )
    expert_load = np.array(
        [
            [[100, 90, 1, 1], [100, 1, 90, 1]],
            [[90, 100, 1, 1], [90, 1, 100, 1]],
        ],
        dtype=np.float64,
    )

    result = StairEplbPolicy().rebalance_experts(
        expert_load,
        current,
        candidate,
        num_ranks=2,
    )

    np.testing.assert_array_equal(result[0], candidate[0])
    np.testing.assert_array_equal(result[1], current[1])


def test_stair_keeps_zero_load_and_unchanged_candidates():
    current = np.array([[0, 1, 2, 3]], dtype=np.int64)
    candidate = np.array([[0, 2, 1, 3]], dtype=np.int64)

    zero_result = StairEplbPolicy().rebalance_experts(
        np.zeros((4, 1, 4), dtype=np.float64),
        current,
        candidate,
        num_ranks=2,
    )
    unchanged_result = StairEplbPolicy().rebalance_experts(
        np.ones((4, 1, 4), dtype=np.float64),
        current,
        current,
        num_ranks=2,
    )

    np.testing.assert_array_equal(zero_result, current)
    np.testing.assert_array_equal(unchanged_result, current)


def test_stair_uses_flashlb_hysteresis_and_absolute_thresholds():
    policy = StairEplbPolicy()
    policy.average_to_peak_history[0] = 1.0

    assert not policy._needs_flash_update(0, current_score=1 / 0.96, num_ranks=2)
    assert policy._needs_flash_update(0, current_score=1 / 0.94, num_ranks=2)

    policy.average_to_peak_history[0] = 0.92
    assert policy._needs_flash_update(0, current_score=1 / 0.89, num_ranks=2)


def test_stair_hysteresis_can_reject_an_improving_swift_candidate():
    policy = StairEplbPolicy()
    current = np.array([[0, 1, 2, 3]], dtype=np.int64)
    candidate = np.array([[0, 2, 1, 3]], dtype=np.int64)
    expert_load = np.tile(np.array([50, 51, 50, 49], dtype=np.float64), (4, 1))[:, None, :]
    current_score = compute_balance_score(expert_load[:, 0], current.reshape(2, 2))
    policy.average_to_peak_history[0] = 1 / current_score
    policy._topology = (1, 4, 4, 2)

    result = policy.rebalance_experts(
        expert_load,
        current,
        candidate,
        num_ranks=2,
    )

    np.testing.assert_array_equal(result, current)


def test_stair_commits_flashlb_history_only_for_changed_layers():
    policy = StairEplbPolicy()
    expert_load = np.tile(np.array([[100, 100, 1, 1], [1, 1, 100, 100]]), (4, 1, 1))
    committed = np.array([0, 2, 1, 3], dtype=np.int64)

    policy.commit_layer(expert_load, 0, committed, num_ranks=2)

    assert set(policy.average_to_peak_history) == {0}
    expected_score = compute_balance_score(expert_load[:, 0], committed.reshape(2, 2))
    assert policy.average_to_peak_history[0] == pytest.approx(1 / expected_score)


def test_stair_clears_history_when_committed_placement_is_not_observed():
    policy = StairEplbPolicy()
    expert_load = np.ones((2, 1, 4), dtype=np.float64)
    current = np.array([[0, 1, 2, 3]], dtype=np.int64)
    committed = np.array([0, 2, 1, 3], dtype=np.int64)
    policy.commit_layer(expert_load, 0, committed, num_ranks=2)

    policy.rebalance_experts(
        expert_load,
        current,
        current,
        num_ranks=2,
    )

    assert policy.average_to_peak_history == {}


@pytest.mark.parametrize(
    ("expert_load", "current", "candidate", "error"),
    [
        (
            np.ones((2, 1, 4)),
            np.array([[0, 1, 2, 4]]),
            np.array([[0, 1, 2, 3]]),
            "invalid logical expert",
        ),
        (
            np.ones((2, 1, 4)),
            np.array([[0, 1, 2, 3]]),
            np.array([[0, 0, 1, 2]]),
            "at least one physical replica",
        ),
        (
            np.ones((2, 1, 3)),
            np.array([[0, 1, 2, 0]]),
            np.array([[0, 0, 1, 2]]),
            "two replicas on the same rank",
        ),
        (
            np.array([[[1.0, -1.0, 1.0, 1.0]]]),
            np.array([[0, 1, 2, 3]]),
            np.array([[0, 1, 2, 3]]),
            "finite, non-negative",
        ),
        (
            np.ones((2, 1, 4)),
            np.array([[0, 1, 2, 3]]),
            np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
            "same shape",
        ),
    ],
)
def test_stair_rejects_invalid_inputs(
    expert_load: np.ndarray,
    current: np.ndarray,
    candidate: np.ndarray,
    error: str,
):
    with pytest.raises(ValueError, match=error):
        StairEplbPolicy().rebalance_experts(
            expert_load,
            current,
            candidate,
            num_ranks=2,
        )
