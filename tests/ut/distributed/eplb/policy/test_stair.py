# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest

import numpy as np
from vllm.distributed.eplb.policy import AbstractEplbPolicy

from vllm_ascend.distributed.eplb.policy.stair import StairEplbPolicy


class TestStairLoadStatistics(unittest.TestCase):
    def test_policy_shares_upstream_abstract_base(self):
        self.assertTrue(issubclass(StairEplbPolicy, AbstractEplbPolicy))

    def test_compression_preserves_all_steps_as_weighted_bins(self):
        samples = np.arange(20).reshape(5, 2, 2)

        compressed, weights = StairEplbPolicy.compress_load_window(samples, 2)

        np.testing.assert_array_equal(weights, [2, 3])
        np.testing.assert_allclose(compressed[0], samples[:2].mean(axis=0))
        np.testing.assert_allclose(compressed[1], samples[2:].mean(axis=0))
        np.testing.assert_allclose(np.average(compressed, axis=0, weights=weights), samples.mean(axis=0))

    def test_weighted_moments_use_covariance(self):
        samples = np.array([[1.0, 4.0], [3.0, 2.0]])
        weights = np.array([2, 1])
        expanded = np.repeat(samples, weights, axis=0)

        mean, variance, covariance = StairEplbPolicy.weighted_moments(samples, weights)

        np.testing.assert_allclose(mean, expanded.mean(axis=0))
        np.testing.assert_allclose(variance, expanded.var(axis=0, ddof=1))
        np.testing.assert_allclose(covariance, np.cov(expanded, rowvar=False))
        self.assertFalse(np.shares_memory(variance, covariance))

    def test_single_sample_has_zero_covariance(self):
        mean, variance, covariance = StairEplbPolicy.weighted_moments(np.array([[2.0, 3.0]]), np.array([1]))

        np.testing.assert_array_equal(mean, [2.0, 3.0])
        np.testing.assert_array_equal(variance, np.zeros(2))
        np.testing.assert_array_equal(covariance, np.zeros((2, 2)))

    def test_score_uses_mean_and_weighted_nearest_rank_p95(self):
        samples = np.array([[8.0, 0.0], [4.0, 4.0]])
        weights = np.array([1, 19])

        imbalance = StairEplbPolicy.placement_imbalance(samples, weights, np.array([[0], [1]]))

        self.assertEqual(imbalance.mean_ratio, 1.05)
        self.assertEqual(imbalance.p95_ratio, 1.0)

    def test_replica_counts_reject_invalid_placements(self):
        for placement in (np.array([[0, 0], [1, 2]]), np.array([[0], [1]])):
            with self.subTest(placement=placement), self.assertRaises(ValueError):
                StairEplbPolicy.placement_replica_counts(placement, 3)

    def test_expert_risk_uses_mean_and_variance(self):
        risk = StairEplbPolicy.expert_risk(np.array([1.0, 2.0]), np.array([4.0, 0.0]), 0.5)

        np.testing.assert_array_equal(risk, [2.0, 2.0])

    def test_replica_search_is_bounded_and_deterministic(self):
        kwargs = dict(
            num_stages=3,
            budget_radius=2,
            beam_size=4,
            candidate_score=lambda value: float(np.square(value - 2).sum()),
        )

        first = StairEplbPolicy.replica_candidates(np.array([8.0, 4.0, 2.0]), 6, 3, **kwargs)
        second = StairEplbPolicy.replica_candidates(np.array([8.0, 4.0, 2.0]), 6, 3, **kwargs)

        self.assertTrue(first)
        self.assertLessEqual(len(first), 4)
        self.assertEqual([item.tolist() for item in first], [item.tolist() for item in second])
        self.assertEqual(len({tuple(item) for item in first}), len(first))
        self.assertTrue(all(item.sum() == 6 and np.all((item >= 1) & (item <= 3)) for item in first))

    def test_replica_search_supports_zero_redundancy(self):
        candidates = StairEplbPolicy.replica_candidates(
            np.array([8.0, 4.0, 2.0]),
            3,
            3,
            num_stages=3,
            budget_radius=2,
            beam_size=4,
            candidate_score=lambda value: float(value.sum()),
        )

        self.assertEqual(len(candidates), 1)
        np.testing.assert_array_equal(candidates[0], [1, 1, 1])

    def test_replica_search_caps_each_expert_at_one_copy_per_rank(self):
        candidates = StairEplbPolicy.replica_candidates(
            np.array([8.0, 3.0]),
            6,
            3,
            num_stages=1,
            budget_radius=0,
            beam_size=1,
            candidate_score=lambda value: float(value.max()),
        )

        np.testing.assert_array_equal(candidates[0], [3, 3])

    def test_replica_search_covers_small_valid_topologies(self):
        for num_experts in range(1, 6):
            for num_ranks in range(1, 4):
                for total_slots in range(num_ranks, num_experts * num_ranks + 1, num_ranks):
                    if total_slots < num_experts:
                        continue
                    with self.subTest(
                        num_experts=num_experts,
                        num_ranks=num_ranks,
                        total_slots=total_slots,
                    ):
                        candidates = StairEplbPolicy.replica_candidates(
                            np.arange(num_experts, 0, -1),
                            total_slots,
                            num_ranks,
                            num_stages=4,
                            budget_radius=2,
                            beam_size=8,
                            candidate_score=lambda value: float(np.square(value).sum()),
                        )
                        self.assertTrue(candidates)
                        self.assertTrue(
                            all(
                                candidate.sum() == total_slots and np.all((candidate >= 1) & (candidate <= num_ranks))
                                for candidate in candidates
                            )
                        )

    def test_replica_search_scores_only_final_candidates(self):
        calls = []

        candidates = StairEplbPolicy.replica_candidates(
            np.arange(8.0, 0.0, -1.0),
            16,
            4,
            num_stages=4,
            budget_radius=4,
            beam_size=8,
            candidate_score=lambda value: calls.append(tuple(value)) or float(np.square(value).sum()),
        )

        self.assertEqual(len(calls), len(candidates))
        self.assertLessEqual(len(calls), 8)

    def test_replica_search_rejects_invalid_topology_and_controls(self):
        kwargs = dict(
            num_stages=2,
            budget_radius=1,
            beam_size=4,
            candidate_score=lambda value: float(value.sum()),
        )
        with self.assertRaises(ValueError):
            StairEplbPolicy.replica_candidates(np.ones(3), 4, 3, **kwargs)
        with self.assertRaises(ValueError):
            StairEplbPolicy.replica_candidates(np.ones(3), 6, 3, **(kwargs | {"budget_radius": 1.0}))

    def test_zero_radius_matches_greedy_replica_allocation(self):
        risk = np.array([8.0, 4.0, 2.0])
        expected = StairEplbPolicy._allocate_extra_replicas(risk, np.ones(3, dtype=np.int64), 3, 3, (0, 1, 2))

        candidates = StairEplbPolicy.replica_candidates(
            risk,
            6,
            3,
            num_stages=3,
            budget_radius=0,
            beam_size=4,
            candidate_score=lambda value: float(value.max()),
        )

        self.assertEqual(len(candidates), 1)
        np.testing.assert_array_equal(candidates[0], expected)

    def test_statistics_reject_invalid_inputs(self):
        invalid_samples = np.array([[[1.0, -1.0]]])
        with self.assertRaises(ValueError):
            StairEplbPolicy.compress_load_window(invalid_samples, 2)
        with self.assertRaises(ValueError):
            StairEplbPolicy.weighted_moments(np.ones((2, 2)), np.array([1.0, 1.0]))
        with self.assertRaises(ValueError):
            StairEplbPolicy.placement_imbalance(np.ones((1, 2)), np.array([0]), np.array([[0], [1]]))
