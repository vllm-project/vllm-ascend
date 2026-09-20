# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from vllm.distributed.eplb.policy import AbstractEplbPolicy

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.policy.stair import (
    LayerPlan,
    PlacementImbalance,
    PlacementPlan,
    StairEplbPolicy,
    StairPlan,
)


class TestStairLoadStatistics(unittest.TestCase):
    def test_policy_implements_upstream_abstract_contract(self):
        self.assertTrue(issubclass(StairEplbPolicy, AbstractEplbPolicy))
        self.assertFalse(StairEplbPolicy.__abstractmethods__)

    def test_upstream_policy_contract_runs_stair_and_flattens_placement(self):
        result = StairEplbPolicy(StairConfig()).rebalance_experts(
            torch.tensor([[8.0, 7.0, 6.0, 5.0]]),
            num_replicas=4,
            num_groups=1,
            num_nodes=1,
            num_ranks=2,
            old_global_expert_indices=torch.tensor([[0, 1, 2, 3]]),
        )

        self.assertEqual(result.device, torch.device("cpu"))
        self.assertEqual(result.dtype, torch.int64)
        torch.testing.assert_close(result, torch.tensor([[0, 3, 2, 1]]))

    def test_upstream_policy_contract_requires_current_placement(self):
        with self.assertRaisesRegex(ValueError, "current expert placement"):
            StairEplbPolicy(StairConfig()).rebalance_experts(torch.ones((1, 2)), 2, 1, 1, 2)

    def test_compression_preserves_all_steps_as_weighted_bins(self):
        samples = np.arange(20).reshape(5, 2, 2)

        compressed, weights = StairEplbPolicy.compress_load_window(samples, 2)

        np.testing.assert_array_equal(weights, [2, 3])
        np.testing.assert_allclose(compressed[0], samples[:2].mean(axis=0))
        np.testing.assert_allclose(compressed[1], samples[2:].mean(axis=0))
        np.testing.assert_allclose(np.average(compressed, axis=0, weights=weights), samples.mean(axis=0))

    def test_temporal_bins_preserve_load_and_sample_counts(self):
        samples = torch.arange(20).reshape(5, 2, 2)
        policy = StairEplbPolicy(StairConfig(load_window_bins=2))

        prepared = policy.prepare_local_load_stats(samples)
        bin_means, numpy_sample_counts = StairEplbPolicy.compress_load_window(samples.numpy(), 2)

        torch.testing.assert_close(prepared.values[0], samples[:2].sum(dim=0))
        torch.testing.assert_close(prepared.values[1], samples[2:].sum(dim=0))
        np.testing.assert_array_equal(prepared.sample_counts, [2, 3])
        np.testing.assert_array_equal(prepared.sample_counts, numpy_sample_counts)
        np.testing.assert_allclose(
            prepared.values.numpy() / prepared.sample_counts[:, None, None],
            bin_means,
        )

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

    def test_layer_gate_skips_zero_load(self):
        result = StairEplbPolicy.gated_layer_imbalance(
            np.zeros((1, 2)), np.ones(1, dtype=np.int64), np.array([[0], [1]]), None, StairConfig()
        )

        self.assertIsNone(result)

    def test_layer_gate_accepts_first_nonzero_window(self):
        for anchor in (None, np.nan):
            with self.subTest(anchor=anchor):
                imbalance = StairEplbPolicy.gated_layer_imbalance(
                    np.array([[11.0, 9.0]]),
                    np.ones(1, dtype=np.int64),
                    np.array([[0], [1]]),
                    anchor,
                    StairConfig(),
                )

                self.assertIsNotNone(imbalance)
                self.assertEqual(imbalance.mean_ratio, 1.1)

    def test_layer_gate_accepts_relative_deterioration(self):
        imbalance = StairEplbPolicy.gated_layer_imbalance(
            np.array([[3.0, 2.0]]),
            np.ones(1, dtype=np.int64),
            np.array([[0], [1]]),
            1.1,
            StairConfig(absolute_balance_threshold=0.5),
        )

        self.assertIsNotNone(imbalance)
        self.assertEqual(imbalance.mean_ratio, 1.2)

    def test_layer_gate_accepts_absolute_imbalance(self):
        imbalance = StairEplbPolicy.gated_layer_imbalance(
            np.array([[3.0, 2.0]]),
            np.ones(1, dtype=np.int64),
            np.array([[0], [1]]),
            1.3,
            StairConfig(),
        )

        self.assertIsNotNone(imbalance)
        self.assertEqual(imbalance.mean_ratio, 1.2)

    def test_layer_gate_rejects_stable_balanced_layer(self):
        result = StairEplbPolicy.gated_layer_imbalance(
            np.array([[11.0, 9.0]]),
            np.ones(1, dtype=np.int64),
            np.array([[0], [1]]),
            1.1,
            StairConfig(),
        )

        self.assertIsNone(result)

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

    def test_lpt_placement_co_locates_negatively_correlated_experts(self):
        means = np.zeros(4)
        variances = np.ones(4)
        covariance = np.eye(4)
        covariance[0, 1] = covariance[1, 0] = -0.9

        placement = StairEplbPolicy.lpt_placement(
            means,
            variances,
            covariance,
            np.ones(4, dtype=np.int64),
            num_ranks=2,
            z_score=1.0,
            current_rank_expert_ids=np.array([[0, 1], [2, 3]]),
            rank_node_ids=np.zeros(2, dtype=np.int64),
            rank_pair_migration_limit=1,
            backtrack_limit=0,
        )

        np.testing.assert_array_equal(placement.rank_expert_ids, [[0, 1], [2, 3]])

    def test_lpt_placement_is_deterministic_and_preserves_replica_constraints(self):
        kwargs = dict(
            expert_means=np.array([8.0, 3.0, 1.0]),
            expert_variances=np.array([2.0, 1.0, 0.5]),
            expert_covariance=np.diag([2.0, 1.0, 0.5]),
            replica_counts=np.array([2, 1, 1]),
            num_ranks=2,
            z_score=0.5,
            current_rank_expert_ids=np.array([[0, 1], [0, 2]]),
            rank_node_ids=np.zeros(2, dtype=np.int64),
            rank_pair_migration_limit=1,
            backtrack_limit=0,
        )

        first = StairEplbPolicy.lpt_placement(**kwargs)
        second = StairEplbPolicy.lpt_placement(**kwargs)

        np.testing.assert_array_equal(first.rank_expert_ids, second.rank_expert_ids)
        np.testing.assert_array_equal(first.source_rank_ids, second.source_rank_ids)
        np.testing.assert_array_equal(first.source_slot_ids, second.source_slot_ids)
        self.assertTrue(all(len(set(rank)) == len(rank) for rank in first.rank_expert_ids.tolist()))
        np.testing.assert_array_equal(StairEplbPolicy.placement_replica_counts(first.rank_expert_ids, 3), [2, 1, 1])

    def test_lpt_placement_rejects_invalid_covariance_shape(self):
        with self.assertRaises(ValueError):
            StairEplbPolicy.lpt_placement(
                np.ones(2),
                np.ones(2),
                np.ones((2, 3)),
                np.ones(2, dtype=np.int64),
                2,
                0.5,
                current_rank_expert_ids=np.array([[0], [1]]),
                rank_node_ids=np.zeros(2, dtype=np.int64),
                rank_pair_migration_limit=1,
                backtrack_limit=0,
            )

    def test_lpt_placement_rejects_uneven_rank_capacity(self):
        with self.assertRaises(ValueError):
            StairEplbPolicy.lpt_placement(
                np.ones(2),
                np.ones(2),
                np.eye(2),
                np.array([1, 2]),
                2,
                0.5,
                current_rank_expert_ids=np.array([[0], [1]]),
                rank_node_ids=np.zeros(2, dtype=np.int64),
                rank_pair_migration_limit=1,
                backtrack_limit=0,
            )

    def test_lpt_placement_rejects_invalid_rank_nodes(self):
        with self.assertRaisesRegex(ValueError, "rank_node_ids"):
            StairEplbPolicy.lpt_placement(
                np.ones(2),
                np.ones(2),
                np.eye(2),
                np.ones(2, dtype=np.int64),
                2,
                0.5,
                current_rank_expert_ids=np.array([[0], [1]]),
                rank_node_ids=np.array([0]),
                rank_pair_migration_limit=1,
                backtrack_limit=0,
            )

    def test_migration_sources_enforce_directed_rank_pair_limit(self):
        current = np.array([[0, 1], [2, 3], [4, 5]])
        target = np.array([[2, 3], [0, 4], [1, 5]])
        expert_sources = [np.flatnonzero(np.any(current == expert, axis=1)).tolist() for expert in range(6)]

        self.assertIsNone(StairEplbPolicy._migration_sources(current, target, 1, expert_sources))
        sources = StairEplbPolicy._migration_sources(current, target, 2, expert_sources)

        np.testing.assert_array_equal(sources, [[1, 1], [0, 2], [0, 2]])

    def test_migration_sources_reassign_earlier_demand(self):
        current = np.array([[2, 3], [0, 1], [0, 4]])
        partial_target = np.array([[0, 1], [-1, -1], [-1, -1]])
        expert_sources = [np.flatnonzero(np.any(current == expert, axis=1)).tolist() for expert in range(5)]

        sources = StairEplbPolicy._migration_sources(current, partial_target, 1, expert_sources)

        np.testing.assert_array_equal(sources[0], [2, 1])

    def test_migration_sources_follow_multi_hop_augmenting_path(self):
        current = np.array([[3, 4, 5], [0, 2, 6], [0, 1, 7], [1, 8, 9]])
        partial_target = np.full_like(current, -1)
        partial_target[0] = [0, 1, 2]
        expert_sources = [np.where(current == expert)[0].tolist() for expert in range(10)]

        sources = StairEplbPolicy._migration_sources(current, partial_target, 1, expert_sources)

        np.testing.assert_array_equal(sources[0], [2, 3, 1])

    def test_minimum_cost_sources_prefer_same_node_then_lowest_rank(self):
        current = np.array([[1], [0], [0]])
        target = np.array([[0], [-1], [-1]])
        expert_sources = [[1, 2], [0]]

        same_node = StairEplbPolicy._minimum_cost_migration_sources(
            current, target, 1, expert_sources, np.array([0, 1, 0])
        )
        tied = StairEplbPolicy._minimum_cost_migration_sources(
            current, target, 1, expert_sources, np.zeros(3, dtype=np.int64)
        )

        self.assertEqual(same_node[0, 0], 2)
        self.assertEqual(tied[0, 0], 1)

    def test_minimum_cost_sources_preserve_global_topology_optimum(self):
        current = np.array([[2, 3], [0, 1], [0, 4], [1, 5]])
        target = np.full_like(current, -1)
        target[0] = [0, 1]
        expert_sources = [np.where(current == expert)[0].tolist() for expert in range(6)]

        sources = StairEplbPolicy._minimum_cost_migration_sources(
            current, target, 1, expert_sources, np.array([0, 0, 0, 1])
        )

        # Taking rank 1 for expert 0 would force expert 1 to cross nodes.
        np.testing.assert_array_equal(sources[0], [2, 1])

    def test_lpt_placement_aligns_slots_with_topology_aware_sources(self):
        current = np.array([[2, 0], [1, 0], [3, 1]])
        placement = StairEplbPolicy.lpt_placement(
            np.zeros(4),
            np.zeros(4),
            np.zeros((4, 4)),
            np.array([2, 2, 1, 1]),
            num_ranks=3,
            z_score=0.0,
            current_rank_expert_ids=current,
            rank_node_ids=np.array([0, 1, 0]),
            rank_pair_migration_limit=1,
            backtrack_limit=0,
        )

        self.assertIsNotNone(placement)
        np.testing.assert_array_equal(placement.rank_expert_ids, [[1, 0], [1, 0], [3, 2]])
        np.testing.assert_array_equal(placement.source_rank_ids, [[2, 0], [1, 1], [2, 0]])
        np.testing.assert_array_equal(placement.source_slot_ids, [[1, 1], [0, 1], [0, 0]])

    def test_lpt_variance_scales_by_replica_count(self):
        variance, scale = StairEplbPolicy._updated_rank_variance(
            expert=1,
            rank_experts=np.array([0]),
            current_variance=1.0,
            current_scale=1.0,
            expert_variances=np.array([4.0, 9.0]),
            expert_covariance=np.array([[4.0, 6.0], [6.0, 9.0]]),
            replica_counts=np.array([2, 3]),
        )

        self.assertEqual(variance, 4.0)
        self.assertEqual(scale, 4.0)

    def test_lpt_placement_backtracks_from_greedy_dead_end(self):
        means = np.array([100.0, 6.0, 5.0, 4.0, 3.0, 2.0, 3.0])
        current = np.array([[0, 1, 6], [2, 3, 6], [4, 5, 6]])
        kwargs = dict(
            expert_means=means,
            expert_variances=np.zeros(7),
            expert_covariance=np.zeros((7, 7)),
            replica_counts=np.array([1, 1, 1, 1, 1, 1, 3]),
            num_ranks=3,
            z_score=0.0,
            current_rank_expert_ids=current,
            rank_node_ids=np.zeros(3, dtype=np.int64),
            rank_pair_migration_limit=1,
        )

        # Greedy choices fill one rank too early, leaving no three distinct
        # ranks for the final expert even though a valid placement exists.
        self.assertIsNone(StairEplbPolicy.lpt_placement(**kwargs, backtrack_limit=3))
        placement = StairEplbPolicy.lpt_placement(**kwargs, backtrack_limit=4)

        self.assertIsNotNone(placement)
        for dst_rank, target_experts in enumerate(placement.rank_expert_ids):
            for slot, expert in enumerate(target_experts):
                self.assertIn(expert, current[placement.source_rank_ids[dst_rank, slot]])
            for src_rank in range(3):
                if src_rank != dst_rank:
                    self.assertLessEqual(np.sum(placement.source_rank_ids[dst_rank] == src_rank), 1)

    def test_lpt_placement_accepts_valid_covariance_with_strong_cancellation(self):
        deviations = np.array([-0.04, -0.01, 0.03, -0.15, 0.20, 43.74, 0.18, -43.95])
        center = np.full(8, 44.95)
        samples = np.stack([center + deviations, center - deviations])
        means, variances, covariance = StairEplbPolicy.weighted_moments(samples, np.ones(2, dtype=np.int64))

        placement = StairEplbPolicy.lpt_placement(
            means,
            variances,
            covariance,
            np.ones(8, dtype=np.int64),
            num_ranks=1,
            z_score=1.0,
            current_rank_expert_ids=np.arange(8).reshape(1, 8),
            rank_node_ids=np.zeros(1, dtype=np.int64),
            rank_pair_migration_limit=1,
            backtrack_limit=0,
        )

        self.assertIsNotNone(placement)

    def test_lpt_placement_rejects_small_indefinite_covariance(self):
        with self.assertRaises(ValueError):
            StairEplbPolicy.lpt_placement(
                np.zeros(2),
                np.full(2, 1e-30),
                np.array([[1e-30, -1e-15], [-1e-15, 1e-30]]),
                np.ones(2, dtype=np.int64),
                num_ranks=1,
                z_score=1.0,
                current_rank_expert_ids=np.array([[0, 1]]),
                rank_node_ids=np.zeros(1, dtype=np.int64),
                rank_pair_migration_limit=1,
                backtrack_limit=0,
            )

    def test_plan_layer_accepts_mean_improvement(self):
        samples = np.array([[8.0, 7.0, 6.0, 5.0]])
        current = np.array([[0, 1], [2, 3]])

        plan = StairEplbPolicy.plan_layer(
            samples,
            np.ones(1, dtype=np.int64),
            current,
            np.array([0, 1]),
            StairConfig(),
        )

        self.assertIsNotNone(plan)
        np.testing.assert_array_equal(plan.placement.rank_expert_ids, [[0, 3], [2, 1]])
        self.assertEqual(plan.predicted_imbalance.mean_ratio, 1.0)

    def test_plan_layer_skips_noop(self):
        self.assertIsNone(
            StairEplbPolicy.plan_layer(
                np.array([[8.0, 7.0, 6.0, 5.0]]),
                np.ones(1, dtype=np.int64),
                np.array([[0, 3], [2, 1]]),
                np.array([0, 1]),
                StairConfig(),
            )
        )

    def test_plan_layer_breaks_mean_tie_by_migration_count(self):
        plan = StairEplbPolicy.plan_layer(
            np.ones((1, 3)),
            np.ones(1, dtype=np.int64),
            np.array([[0, 2], [0, 1]]),
            np.array([0, 1]),
            StairConfig(),
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.predicted_imbalance.mean_ratio, 1.0)
        np.testing.assert_array_equal(plan.placement.rank_expert_ids, [[0, 2], [2, 1]])
        self.assertEqual(np.sum(plan.placement.source_rank_ids != np.arange(2)[:, None]), 1)

    def test_plan_layer_rejects_mean_regression(self):
        samples = np.array(
            [
                [21, 28, 26, 15],
                [28, 29, 29, 2],
                [13, 18, 8, 11],
                [18, 24, 17, 5],
                [20, 26, 6, 16],
            ]
        )

        plan = StairEplbPolicy.plan_layer(
            samples,
            np.array([2, 5, 1, 3, 5]),
            np.array([[0, 3], [1, 2]]),
            np.array([0, 1]),
            StairConfig(),
        )

        self.assertIsNone(plan)

    def test_plan_layer_rejects_p95_regression(self):
        placement = PlacementPlan(
            rank_expert_ids=np.array([[0, 2], [1, 3]]),
            source_rank_ids=np.array([[0, 1], [0, 1]]),
            source_slot_ids=np.array([[0, 0], [1, 1]]),
        )
        with (
            patch.object(StairEplbPolicy, "replica_candidates", return_value=[np.ones(4, dtype=np.int64)]),
            patch.object(StairEplbPolicy, "lpt_placement", return_value=placement),
            patch.object(
                StairEplbPolicy,
                "placement_imbalance",
                side_effect=[PlacementImbalance(1.1, 1.0), PlacementImbalance(1.0, 1.1)],
            ),
        ):
            plan = StairEplbPolicy.plan_layer(
                np.ones((1, 4)),
                np.ones(1, dtype=np.int64),
                np.array([[0, 1], [2, 3]]),
                np.array([0, 1]),
                StairConfig(),
            )

        self.assertIsNone(plan)

    def test_plan_rebalance_preserves_filtered_layers(self):
        samples = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [8.0, 7.0, 6.0, 5.0],
                ]
            ]
        )
        current = np.array(
            [
                [[0, 1], [2, 3]],
                [[0, 3], [2, 1]],
                [[0, 1], [2, 3]],
            ]
        )

        plan = StairEplbPolicy.plan_rebalance(
            samples,
            current,
            np.array([np.nan, 1.0, np.nan]),
            np.array([0, 1]),
            StairConfig(),
        )

        np.testing.assert_array_equal(plan.rank_expert_ids[:2], current[:2])
        np.testing.assert_array_equal(plan.source_rank_ids[:2], [[[0, 0], [1, 1]]] * 2)
        np.testing.assert_array_equal(plan.source_slot_ids[:2], [[[0, 1], [0, 1]]] * 2)
        np.testing.assert_array_equal(plan.rank_expert_ids[2], [[0, 3], [2, 1]])
        np.testing.assert_array_equal(np.isnan(plan.predicted_mean_ratios), [True, True, False])
        self.assertEqual(plan.predicted_mean_ratios[2], 1.0)

    def test_plan_rebalance_orders_equal_mean_ratios_by_relative_deterioration(self):
        samples = np.array([[[3.0, 1.0], [6.0, 2.0]]])
        current = np.array([[[0], [1]], [[0], [1]]])
        planned_load_totals = []

        # Both layers have ratio 1.5; layer 0 deteriorated further from its anchor.
        with patch.object(
            StairEplbPolicy,
            "plan_layer",
            side_effect=lambda load, *_: planned_load_totals.append(float(load.sum())),
        ):
            StairEplbPolicy.plan_rebalance(
                samples,
                current,
                np.array([1.1, 1.3]),
                np.array([0, 1]),
                StairConfig(),
            )

        self.assertEqual(planned_load_totals, [4.0, 8.0])

    def test_plan_rebalance_limits_work_to_layer_shard(self):
        samples = np.array([[[3.0, 1.0], [6.0, 2.0], [9.0, 3.0]]])
        current = np.array([[[0], [1]]] * 3)
        planned_load_totals = []

        def plan_layer(load, *_):
            planned_load_totals.append(float(load.sum()))
            placement = PlacementPlan(
                rank_expert_ids=np.array([[1], [0]]),
                source_rank_ids=np.array([[1], [0]]),
                source_slot_ids=np.zeros((2, 1), dtype=np.int64),
            )
            return LayerPlan(placement, PlacementImbalance(1.0, 1.0))

        with patch.object(
            StairEplbPolicy,
            "plan_layer",
            side_effect=plan_layer,
        ):
            plan = StairEplbPolicy.plan_rebalance(
                samples,
                current,
                np.full(3, np.nan),
                np.array([0, 1]),
                StairConfig(),
                layer_ids=(1,),
            )

        self.assertEqual(planned_load_totals, [8.0])
        np.testing.assert_array_equal(plan.rank_expert_ids, [current[0], [[1], [0]], current[2]])
        np.testing.assert_array_equal(plan.source_rank_ids, [[[0], [1]], [[1], [0]], [[0], [1]]])
        np.testing.assert_array_equal(plan.source_slot_ids, np.zeros((3, 2, 1), dtype=np.int64))
        np.testing.assert_array_equal(np.isnan(plan.predicted_mean_ratios), [True, False, True])

    def test_plan_rebalance_rejects_invalid_layer_shard(self):
        with self.assertRaisesRegex(ValueError, "stage-local"):
            StairEplbPolicy.plan_rebalance(
                np.ones((1, 1, 2)),
                np.array([[[0], [1]]]),
                np.array([np.nan]),
                np.array([0, 1]),
                StairConfig(),
                layer_ids=(0, 0),
            )

    @patch("vllm_ascend.distributed.eplb.policy.stair.all_gather_layer_shards")
    @patch("vllm_ascend.distributed.eplb.policy.stair.torch.distributed.all_reduce")
    @patch.object(StairEplbPolicy, "plan_rebalance")
    def test_plan_sharded_rebalance_gathers_complete_plan(self, plan_rebalance, all_reduce, all_gather):
        current = np.array([[[0], [1]]] * 3)
        source_ranks = np.array([[[1], [0]]] * 3)
        source_slots = np.zeros_like(current)
        complete_plan = StairPlan(
            rank_expert_ids=np.array([[[1], [0]]] * 3),
            source_rank_ids=source_ranks,
            source_slot_ids=source_slots,
            predicted_mean_ratios=np.ones(3),
        )
        local_plan = StairPlan(
            rank_expert_ids=np.array([current[0], [[1], [0]], current[2]]),
            source_rank_ids=np.array([[[0], [1]], [[1], [0]], [[0], [1]]]),
            source_slot_ids=source_slots,
            predicted_mean_ratios=np.array([np.nan, 1.0, np.nan]),
        )
        plan_rebalance.return_value = local_plan
        pending_field_gathers = [
            (local_plan.rank_expert_ids[1::2], complete_plan.rank_expert_ids),
            (local_plan.source_rank_ids[1::2], complete_plan.source_rank_ids),
            (local_plan.source_slot_ids[1::2], complete_plan.source_slot_ids),
            (local_plan.predicted_mean_ratios[1::2], complete_plan.predicted_mean_ratios),
        ]

        def gather(local_values, num_layers, cpu_group):
            expected_local, complete = pending_field_gathers.pop(0)
            self.assertEqual(num_layers, 3)
            self.assertIs(cpu_group, group)
            torch.testing.assert_close(local_values, torch.from_numpy(expected_local), equal_nan=True)
            return torch.from_numpy(complete.copy())

        group = Mock()
        group.rank.return_value = 1
        group.size.return_value = 2
        all_gather.side_effect = gather

        plan = StairEplbPolicy.plan_sharded_rebalance(
            np.ones((1, 3, 2)),
            current,
            np.full(3, np.nan),
            np.array([0, 1]),
            StairConfig(),
            group,
        )

        self.assertFalse(pending_field_gathers)
        self.assertEqual(all_reduce.call_args.args[0].item(), 1)
        self.assertEqual(plan_rebalance.call_args.kwargs["layer_ids"], range(1, 3, 2))
        np.testing.assert_array_equal(plan.rank_expert_ids, complete_plan.rank_expert_ids)
        np.testing.assert_array_equal(plan.source_rank_ids, complete_plan.source_rank_ids)
        np.testing.assert_array_equal(plan.source_slot_ids, complete_plan.source_slot_ids)
        np.testing.assert_array_equal(plan.predicted_mean_ratios, complete_plan.predicted_mean_ratios)

    @patch("vllm_ascend.distributed.eplb.policy.stair.all_gather_layer_shards")
    @patch("vllm_ascend.distributed.eplb.policy.stair.torch.distributed.all_reduce")
    @patch.object(StairEplbPolicy, "plan_rebalance", side_effect=ValueError("invalid local shard"))
    def test_plan_sharded_rebalance_synchronizes_local_failure(self, _, all_reduce, all_gather):
        group = Mock()
        group.rank.return_value = 0
        group.size.return_value = 2

        with self.assertRaisesRegex(ValueError, "invalid local shard"):
            StairEplbPolicy.plan_sharded_rebalance(
                np.ones((1, 2, 2)),
                np.array([[[0], [1]]] * 2),
                np.full(2, np.nan),
                np.array([0, 1]),
                StairConfig(),
                group,
            )

        self.assertEqual(all_reduce.call_args.args[0].item(), 0)
        all_gather.assert_not_called()

    @patch("vllm_ascend.distributed.eplb.policy.stair.all_gather_layer_shards")
    @patch("vllm_ascend.distributed.eplb.policy.stair.torch.distributed.all_reduce")
    @patch.object(StairEplbPolicy, "plan_rebalance")
    def test_plan_sharded_rebalance_stops_for_remote_failure(self, plan_rebalance, all_reduce, all_gather):
        current = np.array([[[0], [1]]] * 2)
        plan_rebalance.return_value = StairPlan(
            rank_expert_ids=current.copy(),
            source_rank_ids=np.array([[[0], [1]]] * 2),
            source_slot_ids=np.zeros_like(current),
            predicted_mean_ratios=np.full(2, np.nan),
        )
        all_reduce.side_effect = lambda status, **_: status.zero_()
        group = Mock()
        group.rank.return_value = 0
        group.size.return_value = 2

        with self.assertRaisesRegex(RuntimeError, "another EPLB rank"):
            StairEplbPolicy.plan_sharded_rebalance(
                np.ones((1, 2, 2)),
                current,
                np.full(2, np.nan),
                np.array([0, 1]),
                StairConfig(),
                group,
            )

        all_gather.assert_not_called()

    def test_validate_plan_accepts_explicit_sources(self):
        current = np.array([[[0, 1], [2, 3]]])
        plan = StairPlan(
            rank_expert_ids=np.array([[[0, 2], [1, 3]]]),
            source_rank_ids=np.array([[[0, 1], [0, 1]]]),
            source_slot_ids=np.array([[[0, 0], [1, 1]]]),
            predicted_mean_ratios=np.array([1.0]),
        )

        StairEplbPolicy.validate_plan(current, plan, num_experts=4, rank_pair_migration_limit=1)

    def test_validate_plan_rejects_false_source_ownership(self):
        current = np.array([[[0, 1], [2, 3]]])
        plan = StairPlan(
            rank_expert_ids=np.array([[[0, 2], [1, 3]]]),
            source_rank_ids=np.array([[[0, 1], [0, 1]]]),
            source_slot_ids=np.array([[[0, 1], [1, 1]]]),
            predicted_mean_ratios=np.array([1.0]),
        )

        with self.assertRaisesRegex(ValueError, "does not own"):
            StairEplbPolicy.validate_plan(current, plan, num_experts=4, rank_pair_migration_limit=1)

    def test_validate_plan_rejects_invalid_controls_and_pair_limit(self):
        current = np.array([[[0, 1], [2, 3], [4, 5]]])
        plan = StairPlan(
            rank_expert_ids=np.array([[[2, 3], [0, 1], [4, 5]]]),
            source_rank_ids=np.array([[[1, 1], [0, 0], [2, 2]]]),
            source_slot_ids=np.array([[[0, 1], [0, 1], [0, 1]]]),
            predicted_mean_ratios=np.array([1.0]),
        )

        for num_experts, pair_limit in ((6, 1), (6, 1.5), (6, np.nan), (6, np.inf), (6, True), (6.0, 1)):
            with self.subTest(num_experts=num_experts, pair_limit=pair_limit), self.assertRaises(ValueError):
                StairEplbPolicy.validate_plan(current, plan, num_experts, pair_limit)

    def test_validate_plan_rejects_prediction_for_unchanged_layer(self):
        current = np.array([[[0], [1]]])
        plan = StairPlan(
            rank_expert_ids=current.copy(),
            source_rank_ids=np.array([[[0], [1]]]),
            source_slot_ids=np.zeros_like(current),
            predicted_mean_ratios=np.array([1.0]),
        )

        with self.assertRaisesRegex(ValueError, "finite for changed layers"):
            StairEplbPolicy.validate_plan(current, plan, num_experts=2, rank_pair_migration_limit=1)

    def test_validate_plan_rejects_non_float_predictions(self):
        current = np.array([[[0], [1]]])
        plan = StairPlan(
            rank_expert_ids=current.copy(),
            source_rank_ids=np.array([[[0], [1]]]),
            source_slot_ids=np.zeros_like(current),
            predicted_mean_ratios=np.array([True]),
        )

        with self.assertRaisesRegex(ValueError, "floating-point"):
            StairEplbPolicy.validate_plan(current, plan, num_experts=2, rank_pair_migration_limit=1)

    def test_statistics_reject_invalid_inputs(self):
        invalid_samples = np.array([[[1.0, -1.0]]])
        with self.assertRaises(ValueError):
            StairEplbPolicy.compress_load_window(invalid_samples, 2)
        with self.assertRaises(ValueError):
            StairEplbPolicy.weighted_moments(np.ones((2, 2)), np.array([1.0, 1.0]))
        with self.assertRaises(ValueError):
            StairEplbPolicy.placement_imbalance(np.ones((1, 2)), np.array([0]), np.array([[0], [1]]))
