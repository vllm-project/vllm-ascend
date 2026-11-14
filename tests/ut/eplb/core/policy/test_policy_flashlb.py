#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Unit tests for the FlashLB EPLB policy (``vllm_ascend.eplb.core.policy.policy_flashlb``).

Covers the replica-allocation primitives (``min_max_replica``, ``max_delta_replica``,
``percentage_replica``) and their dispatch helper, the LPT deployment and device
variance updates, the load-balance score, the ``FlashTree`` search (statistics,
neighbor search, balanceness optimization), the ``FlashLB`` runtime state
(sliding-window hotness registration, incremental/statistical updates, update
gating), the redeployment-minimization helpers, and the end-to-end
``rebalance_experts`` entry point.
"""

import unittest
from unittest import mock

import numpy as np
import torch

from vllm_ascend.eplb.core.policy import policy_flashlb
from vllm_ascend.eplb.core.policy.policy_flashlb import (
    FlashLB,
    FlashTree,
    compute_score,
    compute_updated_device_variance,
    generate_layered_experts,
    lpt_deployment,
    make_replica,
    max_delta_replica,
    min_max_replica,
    percentage_replica,
)


class TestReplicaAllocationPrimitives(unittest.TestCase):
    """Hand-computed checks for the three replica allocation algorithms."""

    def test_min_max_replica_assigns_to_highest_unit_value(self):
        # unit_value = mu / replicas: expert 0 has 10.0, expert 1 has 4.0.
        mu = np.array([10.0, 4.0])
        var = np.array([0.0, 0.0])
        current = np.array([1, 1], dtype=np.int32)
        replicas, history = min_max_replica(mu, var, 2, current.copy(), 0.0)
        # Step 1: expert 0 (10 > 4) -> [2, 1], unit 0 becomes 5.0.
        # Step 2: expert 0 (5 > 4) -> [3, 1].
        np.testing.assert_array_equal(replicas, np.array([3, 1]))
        self.assertEqual(history.shape, (3, 2))
        np.testing.assert_array_equal(history[0], np.array([1, 1]))
        np.testing.assert_array_equal(history[1], np.array([2, 1]))
        np.testing.assert_array_equal(history[2], np.array([3, 1]))

    def test_min_max_replica_preserves_total_replicas(self):
        rng = np.random.default_rng(0)
        mu = rng.random(6) + 0.5
        var = rng.random(6) * 0.1
        current = rng.integers(1, 5, size=6).astype(np.int32)
        replicas, _ = min_max_replica(mu, var, 7, current.copy(), 0.5)
        self.assertEqual(replicas.sum(), current.sum() + 7)
        self.assertTrue((replicas >= current).all())

    def test_min_max_and_max_delta_diverge_on_shallow_drop(self):
        # Expert 0: mu=2.5, replicas=1 -> unit value 2.5, post-increment 2.5/2=1.25.
        # Expert 1: mu=6.0, replicas=2 -> unit value 3.0, post-increment 6.0/6=1.0.
        # min_max picks the highest *current* unit value (expert 1), while
        # max_delta picks the highest *post-increment* unit value (expert 0).
        mu = np.array([2.5, 6.0])
        var = np.array([0.0, 0.0])
        current = np.array([1, 2], dtype=np.int32)
        min_max, _ = min_max_replica(mu, var, 1, current.copy(), 0.0)
        np.testing.assert_array_equal(min_max, np.array([1, 3]))
        max_delta, _ = max_delta_replica(mu, var, 1, current.copy(), 0.0)
        np.testing.assert_array_equal(max_delta, np.array([2, 2]))

    def test_max_delta_replica_history_tracks_each_step(self):
        mu = np.array([6.0, 3.5])
        var = np.array([0.0, 0.0])
        current = np.array([2, 1], dtype=np.int32)
        _, history = max_delta_replica(mu, var, 2, current.copy(), 0.0)
        # unit values: expert0 6/2=3.0, expert1 3.5/1=3.5.
        # r=1: max_delta compares unit/(r+1): expert0 3.0/3=1.0, expert1 3.5/2=1.75
        # -> expert 1.
        np.testing.assert_array_equal(history[0], np.array([2, 1]))
        np.testing.assert_array_equal(history[1], np.array([2, 2]))
        # r=2: expert0 3.0/3=1.0 vs expert1 (updated) 1.75/3=0.5833 -> expert 0.
        np.testing.assert_array_equal(history[2], np.array([3, 2]))

    def test_percentage_replica_proportional_with_remainder(self):
        mu = np.array([3.0, 1.0])
        var = np.array([0.0, 0.0])
        current = np.array([0, 0], dtype=np.int32)
        replicas, history = percentage_replica(mu, var, 4, current, 0.0)
        # Quotas after 4 rounds: 3.0 / 1.0 -> expert 0 gets 3, expert 1 gets 1.
        np.testing.assert_array_equal(replicas, np.array([3, 1]))
        np.testing.assert_array_equal(history[4], np.array([3, 1]))

    def test_percentage_replica_zero_load_splits_evenly(self):
        mu = np.array([0.0, 0.0, 0.0])
        var = np.array([0.0, 0.0, 0.0])
        current = np.array([1, 1, 1], dtype=np.int32)
        replicas, _ = percentage_replica(mu, var, 5, current, 0.0)
        # Even split over 3 experts for 5 extra replicas: 2/2/1 (first two get extra).
        np.testing.assert_array_equal(replicas, np.array([3, 3, 2]))

    def test_percentage_replica_input_not_mutated(self):
        mu = np.array([3.0, 1.0])
        var = np.array([0.0, 0.0])
        current = np.array([1, 1], dtype=np.int32)
        current_copy = current.copy()
        percentage_replica(mu, var, 3, current, 0.0)
        np.testing.assert_array_equal(current, current_copy)

    def test_make_replica_dispatch(self):
        mu = np.array([10.0, 4.0])
        var = np.array([0.0, 0.0])
        current = np.array([1, 1], dtype=np.int32)
        mm, _ = min_max_replica(mu, var, 2, current.copy(), 0.0)
        md, _ = max_delta_replica(mu, var, 2, current.copy(), 0.0)
        pc, _ = percentage_replica(mu, var, 2, current.copy(), 0.0)
        for method, expected in (
            ("min_max_replica", mm),
            ("max_delta", md),
            ("percentage", pc),
        ):
            dispatched, _ = make_replica(mu, var, 2, current.copy(), 0.0, method)
            np.testing.assert_array_equal(dispatched, expected)
        # Unknown method falls back to min_max.
        fallback, _ = make_replica(mu, var, 2, current.copy(), 0.0, "unknown")
        np.testing.assert_array_equal(fallback, mm)

    def test_flashlb_min_max_wrapper(self):
        policy = FlashLB()
        mu = np.array([10.0, 4.0])
        var = np.array([0.0, 0.0])
        current = np.array([1, 1], dtype=np.int32)
        replicas, _ = policy.min_max_replica(mu, var, 2, current.copy(), 0.0)
        np.testing.assert_array_equal(replicas, np.array([3, 1]))


class TestDeviceVarianceAndLptDeployment(unittest.TestCase):
    """Device variance update and LPT greedy deployment."""

    def test_compute_updated_device_variance_with_covariance(self):
        # expert 1 already occupies the device with variance contribution 9/9=1.0
        # (var=9, replicas=3). Adding expert 0 (var=4, replicas=2) contributes 4/4=1.0
        # plus 2*cov[0,1]/(2*3) = 2*2/6.
        device_slots = np.array([1, -1, -1, -1], dtype=np.int32)
        expert_var = np.array([4.0, 9.0])
        expert_cov = np.array([[0.0, 2.0], [2.0, 0.0]])
        expert_replicas = np.array([2, 3], dtype=np.int32)
        result = compute_updated_device_variance(0, device_slots, 1.0, expert_var, expert_cov, expert_replicas)
        self.assertAlmostEqual(result, 1.0 + 1.0 + 2 * 2.0 / 6.0, places=6)

    def test_compute_updated_device_variance_empty_device(self):
        device_slots = np.array([-1, -1], dtype=np.int32)
        expert_var = np.array([5.0, 1.0])
        expert_cov = np.zeros((2, 2))
        expert_replicas = np.array([2, 4], dtype=np.int32)
        result = compute_updated_device_variance(1, device_slots, 0.0, expert_var, expert_cov, expert_replicas)
        self.assertAlmostEqual(result, 1.0 / 16.0, places=6)

    def test_lpt_deployment_balances_known_case(self):
        mu = np.array([9.0, 3.0, 2.0, 2.0])
        var = np.zeros(4)
        cov = np.zeros((4, 4))
        deployment = -np.ones((2, 4), dtype=np.int32)
        total_replicas = np.array([2, 2, 2, 2], dtype=np.int32)
        result = lpt_deployment(mu, var, cov, deployment, np.zeros(4, dtype=np.int32), total_replicas, 0.0)
        # Risk ties break to the lower device index. Greedy trace (unit load per
        # replica: e0 4.5, e1 1.5, e2/e3 1.0):
        # e0r1->dev0(4.5), e0r2->dev1(4.5), e1r1->dev0(6.0 tie), e1r2->dev1(6.0),
        # e2r1->dev0(7.0 tie), e2r2->dev1(7.0), e3r1->dev0(8.0 tie), e3r2->dev1(8.0).
        np.testing.assert_array_equal(result, np.array([[0, 1, 2, 3], [0, 1, 2, 3]]))

    def test_lpt_deployment_invariants(self):
        rng = np.random.default_rng(1)
        mu = rng.random(8) + 1.0
        mu.sort()
        var = rng.random(8) * 0.1
        cov = np.zeros((8, 8))
        num_devices, slots = 4, 6
        deployment = -np.ones((num_devices, slots), dtype=np.int32)
        total_replicas = np.full(8, 3, dtype=np.int32)  # 24 == 4 * 6
        result = lpt_deployment(mu, var, cov, deployment, np.zeros(8, dtype=np.int32), total_replicas, 0.5)
        self.assertEqual(result.shape, (num_devices, slots))
        # Every slot filled, every replica count satisfied, no expert duplicated on one device.
        self.assertTrue((result >= 0).all())
        counts = np.bincount(result.ravel(), minlength=8)
        np.testing.assert_array_equal(counts, total_replicas)
        for dev in range(num_devices):
            self.assertEqual(len(set(result[dev].tolist())), slots)

    def test_lpt_deployment_keeps_existing_placement(self):
        mu = np.array([9.0, 3.0])
        var = np.zeros(2)
        cov = np.zeros((2, 2))
        deployment = np.array([[0, -1], [-1, 1]], dtype=np.int32)
        total_replicas = np.array([2, 2], dtype=np.int32)
        result = lpt_deployment(mu, var, cov, deployment, np.array([1, 1], dtype=np.int32), total_replicas, 0.0)
        # The pre-placed experts keep their devices; only the free slots are filled.
        self.assertIn(0, result[0].tolist())
        self.assertIn(1, result[1].tolist())
        self.assertEqual(result[0].tolist().count(0), 1)
        self.assertEqual(result[1].tolist().count(1), 1)

    def test_lpt_deployment_prefers_lower_risk_device(self):
        mu = np.array([10.0, 9.0, 0.5, 0.5])
        var = np.zeros(4)
        cov = np.zeros((4, 4))
        deployment = -np.ones((2, 3), dtype=np.int32)
        total_replicas = np.array([2, 2, 1, 1], dtype=np.int32)
        result = lpt_deployment(mu, var, cov, deployment, np.zeros(4, dtype=np.int32), total_replicas, 0.0)
        # Replica counts are respected and no expert duplicates on one device.
        counts = np.bincount(result.ravel(), minlength=4)
        np.testing.assert_array_equal(counts, total_replicas)
        for dev in range(2):
            self.assertEqual(len(set(result[dev].tolist())), 3)
        # Both hot experts (0/1) get one replica per device: the deployment is
        # perfectly balanced, 19.5 load on each device.
        per_device_load = np.array([mu[result[0]].sum(), mu[result[1]].sum()])
        np.testing.assert_allclose(per_device_load, [19.5, 19.5], atol=1e-6)


class TestComputeScore(unittest.TestCase):
    """Load balance score: (max_device_load * D + eps) / (total_load + eps)."""

    def test_perfect_balance_scores_one(self):
        val_data = np.array([[3.0, 1.0], [1.0, 3.0]])
        replicas = np.array([1, 1], dtype=np.int32)
        deployment = np.array([[0, 1]], dtype=np.int32)
        score = compute_score(val_data, replicas, deployment)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_imbalanced_deployment_scores_above_one(self):
        val_data = np.array([[5.0, 5.0, 5.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        replicas = np.array([1, 1, 1, 1], dtype=np.int32)
        deployment = np.array([[0, 2], [1, 3]], dtype=np.int32)
        score = compute_score(val_data, replicas, deployment)
        # t0: device0 5+5=10, device1 5+1=6 -> (10*2+eps)/(16+eps) ~= 1.25.
        # t1: 1.0. mean ~= 1.125.
        self.assertAlmostEqual(score, (20.0 / 16.0 + 1.0) / 2.0, delta=1e-3)
        self.assertGreater(score, 1.0)

    def test_score_uses_replica_counts(self):
        # Same deployment, but expert 0 has 2 replicas: its per-replica load halves.
        val_data = np.array([[10.0, 1.0]])
        replicas_one = np.array([1, 1], dtype=np.int32)
        replicas_two = np.array([2, 1], dtype=np.int32)
        deployment = np.array([[0, 0], [1, 0]], dtype=np.int32)
        score_one = compute_score(val_data, replicas_one, deployment)
        score_two = compute_score(val_data, replicas_two, deployment)
        # device0: 10+10=20 vs 5+5=10; more replicas -> better balance -> lower score.
        self.assertLess(score_two, score_one)


class TestFlashTreeStatistics(unittest.TestCase):
    """FlashTree.compute_statistics and neighbor_search."""

    def test_compute_statistics_multi_step(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean, var, cov = FlashTree.compute_statistics(X)
        np.testing.assert_allclose(mean, [2.0, 3.0])
        # Sample variance with ddof=1: (1 + 1) / (2 - 1) = 2 per column.
        np.testing.assert_allclose(var, [2.0, 2.0])
        np.testing.assert_allclose(cov, [[2.0, 2.0], [2.0, 2.0]])

    def test_compute_statistics_single_step_is_zero(self):
        X = np.array([[1.0, 5.0, -2.0]])
        mean, var, cov = FlashTree.compute_statistics(X)
        np.testing.assert_allclose(mean, [1.0, 5.0, -2.0])
        np.testing.assert_allclose(var, np.zeros(3))
        np.testing.assert_allclose(cov, np.zeros((3, 3)))

    def test_compute_statistics_matches_numpy(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(37, 5))
        mean, var, cov = FlashTree.compute_statistics(X)
        np.testing.assert_allclose(mean, X.mean(axis=0))
        np.testing.assert_allclose(var, X.var(axis=0, ddof=1))
        np.testing.assert_allclose(cov, np.cov(X, rowvar=False))

    def test_neighbor_search_finds_local_minimum(self):
        tree = FlashTree(np.zeros((4, 2)), 4, 2)
        get_score = lambda x, extra: (abs(x - 3.0) + extra, np.zeros((2, 2)))  # noqa: E731
        best_x, best_score, best_sim = tree.neighbor_search(0, 10, 8, 5, get_score, 1.0)
        self.assertEqual(best_x, 3)
        self.assertAlmostEqual(best_score, 1.0)
        self.assertEqual(best_sim.shape, (2, 2))

    def test_neighbor_search_clamps_to_bounds(self):
        tree = FlashTree(np.zeros((4, 2)), 4, 2)
        # initial=2, max_range=5 -> clamped to min(2, 2)=2, so x in [0, 4].
        get_score = lambda x, extra: (float(x), np.zeros((2, 2)))  # noqa: E731
        best_x, best_score, _ = tree.neighbor_search(0, 4, 2, 5, get_score, 0.0)
        self.assertEqual(best_x, 0)
        self.assertAlmostEqual(best_score, 0.0)

    def test_neighbor_search_keeps_initial_on_plateau(self):
        tree = FlashTree(np.zeros((4, 2)), 4, 2)
        get_score = lambda x, extra: (0.0, np.zeros((2, 2)))  # noqa: E731
        best_x, best_score, _ = tree.neighbor_search(0, 4, 2, 3, get_score, 0.0)
        # Strict "<" keeps the first (initial) candidate on a plateau.
        self.assertEqual(best_x, 2)
        self.assertAlmostEqual(best_score, 0.0)


class TestFlashTreeOptimize(unittest.TestCase):
    """FlashTree.optimize_balanceness end-to-end (small problem)."""

    def _assert_valid_deployment(self, deployment, replicas, num_devices, num_replicas, num_experts):
        self.assertEqual(deployment.shape, (num_devices, num_replicas // num_devices))
        self.assertTrue((deployment >= 0).all(), "deployment must be fully filled")
        counts = np.bincount(deployment.ravel(), minlength=num_experts)
        np.testing.assert_array_equal(counts, replicas)
        self.assertTrue((replicas >= 1).all())

    def test_depth_one_allocation(self):
        rng = np.random.default_rng(3)
        X = rng.random((16, 4)) * 10 + 1
        num_replicas, num_devices = 8, 4
        tree = FlashTree(X, num_replicas, num_devices, z_score=0.0, depth=1)
        deployment, replicas, par = tree.optimize_balanceness()
        self._assert_valid_deployment(deployment, replicas, num_devices, num_replicas, 4)
        self.assertTrue(np.isfinite(par))
        self.assertGreaterEqual(par, 1.0 - 1e-6)

    def test_depth_two_allocation_improves_or_keeps_balance(self):
        rng = np.random.default_rng(4)
        X = rng.random((24, 6)) * 10 + 1
        num_replicas, num_devices = 12, 3
        tree = FlashTree(X, num_replicas, num_devices, z_score=0.0, depth=2, width=4)
        deployment, replicas, par = tree.optimize_balanceness()
        self._assert_valid_deployment(deployment, replicas, num_devices, num_replicas, 6)
        # The optimized score must not be worse than a naive fixed deployment with
        # the same replica counts (two copies of every expert).
        naive_replicas = np.full(6, 2, dtype=np.int32)
        naive_deployment = np.array(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [4, 5, 4, 5],
            ],
            dtype=np.int32,
        )
        naive_par = compute_score(X, naive_replicas, naive_deployment)
        self.assertLessEqual(par, naive_par + 1e-6)

    def test_deterministic_for_same_input(self):
        rng = np.random.default_rng(5)
        X = rng.random((20, 4)) * 8 + 1
        num_replicas, num_devices = 8, 2
        d1, r1, p1 = FlashTree(X, num_replicas, num_devices, z_score=0.5, depth=2, width=4).optimize_balanceness()
        d2, r2, p2 = FlashTree(X, num_replicas, num_devices, z_score=0.5, depth=2, width=4).optimize_balanceness()
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(r1, r2)
        self.assertEqual(p1, p2)


class TestFlashLBStatisticsUpdate(unittest.TestCase):
    """sliding_update_stats / incremental_update_stats against batch recomputation."""

    @staticmethod
    def _batch_stats(X):
        return FlashLB.compute_statistics(X)

    def test_sliding_update_matches_recomputation(self):
        rng = np.random.default_rng(6)
        T, N = 32, 4
        X = rng.normal(size=(T, N))
        mean, var, cov = self._batch_stats(X)
        # Remove the first 5 rows, append 5 new rows.
        x_old, x_new = X[:5], rng.normal(size=(5, N))
        new_mean, new_var, new_cov = FlashLB.sliding_update_stats(mean, cov, x_old, x_new, T)
        ref = np.concatenate([X[5:], x_new], axis=0)
        ref_mean, ref_var, ref_cov = self._batch_stats(ref)
        np.testing.assert_allclose(new_mean, ref_mean, atol=1e-10)
        np.testing.assert_allclose(new_var, ref_var, atol=1e-10)
        np.testing.assert_allclose(new_cov, ref_cov, atol=1e-10)

    def test_sliding_update_shape_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            FlashLB.sliding_update_stats(np.zeros(3), np.zeros((3, 3)), np.zeros((2, 3)), np.zeros((4, 3)), 10)

    def test_incremental_update_matches_recomputation(self):
        rng = np.random.default_rng(7)
        N = 4
        X_base = rng.normal(size=(10, N))
        mean, var, cov = self._batch_stats(X_base)
        x_new = rng.normal(size=(5, N))
        new_mean, new_var, new_cov, new_T = FlashLB.incremental_update_stats(mean, cov, x_new, 10)
        self.assertEqual(new_T, 15)
        ref_mean, ref_var, ref_cov = self._batch_stats(np.concatenate([X_base, x_new], axis=0))
        np.testing.assert_allclose(new_mean, ref_mean, atol=1e-10)
        np.testing.assert_allclose(new_var, ref_var, atol=1e-10)
        np.testing.assert_allclose(new_cov, ref_cov, atol=1e-10)

    def test_incremental_update_from_single_sample(self):
        # T == 1 branch: the window grows from one sample.
        base = np.array([[1.0, 2.0]])
        mean, var, cov = self._batch_stats(base)
        x_new = np.array([[3.0, 4.0], [5.0, 0.0]])
        new_mean, new_var, new_cov, new_T = FlashLB.incremental_update_stats(mean, cov, x_new, 1)
        self.assertEqual(new_T, 3)
        ref_mean, ref_var, ref_cov = self._batch_stats(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]]))
        np.testing.assert_allclose(new_mean, ref_mean, atol=1e-12)
        np.testing.assert_allclose(new_var, ref_var, atol=1e-12)
        np.testing.assert_allclose(new_cov, ref_cov, atol=1e-12)


class TestFlashLBHotnessWindow(unittest.TestCase):
    """register_hotness: hotness aggregation and sliding-window buffering."""

    def _make_policy(self, window: int) -> FlashLB:
        policy = FlashLB()
        policy.max_observation_window = window
        return policy

    def test_register_hotness_aggregates_by_placement(self):
        # Production pattern: window equals the number of incoming stages, so the
        # buffer is fully filled on the first call.
        policy = self._make_policy(4)
        deployment = np.array([[[0, 1], [2, 3]]], dtype=np.int32)  # 1 layer
        rank_load = np.zeros((4, 1, 2, 2), dtype=np.int32)
        rank_load[:, :, 0, :] = [10, 20]
        rank_load[:, :, 1, :] = [30, 40]
        policy.register_hotness(deployment, rank_load, num_layers=1, num_experts=4)
        info = policy.hotness_window[0]
        self.assertEqual(info["length"], 4)
        self.assertEqual(info["start"], 0)
        # hotness = rank load routed through the placement, plus the +1 floor.
        np.testing.assert_array_equal(info["buffer"][0], np.array([11.0, 21.0, 31.0, 41.0]))

    def test_register_hotness_multiple_stages_and_layers(self):
        policy = self._make_policy(2)
        deployment = np.array(
            [
                [[0, 1], [2, 3]],  # layer 0
                [[0, 2], [1, 3]],  # layer 1
            ],
            dtype=np.int32,
        )
        rank_load = np.zeros((2, 2, 2, 2), dtype=np.int32)
        rank_load[:, :, 0, 1] = 7  # device 0, slot 1 (both stages/layers)
        policy.register_hotness(deployment, rank_load, num_layers=2, num_experts=4)
        layer0 = policy.hotness_window[0]["buffer"]
        layer1 = policy.hotness_window[1]["buffer"]
        self.assertEqual(policy.hotness_window[0]["length"], 2)
        # layer0 placement [[0,1],[2,3]]: the hot slot routes to expert 1.
        np.testing.assert_array_equal(layer0[-1], np.array([1.0, 8.0, 1.0, 1.0]))
        # layer1 placement [[0,2],[1,3]]: the same hot slot now routes to expert 2.
        np.testing.assert_array_equal(layer1[-1], np.array([1.0, 1.0, 8.0, 1.0]))

    def test_register_hotness_wraps_window(self):
        # Partial-window calls: the buffer keeps the last two stages.
        policy = self._make_policy(2)
        deployment = np.array([[[0, 1]]], dtype=np.int32)
        for value in (1, 2, 3):
            rank_load = np.full((1, 1, 1, 2), value, dtype=np.int32)
            policy.register_hotness(deployment, rank_load, num_layers=1, num_experts=2)
        info = policy.hotness_window[0]
        self.assertEqual(info["length"], 2)
        self.assertEqual(info["start"], 1)
        # After 3 single-stage calls the buffer holds stages 3 and 2 (+1 floor),
        # with start pointing at the next write slot.
        np.testing.assert_array_equal(info["buffer"][0], np.array([4.0, 4.0]))
        np.testing.assert_array_equal(info["buffer"][1], np.array([3.0, 3.0]))

    def test_register_hotness_deduplicates_replicas(self):
        policy = self._make_policy(4)
        # Expert 0 has two replicas on the same device row: loads must be summed.
        deployment = np.array([[[0, 0], [1, 1]]], dtype=np.int32)
        rank_load = np.array([[[[1, 2], [3, 4]]]], dtype=np.int32)
        policy.register_hotness(deployment, rank_load, num_layers=1, num_experts=2)
        np.testing.assert_array_equal(policy.hotness_window[0]["buffer"][0], np.array([4.0, 8.0]))  # (1+2)+1, (3+4)+1


class TestFlashLBNeedUpdate(unittest.TestCase):
    """Update gating via average-to-peak ratio thresholds."""

    def _seed_state(self, policy: FlashLB, ratio_history: float, deployment, replicas, hotness_row):
        policy.average_to_peak_history[0] = ratio_history
        policy.current_deployment[0] = deployment
        policy.current_deployed_replicas[0] = replicas
        window = np.tile(hotness_row[None, :], (4, 1))
        policy.hotness_window[0] = {"buffer": window, "start": 0, "length": 4}

    def test_first_call_forces_update(self):
        policy = FlashLB()
        self.assertTrue(policy.need_update(0))

    def test_balanced_state_does_not_trigger_update(self):
        policy = FlashLB()
        deployment = np.array([[0, 1], [2, 3]], dtype=np.int32)
        replicas = np.array([1, 1, 1, 1], dtype=np.int32)
        # All rows identical -> per-device loads identical -> score 1.0 -> ratio 1.0.
        self._seed_state(policy, 0.95, deployment, replicas, np.array([5.0, 5.0, 5.0, 5.0]))
        self.assertFalse(policy.need_update(0))

    def test_imbalanced_state_triggers_update(self):
        policy = FlashLB()
        deployment = np.array([[0, 2], [1, 3]], dtype=np.int32)
        replicas = np.array([1, 1, 1, 1], dtype=np.int32)
        # device0 load 9+9=18, device1 load 1+1=2 -> ratio ~= 0.5556 < 0.85.
        self._seed_state(policy, 1.0, deployment, replicas, np.array([9.0, 1.0, 9.0, 1.0]))
        self.assertTrue(policy.need_update(0))

    def test_relative_drop_threshold(self):
        # Imbalanced state: ratio ~= 0.5556.
        deployment = np.array([[0, 2], [1, 3]], dtype=np.int32)
        replicas = np.array([1, 1, 1, 1], dtype=np.int32)
        hotness_row = np.array([9.0, 1.0, 9.0, 1.0])
        policy = FlashLB()
        self._seed_state(policy, 0.99, deployment, replicas, hotness_row)
        # 0.5556 < 0.99 * 0.9 = 0.891 -> relative rule fires.
        self.assertTrue(policy.need_update(0))
        # With past 0.6 the relative floor is 0.54 (not hit), but the absolute
        # floor 0.85 still fires: 0.5556 < 0.85.
        policy.average_to_peak_history[0] = 0.6
        self.assertTrue(policy.need_update(0))
        # A mildly unbalanced state (ratio ~= 0.9167) with a close past: neither
        # rule fires.
        policy2 = FlashLB()
        self._seed_state(policy2, 0.95, deployment, replicas, np.array([1.2, 1.0, 1.2, 1.0]))
        self.assertFalse(policy2.need_update(0))

    def test_need_update_uses_layer_specific_state(self):
        policy = FlashLB()
        # Layer 1 has no history -> forced update, independent of layer 0 state.
        policy.average_to_peak_history[0] = 1.0
        policy.hotness_window[0] = {"buffer": np.ones((4, 2)), "start": 0, "length": 4}
        policy.current_deployment[0] = np.array([[0, 1]], dtype=np.int32)
        policy.current_deployed_replicas[0] = np.array([1, 1], dtype=np.int32)
        self.assertFalse(policy.need_update(0))
        self.assertTrue(policy.need_update(1))


class TestFlashLBRedeployMinimization(unittest.TestCase):
    """compute_match and minimize_redeploy_with_inner_permutation."""

    def test_compute_match_identical_rows(self):
        src = np.array([[0, 1], [2, 3]], dtype=np.int32)
        src_counts = np.array([np.bincount(row, minlength=8) for row in src], dtype=np.int32)
        matches = FlashLB.compute_match(src_counts, src_counts, 2, 2)
        np.testing.assert_array_equal(matches[0, 0], 2)
        np.testing.assert_array_equal(matches[1, 1], 2)
        np.testing.assert_array_equal(matches[0, 1], 0)

    def test_compute_match_disjoint_rows(self):
        src = np.array([[0, 1]], dtype=np.int32)
        dst = np.array([[2, 3]], dtype=np.int32)
        src_counts = np.array([np.bincount(src[0], minlength=4)], dtype=np.int32)
        dst_counts = np.array([np.bincount(dst[0], minlength=4)], dtype=np.int32)
        matches = FlashLB.compute_match(src_counts, dst_counts, 1, 2)
        np.testing.assert_array_equal(matches, np.zeros((1, 1), dtype=np.int32))

    def test_minimize_redeploy_identical_matrices(self):
        src = np.array([[0, 1], [2, 3]], dtype=np.int32)
        result = FlashLB.minimize_redeploy_with_inner_permutation(src, src.copy())
        np.testing.assert_array_equal(result, src)

    def test_minimize_redeploy_row_permutation(self):
        src = np.array([[0, 1], [2, 3]], dtype=np.int32)
        dst = np.array([[2, 3], [0, 1]], dtype=np.int32)
        result = FlashLB.minimize_redeploy_with_inner_permutation(src, dst)
        np.testing.assert_array_equal(result, src)

    def test_minimize_redeploy_column_swap_stays_in_place(self):
        src = np.array([[0, 1], [2, 3]], dtype=np.int32)
        dst = np.array([[1, 0], [3, 2]], dtype=np.int32)
        # Experts keep their slots: reordering dst must not move any expert.
        result = FlashLB.minimize_redeploy_with_inner_permutation(src, dst)
        np.testing.assert_array_equal(result, src)

    def test_minimize_redeploy_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            FlashLB.minimize_redeploy_with_inner_permutation(
                np.zeros((2, 2), dtype=np.int32), np.zeros((2, 3), dtype=np.int32)
            )

    def test_minimize_redeploy_preserves_expert_multiset(self):
        # Row multisets: A={1,2,2}, B={3,3,5}, C={0,4,4}, D={1,6,7}.
        # dst shuffles both the row order and the intra-row positions.
        src = np.array(
            [
                [1, 2, 2],
                [3, 3, 5],
                [0, 4, 4],
                [1, 6, 7],
            ],
            dtype=np.int32,
        )
        dst = np.array(
            [
                [7, 6, 1],
                [4, 0, 4],
                [3, 5, 3],
                [2, 1, 2],
            ],
            dtype=np.int32,
        )
        result = FlashLB.minimize_redeploy_with_inner_permutation(src, dst)
        for i in range(4):
            self.assertEqual(sorted(result[i].tolist()), sorted(src[i].tolist()))


class TestFlashLBRebalanceExperts(unittest.TestCase):
    """End-to-end rebalance_experts with the dynamic EPLB group mocked."""

    def _mock_group(self, world_size: int):
        group = mock.Mock()
        group.world_size = world_size
        return mock.patch.object(policy_flashlb, "get_dynamic_eplb_group", return_value=group)

    def _assert_layer_valid(self, layer_deployment, num_devices, num_slots, num_experts):
        self.assertEqual(layer_deployment.shape, (num_devices, num_slots))
        self.assertTrue((layer_deployment >= 0).all())
        self.assertLessEqual(int(layer_deployment.max()), num_experts - 1)
        counts = np.bincount(layer_deployment.ravel(), minlength=num_experts)
        self.assertTrue((counts >= 1).all(), "every expert keeps at least one replica")

    def test_small_imbalanced_problem_improves(self):
        policy = FlashLB()
        # 2 layers x 2 devices x 4 slots; 4 experts, two replicas each.
        # Initial placement is unbalanced: device 0 holds both copies of
        # experts 0/1.
        current_table = torch.zeros((2, 2, 4), dtype=torch.int32)
        current_table[:, 0, :] = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        current_table[:, 1, :] = torch.tensor([2, 3, 2, 3], dtype=torch.int32)
        # Device 0 is hot on every slot, device 1 is cold; experts 0/1 live on
        # device 0, so the initial placement concentrates all load on device 0.
        workload = torch.zeros((4, 2, 2, 4), dtype=torch.int32)
        workload[:, :, 0, :] = 100
        with self._mock_group(4):
            change, priority, new_deployment = policy.rebalance_experts(current_table, workload)
        self.assertTrue(change)
        self.assertTrue(len(priority) > 0)
        self.assertTrue(set(priority.tolist()).issubset({0, 1}))
        for layer in range(2):
            self._assert_layer_valid(new_deployment[layer], 2, 4, 4)
            # Every expert must keep at least one replica.
            counts = np.bincount(new_deployment[layer].ravel(), minlength=4)
            self.assertTrue((counts >= 1).all())

    def test_small_balanced_problem_may_skip_update(self):
        policy = FlashLB()
        # Already-optimal placement: each device hosts one copy of every expert.
        current_table = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]], dtype=torch.int32)
        workload = torch.zeros((4, 1, 2, 4), dtype=torch.int32)
        workload[:, :, 0, 0] = 100
        workload[:, :, 1, 1] = 100
        with self._mock_group(4):
            change, priority, new_deployment = policy.rebalance_experts(current_table, workload)
        # Whether the optimizer finds an improvement or not, the output must stay
        # a valid 1-layer deployment.
        self._assert_layer_valid(new_deployment[0], 2, 4, 4)
        self.assertEqual(change, len(priority) > 0)

    def test_rebalance_is_deterministic(self):
        def run():
            policy = FlashLB()
            current_table = torch.zeros((2, 2, 4), dtype=torch.int32)
            current_table[:, 0, :] = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
            current_table[:, 1, :] = torch.tensor([2, 3, 2, 3], dtype=torch.int32)
            workload = torch.zeros((4, 2, 2, 4), dtype=torch.int32)
            workload[:, :, 0, :] = 100
            with self._mock_group(4):
                _, _, deployment = policy.rebalance_experts(current_table, workload)
            return deployment

        np.testing.assert_array_equal(run(), run())

    def test_rebalance_updates_internal_state_for_priority_layers(self):
        policy = FlashLB()
        current_table = torch.zeros((1, 2, 4), dtype=torch.int32)
        current_table[0, 0, :] = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        current_table[0, 1, :] = torch.tensor([2, 3, 2, 3], dtype=torch.int32)
        workload = torch.zeros((4, 1, 2, 4), dtype=torch.int32)
        workload[:, :, 0, :] = 50
        with self._mock_group(4):
            change, priority, new_deployment = policy.rebalance_experts(current_table, workload)
        if change:
            for layer in priority:
                np.testing.assert_array_equal(policy.current_deployment[layer], new_deployment[layer])
                self.assertIn(layer, policy.average_to_peak_history)

    def test_uninitialized_group_keeps_default_thresholds(self):
        policy = FlashLB()
        with mock.patch.object(policy_flashlb, "get_dynamic_eplb_group", side_effect=RuntimeError("not initialized")):
            change, _, _ = policy.rebalance_experts(
                torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]], dtype=torch.int32),
                torch.zeros((2, 1, 2, 4), dtype=torch.int32),
            )
        # Default thresholds (0.9 / 0.85) remain in effect.
        self.assertEqual(policy.update_threshold_ratio, 0.9)
        self.assertEqual(policy.update_threshold_value, 0.85)
        self.assertIsInstance(change, bool)

    def test_workload_3d_is_treated_as_single_stage(self):
        policy = FlashLB()
        current_table = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]], dtype=torch.int32)
        workload = torch.ones((1, 2, 4), dtype=torch.int32)  # (layers, devices, slots)
        with self._mock_group(4):
            change, priority, new_deployment = policy.rebalance_experts(current_table, workload)
        self.assertEqual(policy.max_observation_window, 1)
        self._assert_layer_valid(new_deployment[0], 2, 4, 4)
        self.assertEqual(change, len(priority) > 0)


class TestGenerateLayeredExperts(unittest.TestCase):
    """generate_layered_experts helper used by the EPLB test fixtures."""

    def test_small_custom_range(self):
        torch.manual_seed(42)
        layers = generate_layered_experts(num_layers=2, layer_shape=(8, 8), expert_min=0, expert_max=7)
        self.assertEqual(layers.shape, (2, 8, 8))
        self.assertEqual(layers.dtype, torch.int64)
        for layer in layers:
            # Every expert in [0, 8) appears at least once per layer.
            self.assertGreaterEqual(len(torch.unique(layer)), 8)
            for expert in range(8):
                self.assertTrue(torch.any(layer == expert))

    def test_default_shape_covers_all_256_experts(self):
        torch.manual_seed(42)
        layers = generate_layered_experts()
        self.assertEqual(layers.shape, (58, 32, 9))
        for layer in layers:
            self.assertEqual(len(torch.unique(layer)), 256)
            self.assertGreaterEqual(int(layer.min()), 0)
            self.assertLessEqual(int(layer.max()), 255)

    def test_layer_capacity_must_fit_experts(self):
        with self.assertRaises(AssertionError):
            generate_layered_experts(num_layers=1, layer_shape=(4, 4), expert_min=0, expert_max=19)


if __name__ == "__main__":
    unittest.main(verbosity=2)
