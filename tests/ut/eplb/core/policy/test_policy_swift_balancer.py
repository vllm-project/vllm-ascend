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
"""Unit tests for the SwiftBalance EPLB policy (``vllm_ascend.eplb.core.policy.policy_swift_balancer``).

Covers per-layer heat aggregation, original-workload aggregation, the
local-exchange constraint, imbalance scoring, redundant-expert statistics and
assignment, the swap search (single swap and the exchange loop), and the
end-to-end ``rebalance_experts`` entry point including the multi-node path.
"""

from collections import defaultdict
from unittest import TestCase, mock

import numpy as np
import torch

from vllm_ascend.eplb.core.policy.policy_swift_balancer import SwiftBalanceEplb


def make_policy(**overrides) -> SwiftBalanceEplb:
    """Build a policy without depending on the number of visible NPU devices."""
    with mock.patch("torch.npu.device_count", return_value=1):
        policy = SwiftBalanceEplb()
    for key, value in overrides.items():
        setattr(policy, key, value)
    return policy


class TestHeatAndWorkloadAggregation(TestCase):
    def test_calculate_max_heat_per_layer(self):
        workload = np.array([[[100, 0], [50, 1]], [[1, 1], [2, 3]]])
        # layer0 NPU heats: [100, 51] -> max 100; layer1: [2, 5] -> max 5.
        heats = SwiftBalanceEplb.calculate_max_heat_per_layer(workload)
        self.assertEqual(len(heats), 2)
        self.assertAlmostEqual(float(heats[0]), 100.0)
        self.assertAlmostEqual(float(heats[1]), 5.0)

    def test_get_original_workload_aggregates_by_placement(self):
        table = np.array([[[0, 0], [1, 1]]])
        workload = np.array([[[100, 0], [0, 0]]])
        original = SwiftBalanceEplb.get_original_workload(table, workload, num_original_expert=2)
        np.testing.assert_array_equal(original, np.array([[100, 0]]))

    def test_get_original_workload_sums_duplicate_placement(self):
        table = np.array([[[0, 1], [0, 2]]])
        workload = np.array([[[10, 20], [30, 40]]])
        original = SwiftBalanceEplb.get_original_workload(table, workload, num_original_expert=3)
        # expert 0 is placed twice (10 + 30).
        np.testing.assert_array_equal(original, np.array([[40, 20, 40]]))

    def test_constraint_expert_local_exchange_keeps_positions(self):
        old = np.array([[[0, 1], [2, 3]]])
        new = np.array([[[1, 2], [3, 0]]])
        # The alignment is applied in place.
        SwiftBalanceEplb.constraint_expert_local_exchange(old, new)
        # Expert 1 keeps column 1 of rank 0; the new expert 2 fills column 0.
        # Expert 3 keeps column 1 of rank 1; the new expert 0 fills column 0.
        np.testing.assert_array_equal(new, np.array([[[2, 1], [0, 3]]]))

    def test_constraint_expert_local_exchange_no_op_on_identical(self):
        old = np.array([[[0, 1], [2, 3]]])
        new = old.copy()
        SwiftBalanceEplb.constraint_expert_local_exchange(old, new)
        np.testing.assert_array_equal(new, old)


class TestImbalanceAndStatistics(TestCase):
    def _imbalance_policy(self, num_original_experts=4):
        policy = make_policy()
        policy.num_ranks = 2
        policy.num_original_experts = num_original_experts
        policy.num_experts_per_rank = 2
        return policy

    def test_calculate_imbalance(self):
        policy = self._imbalance_policy()
        deployment = np.array([[[0, 1], [2, 3]]])
        load = np.array([[10.0, 10.0, 0.0, 0.0]])
        imbalance = policy.calculate_imbalance(deployment, load)
        # rank0: 10 + 10 = 20, rank1: 0 -> max/avg = 20/10 = 2.0.
        self.assertEqual(len(imbalance), 1)
        self.assertAlmostEqual(imbalance[0], 2.0, places=6)

    def test_calculate_imbalance_zero_load(self):
        policy = self._imbalance_policy()
        deployment = np.array([[[0, 1], [2, 3]]])
        load = np.zeros((1, 4))
        self.assertAlmostEqual(policy.calculate_imbalance(deployment, load)[0], 1.0)

    def test_calculate_imbalance_splits_replica_load(self):
        policy = self._imbalance_policy()
        # Expert 0 is replicated on both ranks: its load is divided by 2.
        deployment = np.array([[[0, 1], [0, 3]]])
        load = np.array([[100.0, 0.0, 100.0, 0.0]])
        imbalance = policy.calculate_imbalance(deployment, load)
        # rank0: 100/2 + 0 + 100/2 = 100, rank1: same -> 1.0.
        self.assertAlmostEqual(imbalance[0], 1.0, places=6)

    def test_statistics_expert_distribution_no_redundancy(self):
        policy = self._imbalance_policy()
        deployment = np.array([[0, 1], [2, 3]])
        redundant_pos, from_rank, existing, count = policy.statistics_expert_distribution(deployment)
        # Column-major scan: column 0 first-seen (expert 0 on rank 0, expert
        # 2 on rank 1), column 1 (expert 1 on rank 0, expert 3 on rank 1).
        np.testing.assert_array_equal(from_rank, np.array([0, 0, 1, 1]))
        self.assertEqual(count, 0)
        self.assertEqual(existing, {0, 1, 2, 3})
        self.assertEqual(len(redundant_pos), 2)
        self.assertEqual(sum(len(row) for row in redundant_pos), 0)

    def test_statistics_expert_distribution_with_redundancy(self):
        policy = make_policy()
        policy.num_original_experts = 2
        policy.num_experts_per_rank = 2
        deployment = np.array([[0, 1], [0, 1]])
        redundant_pos, from_rank, existing, count = policy.statistics_expert_distribution(deployment)
        # Both experts are first seen on rank 0; rank 1 holds the redundant
        # copies at column positions 0 and 1.
        np.testing.assert_array_equal(from_rank, np.array([0, 0]))
        self.assertEqual(count, 2)
        self.assertEqual(existing, {0, 1})
        self.assertEqual([list(row) for row in redundant_pos], [[], [0, 1]])

    def test_statistics_expert_distribution_single_redundancy(self):
        policy = self._imbalance_policy()
        deployment = np.array([[0, 1], [0, 3]])
        redundant_pos, from_rank, existing, count = policy.statistics_expert_distribution(deployment)
        # Expert 2 is absent, so its from_rank slot keeps the zero init value.
        np.testing.assert_array_equal(from_rank, np.array([0, 0, 0, 1]))
        self.assertEqual(count, 1)
        self.assertEqual(existing, {0, 1, 3})
        self.assertEqual([list(row) for row in redundant_pos], [[], [0]])


class TestRedundantAssignment(TestCase):
    def _policy(self):
        policy = make_policy()
        policy.num_max_com = 1
        policy.num_original_experts = 2
        policy.num_experts_per_rank = 2
        return policy

    def test_compute_redundant_assignments_hottest_first_with_cap(self):
        policy = make_policy()
        policy.num_original_experts = 4
        redundant_list, update_weights = policy.compute_redundant_assignments(
            [(0, 100), (1, 100), (2, 1), (3, 1)], num_redundant_experts=2, num_ranks=2
        )
        # Round 1 picks expert 1 (tie broken by the reversed stable sort),
        # round 2 then picks expert 0; both weights are diluted to 100*2/3
        # (float32 storage, so compare at low precision).
        self.assertEqual(len(redundant_list), 2)
        self.assertEqual(sorted(expert for expert, _ in redundant_list), [0, 1])
        for _, weight in redundant_list:
            self.assertAlmostEqual(weight, 100.0 * 2 / 3, places=4)
        self.assertAlmostEqual(float(update_weights[0]), 100.0 * 2 / 3, places=4)
        self.assertAlmostEqual(float(update_weights[1]), 100.0 * 2 / 3, places=4)
        self.assertAlmostEqual(float(update_weights[2]), 1.0)

    def test_compute_redundant_assignments_num_ranks_cap(self):
        policy = make_policy()
        policy.num_original_experts = 1
        # A single expert cannot hold more than num_ranks copies in total.
        redundant_list, update_weights = policy.compute_redundant_assignments(
            [(0, 100)], num_redundant_experts=3, num_ranks=1
        )
        self.assertEqual(redundant_list, [])
        self.assertAlmostEqual(float(update_weights[0]), 100.0)

    def test_non_redundant_expert_information(self):
        policy = self._policy()
        deployment = np.array([[0, 0], [1, 1]])
        assignments, loads = policy.non_redundant_expert_information(deployment, np.array([100.0, 0.0]), [[1], [1]])
        np.testing.assert_array_equal(assignments, np.array([[0, -1], [1, -1]]))
        np.testing.assert_allclose(loads, [100.0, 0.0])

    def test_recomputing_initial_weight(self):
        policy = self._policy()
        update_weights, counts = policy.recomputing_initial_weight([(0, 100.0), (1, 0.0)], np.array([[0, -1], [1, -1]]))
        np.testing.assert_allclose(update_weights, [100.0, 0.0])
        np.testing.assert_array_equal(counts, [1, 1])

    def test_recomputing_initial_weight_missing_expert_asserts(self):
        policy = self._policy()
        with self.assertRaises(AssertionError):
            policy.recomputing_initial_weight([(0, 100.0), (1, 0.0)], np.array([[-1, -1], [0, -1]]))

    def test_distribute_redundant_experts_picks_least_loaded_rank(self):
        policy = self._policy()
        rank_assignments = np.array([[0, -1], [1, -1]])
        rank_loads = np.array([50.0, 0.0])
        # Both redundant copies are placed: expert 0 goes to the least loaded
        # rank 1, expert 1 then goes to the only remaining redundant slot.
        com, rev, undeployed = policy.distribute_redundant_experts(
            rank_assignments, rank_loads, [(0, 50.0), (1, 0.0)], np.array([0, 1]), [[1], [1]]
        )
        np.testing.assert_array_equal(rank_assignments, np.array([[0, 1], [1, 0]]))
        np.testing.assert_allclose(rank_loads, [50.0, 50.0])
        self.assertEqual(com[0][1], 1)
        self.assertEqual(com[1][0], 1)
        self.assertEqual(rev[1], {0})
        self.assertEqual(rev[0], {1})
        self.assertEqual(undeployed, [])


class TestSwapSearch(TestCase):
    def _policy(self):
        policy = make_policy()
        policy.num_max_com = 1
        policy.swap_threshold = 0.0
        policy.max_swap_times = 100
        return policy

    def test_swap_experts_between_ranks_finds_improving_pair(self):
        # max rank: experts 0/1 (loads 10 + 2 = 12); swap rank: 2/3 (4 + 0 = 4).
        policy = make_policy()
        expert, partner, best = policy.swap_experts_between_ranks(
            {0, 1}, {2, 3}, set(), set(), np.array([10.0, 2.0, 4.0, 0.0]), 12.0, 4.0
        )
        self.assertEqual(expert, 0)
        self.assertEqual(partner, 2)
        self.assertAlmostEqual(best, 10.0)

    def test_swap_experts_between_ranks_no_improvement(self):
        policy = make_policy()
        expert, partner, best = policy.swap_experts_between_ranks(
            {0, 1}, {2, 3}, set(), set(), np.array([5.0, 5.0, 5.0, 5.0]), 10.0, 10.0
        )
        self.assertEqual(expert, -1)
        self.assertEqual(partner, -1)
        self.assertAlmostEqual(best, 10.0)

    def test_expert_exchange_between_ranks_single_swapped_round(self):
        policy = self._policy()
        rank_assignments = np.array([[0, 1], [2, 3]])
        rank_loads = np.array([12.0, 4.0])
        com_per_rank = np.zeros((2, 2), dtype=np.int64)
        rev_expert_per_rank: dict = defaultdict(set)
        updated_weights = np.array([10.0, 2.0, 4.0, 0.0])
        rows, max_rank_load = policy.expert_exchange_between_ranks(
            rank_assignments, rank_loads, com_per_rank, rev_expert_per_rank, updated_weights
        )
        # Round 1: rank 0 (load 12) gives expert 0 to rank 1 in exchange for
        # expert 2 (loads 12/4 -> 10/6). Round 2: both ranks have exchanged
        # num_max_com times already, so no further swap is allowed.
        self.assertAlmostEqual(max_rank_load, 10.0)
        self.assertEqual(sorted(rows[0]), [1, 2])
        self.assertEqual(sorted(rows[1]), [0, 3])
        self.assertEqual(com_per_rank[0][1], 1)
        self.assertEqual(com_per_rank[1][0], 1)

    def test_expert_exchange_between_ranks_zero_swap_budget(self):
        policy = self._policy()
        policy.max_swap_times = 0
        rank_assignments = np.array([[0, 1], [2, 3]])
        rows, max_rank_load = policy.expert_exchange_between_ranks(
            rank_assignments,
            np.array([12.0, 4.0]),
            np.zeros((2, 2), dtype=np.int64),
            defaultdict(set),
            np.array([10.0, 2.0, 4.0, 0.0]),
        )
        # With no swap budget the loop body never runs.
        self.assertEqual(max_rank_load, 0.0)
        self.assertEqual(sorted(rows[0]), [0, 1])
        self.assertEqual(sorted(rows[1]), [2, 3])


class TestRebalanceExperts(TestCase):
    def _assert_valid_rows(self, rows, num_ranks, num_slots):
        self.assertEqual(len(rows), num_ranks)
        for row in rows:
            self.assertEqual(len(row), num_slots)
            self.assertTrue(all(v >= 0 for v in row))

    def test_imbalanced_placement_is_improved(self):
        policy = make_policy(num_die_per_host=1)
        placement = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32)
        # NPU 0 carries both hot experts on hot slots; NPU 1 is idle.
        workload = torch.tensor([[[100, 100], [0, 0]]], dtype=torch.int32)
        change, priority, new_deployment = policy.rebalance_experts(placement, workload)
        self.assertEqual(change, 1)
        self.assertEqual(list(np.asarray(priority)), [0])
        self._assert_valid_rows(new_deployment[0], 2, 2)
        # Both ranks end up hosting one hot and one cold expert; the
        # local-exchange constraint pins the experts that did not move.
        self.assertEqual(new_deployment[0][0], [2, 1])
        self.assertEqual(new_deployment[0][1], [0, 3])

    def test_balanced_placement_is_unchanged(self):
        policy = make_policy(num_die_per_host=1)
        placement = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32)
        workload = torch.tensor([[[50, 50], [50, 50]]], dtype=torch.int32)
        change, priority, new_deployment = policy.rebalance_experts(placement, workload)
        self.assertEqual(change, 0)
        self.assertEqual(list(np.asarray(priority)), [0])
        self.assertEqual(new_deployment, [[[0, 1], [2, 3]]])

    def test_priority_orders_layers_by_improvement(self):
        policy = make_policy(num_die_per_host=1)
        placement = torch.tensor([[[0, 1], [2, 3]], [[0, 1], [2, 3]]], dtype=torch.int32)
        workload = torch.tensor([[[100, 100], [0, 0]], [[1, 1], [1, 1]]], dtype=torch.int32)
        change, priority, new_deployment = policy.rebalance_experts(placement, workload)
        self.assertEqual(change, 1)
        # Layer 0 improves to ratio 0.5; layer 1 stays at 1.0 -> [0, 1].
        self.assertEqual(list(np.asarray(priority)), [0, 1])
        self._assert_valid_rows(new_deployment[0], 2, 2)
        self.assertEqual(new_deployment[1], [[0, 1], [2, 3]])

    def test_multi_node_mode_preserves_node_expert_multisets(self):
        policy = make_policy(num_die_per_host=2)
        # 4 ranks -> 2 nodes of 2 ranks; only rank 0 is hot.
        placement = torch.tensor([[[0, 1], [2, 3], [0, 2], [1, 3]]], dtype=torch.int32)
        workload = torch.tensor([[[100, 0], [0, 0], [0, 0], [0, 0]]], dtype=torch.int32)
        change, priority, new_deployment = policy.rebalance_experts(placement, workload, is_node_redundant=True)
        self.assertEqual(policy.num_nodes, 2)
        self._assert_valid_rows(new_deployment[0], 4, 2)
        # No cross-node movement: each node keeps its expert multiset.
        self.assertEqual(sorted(sum([sorted(row) for row in new_deployment[0][:2]], [])), [0, 1, 2, 3])
        self.assertEqual(sorted(sum([sorted(row) for row in new_deployment[0][2:]], [])), [0, 1, 2, 3])
        self.assertEqual(change, 0)
        self.assertEqual(new_deployment, [[[0, 1], [2, 3], [0, 2], [1, 3]]])

    def test_imbalanced_single_expert_placement(self):
        # Expert 0 (load 100) is duplicated on the hot rank. The redundant
        # copy is redistributed to the idle rank with a diluted weight
        # (100 * 2 / 3), so the per-layer max heat drops from 100 to ~66.7
        # and the update is adopted: each rank ends up with one copy of each
        # expert.
        policy = make_policy(num_die_per_host=1)
        placement = torch.tensor([[[0, 0], [1, 1]]], dtype=torch.int32)
        workload = torch.tensor([[[100, 0], [0, 0]]], dtype=torch.int32)
        change, priority, new_deployment = policy.rebalance_experts(placement, workload)
        self.assertEqual(change, 1)
        self.assertEqual(list(np.asarray(priority)), [0])
        self.assertEqual(new_deployment, [[[0, 1], [1, 0]]])


if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
