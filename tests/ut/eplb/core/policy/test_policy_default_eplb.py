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
"""Unit tests for the default EPLB policy (``vllm_ascend.eplb.core.policy.policy_default_eplb``).

Covers workload aggregation, redundancy counting, the balanced pack (with and
without redundancy), the local-exchange constraint and the end-to-end
``rebalance_experts`` entry point, including its validation errors.
"""

from unittest import TestCase

import numpy as np

from vllm_ascend.eplb.core.policy.policy_default_eplb import DefaultEplb


class TestWorkloadAggregation(TestCase):
    def test_add_redundant_maps_slots_to_logical_experts(self):
        table = np.array([[[0, 0], [1, 1]]])
        workload = np.array([[[100, 0], [0, 0]]])
        result = DefaultEplb.add_redundant(table, workload, num_original_expert=2)
        np.testing.assert_array_equal(result, np.array([[100, 0]]))

    def test_add_redundant_sums_across_ranks(self):
        table = np.array([[[0, 1], [0, 2]]])
        workload = np.array([[[10, 20], [30, 40]]])
        result = DefaultEplb.add_redundant(table, workload, num_original_expert=3)
        np.testing.assert_array_equal(result, np.array([[40, 20, 40]]))

    def test_get_redundant_num(self):
        self.assertEqual(int(DefaultEplb.get_redundant_num(2, np.array([2, 2]))), 2)
        self.assertEqual(int(DefaultEplb.get_redundant_num(4, np.array([1, 1, 1, 1]))), 0)
        self.assertEqual(int(DefaultEplb.get_redundant_num(2, np.array([3, 1]))), 2)

    def test_calculate_max_heat_per_layer(self):
        workload = np.array([[[100, 0], [50, 1]], [[1, 1], [2, 3]]])
        heats = DefaultEplb.calculate_max_heat_per_layer(workload, layer_num=2)
        self.assertAlmostEqual(float(heats[0]), 100.0)
        self.assertAlmostEqual(float(heats[1]), 5.0)


class TestBalancedPack(TestCase):
    def test_no_redundancy_pack_splits_by_weight(self):
        weights = [(0, 10), (1, 2), (2, 2), (3, 6)]
        result, _boxes = DefaultEplb.compute_balanced_pack_redundancy(weights, card_num=2, num_redundancy_expert=0)
        self.assertEqual(result[0]["items"], [0, 1])
        self.assertAlmostEqual(result[0]["total_weight"], 12.0)
        self.assertEqual(result[1]["items"], [3, 2])
        self.assertAlmostEqual(result[1]["total_weight"], 8.0)

    def test_redundancy_card_num_zero_raises(self):
        with self.assertRaises(RuntimeError):
            DefaultEplb.compute_balanced_pack_redundancy([(0, 10)], card_num=0, num_redundancy_expert=0)

    def test_compute_balanced_pack_redundancy_dilutes_hottest_expert(self):
        # Expert 0 is hottest; both redundancy slots are assigned to it and
        # its weight is diluted to 9 / 3 = 3, so all four items weigh 3 and the
        # packer keeps the boxes weight-balanced at 6 each.
        weights = [(0, 9), (1, 3)]
        result, _boxes = DefaultEplb.compute_balanced_pack_redundancy(weights, card_num=2, num_redundancy_expert=2)
        self.assertEqual(sorted(result[0]["items"]), [0, 1])
        self.assertAlmostEqual(result[0]["total_weight"], 6.0)
        self.assertEqual(result[1]["items"], [0, 0])
        self.assertAlmostEqual(result[1]["total_weight"], 6.0)

    def test_compute_balanced_pack_redundancy_single_expert(self):
        result, _boxes = DefaultEplb.compute_balanced_pack_redundancy([(0, 9)], card_num=2, num_redundancy_expert=1)
        self.assertEqual(result[0]["items"], [0])
        self.assertAlmostEqual(result[0]["total_weight"], 4.5)
        self.assertEqual(result[1]["items"], [0])
        self.assertAlmostEqual(result[1]["total_weight"], 4.5)

    def test_redundant_variants_agree_on_safe_inputs(self):
        weights = [(0, 9), (1, 3)]
        original, original_boxes = DefaultEplb.original_compute_balanced_pack_redundancy(
            weights, card_num=2, num_redundancy_expert=2
        )
        fixed, fixed_boxes = DefaultEplb.compute_balanced_pack_redundancy(weights, card_num=2, num_redundancy_expert=2)
        self.assertEqual(original_boxes, fixed_boxes)
        for original_box, fixed_box in zip(original, fixed):
            self.assertEqual(original_box["items"], fixed_box["items"])
            self.assertAlmostEqual(original_box["total_weight"], fixed_box["total_weight"])


class TestConstraintAndRebalance(TestCase):
    def test_constraint_expert_local_exchange_keeps_positions(self):
        old = np.array([[[0, 1], [2, 3]]])
        new = [[[1, 2], [3, 0]]]
        result = DefaultEplb.constraint_expert_local_exchange(old, new)
        # Expert 1 keeps column 1 of card 0 and expert 3 keeps column 1 of
        # card 1; the newly moved experts fill the freed columns.
        self.assertEqual(result, [[[2, 1], [0, 3]]])

    def test_rebalance_hot_layer_moves_experts(self):
        policy = DefaultEplb()
        placement = np.array([[[0, 1], [2, 3]]])
        # NPU 0 holds both hot experts; NPU 1 is idle.
        workload = np.array([[[100, 100], [0, 0]]])
        change, priority, new_deployment = policy.rebalance_experts(placement, workload)
        self.assertEqual(change, 1)
        self.assertEqual(list(np.asarray(priority)), [0])
        # Packing puts (expert 1, expert 3) on card 0 and (expert 0, expert 2)
        # on card 1; the local-exchange constraint then pins expert 1 into
        # column 1 of card 0 and expert 2 into column 0 of card 1.
        self.assertEqual(new_deployment, [[[3, 1], [2, 0]]])

    def test_rebalance_equal_loads_keeps_update_disabled(self):
        policy = DefaultEplb()
        placement = np.array([[[0, 1], [2, 3]]])
        workload = np.array([[[50, 50], [50, 50]]])
        change, priority, new_deployment = policy.rebalance_experts(placement, workload)
        # Every expert carries load 100, so the packed boxes each total 200
        # against the previous per-NPU heat of 100: the update is computed but
        # not reported as a change.
        self.assertEqual(change, 0)
        self.assertEqual(list(np.asarray(priority)), [0])
        self.assertEqual(new_deployment, [[[3, 1], [2, 0]]])

    def test_rebalance_priority_orders_layers(self):
        policy = DefaultEplb()
        placement = np.array([[[0, 1], [2, 3]], [[0, 1], [2, 3]]])
        workload = np.array([[[100, 100], [0, 0]], [[50, 50], [50, 50]]])
        change, priority, new_deployment = policy.rebalance_experts(placement, workload)
        # Layer 0 improves (ratio 0.5) before layer 1 (ratio 1.0), so the
        # priority order is [0, 1]; the combined after heat 200 beats the
        # 0.95 * 300 threshold, so the update is reported.
        self.assertEqual(change, 1)
        self.assertEqual(list(np.asarray(priority)), [0, 1])
        self.assertEqual(new_deployment, [[[3, 1], [2, 0]], [[3, 1], [2, 0]]])

    def test_rebalance_raises_when_redundancy_exceeds_ranks(self):
        policy = DefaultEplb()
        # One expert occupies all slots -> 3 redundancies > 2 ranks.
        placement = np.array([[[0, 0], [0, 0]]])
        workload = np.array([[[100, 0], [0, 0]]])
        with self.assertRaises(ValueError):
            policy.rebalance_experts(placement, workload)


if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
