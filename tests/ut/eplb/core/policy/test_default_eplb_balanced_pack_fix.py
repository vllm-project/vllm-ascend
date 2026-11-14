#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""CPU-only regression tests for the DefaultEplb balanced-packing functions.

Covers the two hardening fixes in ``policy_default_eplb.py``:

* ``compute_balanced_pack`` previously indexed a Python list with a numpy
  array of indices, which raises ``TypeError`` under numpy >= 2 (the
  project does not pin the numpy major version).
* ``original_compute_balanced_pack_redundancy`` packed every redundant
  expert into a distinct card and raised a bare ``IndexError`` when the
  number of redundant experts exceeded the number of cards; it now
  validates up front with a descriptive error, and rejects ``card_num=0``
  the same way its guarded sibling does.
"""

import unittest
from unittest.mock import patch

import numpy as np

from vllm_ascend.eplb.core.policy.policy_default_eplb import DefaultEplb


class TestComputeBalancedPack(unittest.TestCase):
    """The non-redundant packing helper that crashed on numpy >= 2."""

    def test_sorts_descending_and_balances_two_cards(self):
        weights = [(0, 100), (1, 50), (2, 30), (3, 10)]
        result, boxes = DefaultEplb.compute_balanced_pack(weights, 2)
        # Greedy fill of the lightest non-full box: card0 takes the hottest
        # expert plus the last (lightest) expert, card1 takes the middle two.
        self.assertEqual(boxes, [[0, 3], [1, 2]])
        self.assertEqual(result[0]["box_index"], 1)
        self.assertEqual(result[0]["total_weight"], 110)
        self.assertEqual(result[1]["total_weight"], 80)
        self.assertEqual(result[1]["item_count"], 2)
        self.assertEqual(result[1]["weight"], [50, 30])

    def test_odd_split_distributes_remainder(self):
        weights = [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]
        result, boxes = DefaultEplb.compute_balanced_pack(weights, 2)
        # Five items over two cards: one card takes three, the other two.
        self.assertEqual(sorted(len(box) for box in boxes), [2, 3])
        self.assertEqual(sum(r["total_weight"] for r in result), 25)

    def test_single_card_keeps_all_items_descending(self):
        weights = [(0, 10), (1, 20), (2, 5)]
        result, boxes = DefaultEplb.compute_balanced_pack(weights, 1)
        self.assertEqual(boxes, [[1, 0, 2]])
        self.assertEqual(result[0]["total_weight"], 35)

    def test_equal_weights_reversed_by_stable_descending_sort(self):
        # argsort(kind="stable")[::-1] reverses the tie order, which the
        # algorithm relies on; pin the current (quirky but deterministic)
        # behavior so a future change is explicit.
        result, boxes = DefaultEplb.compute_balanced_pack([(0, 7), (1, 7)], 2)
        self.assertEqual(boxes, [[1], [0]])

    def test_card_num_zero_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            DefaultEplb.compute_balanced_pack([(0, 1)], 0)
        self.assertIn("card_num can not be 0", str(ctx.exception))

    def test_numpy2_indexing_regression(self):
        # This exact call raised
        # "TypeError: only integer scalar arrays can be converted to a
        # scalar index" before the list-comprehension fix, because
        # `origin_weights[sorted_indices]` indexed a Python list with a
        # numpy array.
        self.assertGreaterEqual(np.__version__.split(".")[0], "1")
        weights = [(0, 3), (1, 1), (2, 2), (3, 9), (4, 4)]
        result, boxes = DefaultEplb.compute_balanced_pack(weights, 2)
        self.assertEqual(sum(r["total_weight"] for r in result), 19)
        self.assertEqual(len(boxes), 2)


class TestOriginalRedundancyPacking(unittest.TestCase):
    def test_more_redundancy_than_cards_raises_value_error(self):
        # Regression: this used to die deep in step 3 with a bare
        # "IndexError: list index out of range".
        with self.assertRaises(ValueError) as ctx:
            DefaultEplb.original_compute_balanced_pack_redundancy([(0, 100)], 2, 3)
        self.assertIn("must not exceed", str(ctx.exception))

        with self.assertRaises(ValueError):
            DefaultEplb.original_compute_balanced_pack_redundancy([(0, 100), (1, 50)], 2, 3)

    def test_card_num_zero_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            DefaultEplb.original_compute_balanced_pack_redundancy([(0, 100)], 0, 0)
        self.assertIn("card_num can not be 0", str(ctx.exception))

    def test_equal_boundary_still_works(self):
        # num_redundancy_expert == card_num is the largest valid case.
        result, boxes = DefaultEplb.original_compute_balanced_pack_redundancy([(0, 9)], 1, 1)
        self.assertEqual(sum(len(box) for box in boxes), 2)
        # Original weight 9 is split 4.5 / 4.5 between the two copies.
        self.assertEqual(result[0]["total_weight"], 9.0)

    def test_two_experts_two_cards_two_redundancy(self):
        # Both redundancies land on the hottest expert (0). The original
        # expert 0 shares a card with expert 1 in the result.
        result, boxes = DefaultEplb.original_compute_balanced_pack_redundancy([(0, 9), (1, 3)], 2, 2)
        self.assertEqual([sorted(box) for box in boxes], [[0, 1], [0, 0]])
        self.assertEqual(result[0]["total_weight"], 6.0)
        self.assertEqual(result[0]["total_weight"], result[1]["total_weight"])

    def test_single_expert_multi_redundancy_single_card(self):
        # All copies of one expert must fit the single card.
        result, boxes = DefaultEplb.original_compute_balanced_pack_redundancy([(0, 100)], 1, 1)
        self.assertEqual(boxes, [[0, 0]])
        self.assertEqual(result[0]["item_count"], 2)


class TestRebalanceProductionPath(unittest.TestCase):
    def test_redundancy_exceeding_npuses_is_rejected_up_front(self):
        # The production entry point keeps its own validation so the old
        # failure stays a clear error end to end. With 2 NPUs and a placement
        # in which one expert occupies every slot, the derived redundancy
        # count exceeds the NPU count and must be rejected.
        current = [[[0, 0, 0, 0], [0, 0, 0, 0]]]
        workloads = [[[1, 2, 3, 4], [5, 6, 7, 8]]]
        with patch("torch.npu.device_count", return_value=1):
            policy = DefaultEplb()
            with self.assertRaises(ValueError) as ctx:
                policy.rebalance_experts(current, workloads)
        self.assertIn("greater than or equal to the number of redundant experts", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
