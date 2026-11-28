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
"""Unit tests for the random EPLB policy and the policy factory."""

from unittest import TestCase, mock

import torch

from vllm_ascend.eplb.core.policy.policy_default_eplb import DefaultEplb
from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory
from vllm_ascend.eplb.core.policy.policy_flashlb import FlashLB
from vllm_ascend.eplb.core.policy.policy_random import RandomLoadBalance
from vllm_ascend.eplb.core.policy.policy_swift_balancer import SwiftBalanceEplb


class TestRandomLoadBalance(TestCase):
    def test_single_layer_swap_between_ranks_one_and_three(self):
        current = torch.tensor([[[0, 1], [2, 3], [4, 5], [6, 7]]], dtype=torch.int32)
        change, priority, new_table = RandomLoadBalance().rebalance_experts(current, None)
        self.assertEqual(change, 1)
        self.assertEqual(list(priority), [0])
        # Rank 3 receives the last value of rank 1 and vice versa.
        self.assertTrue(torch.equal(new_table[0][1], torch.tensor([2, 7])))
        self.assertTrue(torch.equal(new_table[0][3], torch.tensor([6, 3])))
        # Every other rank is untouched.
        self.assertTrue(torch.equal(new_table[0][0], current[0][0]))
        self.assertTrue(torch.equal(new_table[0][2], current[0][2]))
        # The input table is not mutated.
        self.assertTrue(torch.equal(current[0][1], torch.tensor([2, 3])))
        self.assertTrue(torch.equal(current[0][3], torch.tensor([6, 7])))

    def test_multi_layer_swaps_use_inverted_layer_priority(self):
        current = torch.tensor(
            [
                [[0, 1], [2, 3], [4, 5], [6, 7]],
                [[8, 9], [10, 11], [12, 13], [14, 15]],
            ],
            dtype=torch.int32,
        )
        change, priority, new_table = RandomLoadBalance().rebalance_experts(current, None)
        self.assertEqual(change, 1)
        # The per-layer priority is -layer_index (0, -1, ...).
        self.assertEqual(list(priority), [0, -1])
        self.assertTrue(torch.equal(new_table[0][3], torch.tensor([6, 3])))
        self.assertTrue(torch.equal(new_table[1][3], torch.tensor([14, 11])))
        self.assertTrue(torch.equal(new_table[1][1], torch.tensor([10, 15])))


class TestPolicyFactory(TestCase):
    def test_type_zero_maps_to_random_policy(self):
        self.assertIsInstance(PolicyFactory.generate_policy(0), RandomLoadBalance)

    def test_type_one_maps_to_default_eplb(self):
        self.assertIsInstance(PolicyFactory.generate_policy(1), DefaultEplb)

    def test_type_two_maps_to_swift_balance(self):
        with mock.patch("torch.npu.device_count", return_value=1):
            self.assertIsInstance(PolicyFactory.generate_policy(2), SwiftBalanceEplb)

    def test_type_three_maps_to_flashlb_and_warms_up(self):
        with (
            mock.patch("torch.npu.device_count", return_value=1),
            mock.patch("vllm_ascend.eplb.core.policy.policy_factory.warm_up") as warm_up,
        ):
            policy = PolicyFactory.generate_policy(3)
        self.assertIsInstance(policy, FlashLB)
        warm_up.assert_called_once()

    def test_unknown_type_falls_back_to_random_policy(self):
        self.assertIsInstance(PolicyFactory.generate_policy(99), RandomLoadBalance)
        self.assertIsInstance(PolicyFactory.generate_policy(-1), RandomLoadBalance)


if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
