# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import sys
import unittest
# import torch_npu
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from ..omni_placement.cluster_status import ClusterStatus
from ..omni_placement.config import Config
from ..omni_placement.omni_planner import OmniPlanner


#torch.set_printoptions(threshold=float("inf"))  # Print all values
class TestOmniPlanner(unittest.TestCase):

    def setUp(self):
        self.planner = OmniPlanner("./tests/config_test.yaml", world_size=32)

    def tearDown(self):
        OmniPlanner.cleanup()
        if hasattr(self, 'planner'):
            del self.planner

    @unittest.skip("not for our strategy.")
    def test_omni_planner_plan(self):
        # Example input: 3 tokens, 4 experts each, with importance scores
        input_token = torch.tensor(
            [
                [0, 1, 2, 3],  # Token 1
                [1, 0, 3, 2],  # Token 2
                [3, 2, 1, 0]  # Token 3
            ],
            dtype=torch.float32).npu()

        input_expert_id = torch.tensor(
            [
                [0, 1, 2, 3],  # Token 1 expert
                [1, 0, 3, 2],  # Token 2 expert
                [3, 2, 1, 0]  # Token 3 expert
            ],
            dtype=torch.long).npu()

        input_expert_score = torch.tensor(
            [
                [0.4, 0.3, 0.2, 0.1],  # Token 1 expert score
                [0.4, 0.2, 0.2, 0.2],  # Token 2 expert score
                [0.7, 0.11, 0.1, 0.09]  # Token 3 expert score
            ],
            dtype=torch.float32).npu()

        token, token_expert_ids, token_scores = self.planner.plan(
            layer_idx_moe=0,
            tokens=input_token,
            token_expert_ids=input_expert_id,
            token_expert_scores=input_expert_score)

        print("Input mapping:")
        print(input_token, input_expert_id, input_expert_score)

        print("\nOptimized mapping:")
        print(token, token_expert_ids, token_scores)

        # graph plan only run redundant optimizer
        except_expert_ids = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2], [3, 2, 1, 0]],
            dtype=torch.long).npu()
        print(token_expert_ids)

        self.assertTrue(torch.equal(token_expert_ids, except_expert_ids))

    def test_get_deepseek_v3_moe_layer_idx(self):
        self.assertEqual(
            OmniPlanner.get_deepseek_v3_moe_layer_idx(
                "model.layers.5.mlp.experts"), 2)
        self.assertEqual(
            OmniPlanner.get_deepseek_v3_moe_layer_idx(
                "model.layers.10.mlp.experts"), 7)

    def test_initialization(self):
        self.assertIsInstance(self.planner.config, Config)
        self.assertIsNotNone(self.planner.expert_mapping)
        self.assertIsInstance(self.planner.cluster_status, ClusterStatus)
        self.assertEqual(len(self.planner.optimizers), 2)
        self.assertIn(
            "AdaRouter",
            [type(instance).__name__ for instance in self.planner.optimizers])

    def test_total_deployed_experts(self):
        expected_experts = 256
        np.testing.assert_array_equal(self.planner.total_deployed_experts,
                                      expected_experts)

    def test_is_expert_on_current_rank(self):
        exists, position = self.planner.is_expert_on_current_rank(0, 0, 0, 3)
        self.assertTrue(exists)
        self.assertEqual(position, 0)

        exists, position = self.planner.is_expert_on_current_rank(0, 100, 0, 3)
        self.assertFalse(exists)
        self.assertEqual(position, 64)

        exists, position = self.planner.is_expert_on_current_rank(0, 1, 1, 2)
        self.assertFalse(exists)
        self.assertEqual(position, 0)

        exists, position = self.planner.is_expert_on_current_rank(1, 3, 0, 2)
        self.assertTrue(exists)
        self.assertEqual(position, 3)

    def test_get_num_of_redundant_experts(self):
        num_of_redundant_experts = self.planner.get_num_of_redundant_experts(
            0, 63, 0)
        self.assertEqual(num_of_redundant_experts, 1)


@unittest.skip("not for our strategy.")
class TestOmniPlanner_with_none_redundant_optimizer(unittest.TestCase):

    def setUp(self):
        self.planner = OmniPlanner("./tests/config_none_redundant.yaml",
                                   world_size=32)

    def tearDown(self):
        OmniPlanner.cleanup()
        del self.planner

    def test_none_redundant_optimizer(self):
        # Example input: 3 tokens, 4 experts each, with importance scores
        input_token = torch.tensor(
            [
                [0, 1, 2, 3],  # Token 1
                [1, 0, 3, 2],  # Token 2
                [3, 2, 1, 0]  # Token 3
            ],
            dtype=torch.float32).npu()

        input_expert_id = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],  # Token 1 expert
                [4, 5, 6, 7, 8, 9, 10, 11]  # Token 2 expert
            ],
            dtype=torch.long).npu()  #.expand(10000, -1)

        input_expert_score = torch.tensor(
            [
                [0.4, 0.3, 0.2, 0.1],  # Token 1 expert score
                [0.4, 0.2, 0.2, 0.2],  # Token 2 expert score
                [0.7, 0.11, 0.1, 0.09]  # Token 3 expert score
            ],
            dtype=torch.float32).npu()

        # 测量 self.planner.plan 的运行时间
        # start_time = time.time()  # 记录开始时间

        print("Input mapping:")
        print(input_expert_id)

        token, token_expert_ids, token_scores = self.planner.plan(
            layer_idx_moe=42,
            tokens=input_token,
            token_expert_ids=input_expert_id,
            token_expert_scores=input_expert_score)
        # end_time = time.time()  # 记录结束时间
        # elapsed_time = end_time - start_time  # 计算时间差
        # print(f"self.planner.plan execution time: {elapsed_time:.6f} seconds")  # 打印运行时间

        print("\nOptimized mapping:")
        print(token_expert_ids)

        # graph plan only run redundant optimizer
        except_expert_ids = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 9, 10, 11, 12]],
            dtype=torch.long).npu()

        self.assertTrue(torch.equal(token_expert_ids, except_expert_ids))


class TestOmniPlanner_with_heat_optimizer(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_max_num_deployed_expert_per_rank_0(self):
        planner = OmniPlanner("./tests/config_heat.yaml",
                              world_size=64,
                              num_devices_per_host=16)
        self.assertEqual(planner.get_max_num_deployed_expert_per_rank(), 5)
        planner.cleanup()
        del planner

    def test_get_max_num_deployed_expert_per_rank_1(self):
        planner = OmniPlanner("./tests/config_heat.yaml",
                              rank=1,
                              world_size=64,
                              num_devices_per_host=16)
        self.assertEqual(planner.get_max_num_deployed_expert_per_rank(), 5)
        planner.cleanup()
        del planner


class TestEmptyPattern(unittest.TestCase):

    def setUp(self):
        # 创建一个空pattern的OmniPlanner实例用于测试
        self.planner = OmniPlanner("./tests/config_empty.yaml")

    def tearDown(self):
        OmniPlanner.cleanup()
        del self.planner

    @unittest.skip("singleton instance can not change config.")
    def test_empty_pattern(self):
        self.assertEqual(self.planner.placement_pattern, None)
        self.assertEqual(self.planner.is_expert_on_current_rank(0, 16, 1, 16),
                         (True, 0))


if __name__ == '__main__':
    unittest.main()
