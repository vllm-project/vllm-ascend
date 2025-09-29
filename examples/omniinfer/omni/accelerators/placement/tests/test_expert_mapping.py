# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import unittest
from unittest.mock import patch
import numpy as np
import torch
from pathlib import Path
import tempfile
from ..omni_placement import omni_placement
from ..omni_placement.expert_mapping import ExpertMapping

class TestExpertMapping(unittest.TestCase):
    def setUp(self):
        # 创建临时文件用于测试 placement pattern
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pattern_file = os.path.join(self.temp_dir.name, "pattern.npy")

        # 创建一个简单的 pattern: [2 devices, 2 layers, 4 experts]
        pattern_data = np.array([
            [[1, 0, 1, 0], [0, 1, 0, 1]],  # Device 0
            [[0, 1, 0, 1], [1, 0, 1, 0]]   # Device 1
        ], dtype=np.int32)
        np.save(self.pattern_file, pattern_data)

        # 初始化 ExpertMapping 实例
        self.expert_mapping = ExpertMapping(
            pattern_path=self.pattern_file,
            device="npu",
            num_devices_per_host=2
        )

    def tearDown(self):
        # 清理临时目录
        self.temp_dir.cleanup()

    def test_init(self):
        # 测试初始化
        self.assertEqual(self.expert_mapping.pattern_path, self.pattern_file)
        self.assertEqual(self.expert_mapping.device, "npu")
        self.assertEqual(self.expert_mapping.num_devices_per_host, 2)
        self.assertIsNotNone(self.expert_mapping.placement_pattern)
        self.assertEqual(self.expert_mapping.placement_pattern.shape, (2, 2, 4))

    def test_resolve_pattern_path(self):
        # 测试路径解析
        relative_path = "pattern.npy"
        resolved_path = self.expert_mapping._resolve_pattern_path()
        self.assertTrue(os.path.isabs(resolved_path))

        # 测试空路径
        em = ExpertMapping(pattern_path="", device="cpu")
        self.assertIsNone(em._resolve_pattern_path())

    def test_load_placement_pattern_with_validation(self):
        # 测试加载和验证
        pattern = self.expert_mapping._load_placement_pattern_with_validation()
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.shape, (2, 2, 4))
        self.assertEqual(pattern.dtype, torch.int32)

        # 测试无效路径，期望抛出 FileNotFoundError
        with self.assertRaises(FileNotFoundError) as context:
            em = ExpertMapping(pattern_path="nonexistent.npy", device="cpu")
        self.assertIn("Placement pattern file not found", str(context.exception))

    def test_is_expert_on_current_rank(self):
        # 测试专家是否在当前 rank 上
        exists, position = self.expert_mapping.is_expert_on_current_rank(
            layer_idx_moe=0, expert_id=0, current_rank=0, experts_per_rank=2
        )
        self.assertTrue(exists)  # pattern[0, 0, 0] = 1
        self.assertEqual(position, 0)

        exists, position = self.expert_mapping.is_expert_on_current_rank(
            layer_idx_moe=0, expert_id=1, current_rank=0, experts_per_rank=2
        )
        self.assertFalse(exists)  # pattern[0, 0, 1] = 0
        self.assertEqual(position, 1)

        # 测试 None pattern 的默认部署
        em = ExpertMapping(pattern_path="", device="cpu")
        exists, position = em.is_expert_on_current_rank(
            layer_idx_moe=0, expert_id=1, current_rank=0, experts_per_rank=2
        )
        self.assertTrue(exists)  # 默认部署：expert_id 1 在 rank 0 上
        self.assertEqual(position, 1)

    def test_get_num_of_redundant_experts(self):
        # 测试冗余专家数量
        num_redundant = self.expert_mapping.get_num_of_redundant_experts(
            moe_layer_idx=0, num_expert_per_device_origin=2, rank_device=0
        )
        self.assertEqual(num_redundant, 0)  # pattern[0, 0] 有 2 个专家，等于原始数量

        num_redundant = self.expert_mapping.get_num_of_redundant_experts(
            moe_layer_idx=1, num_expert_per_device_origin=1, rank_device=0
        )
        self.assertEqual(num_redundant, 1)  # pattern[0, 1] 有 2 个专家，多于 1

        # 测试 None pattern
        em = ExpertMapping(pattern_path="", device="cpu")
        self.assertEqual(em.get_num_of_redundant_experts(0), 0)

    def test_get_total_num_layers(self):
        # 测试获取总层数
        num_layers = self.expert_mapping.get_total_num_layers()
        self.assertEqual(num_layers, 2)

    def test_get_total_deployed_experts(self):
        # 测试获取总部署专家数
        total_experts = self.expert_mapping.get_total_deployed_experts()
        self.assertEqual(total_experts, 4)  # pattern[:, 0, :] 总和为 4

    def test_apply_local_expert_mapping(self):
        # 测试本地专家映射
        token_expert_ids = torch.tensor([0, 2], dtype=torch.int32)
        result = self.expert_mapping._apply_local_expert_mapping(
            layer_idx_moe=0, token_expert_ids=token_expert_ids
        )
        self.assertEqual(result.shape, token_expert_ids.shape)
        self.assertTrue(torch.all(result == 0))  # local_expert_mapping 初始化为全 0

    def test_get_local_expert_indices_offset(self):
        # 测试获取总部署专家数
        layer0_rank0_offset = self.expert_mapping.get_local_expert_indices_offset(0, 0, 4)
        layer0_rank1_offset = self.expert_mapping.get_local_expert_indices_offset(0, 1, 4)
        layer1_rank0_offset = self.expert_mapping.get_local_expert_indices_offset(1, 0, 4)
        layer1_rank1_offset = self.expert_mapping.get_local_expert_indices_offset(1, 1, 4)
        self.assertEqual(layer0_rank0_offset, 0)
        self.assertEqual(layer0_rank1_offset, 2)
        self.assertEqual(layer1_rank0_offset, 0)
        self.assertEqual(layer1_rank1_offset, 2)

    def test_get_local_expert_indices_offset_default(self):
        with patch.object(self.expert_mapping, 'placement_pattern', None):
            # 测试获取总部署专家数
            layer0_rank0_offset = self.expert_mapping.get_local_expert_indices_offset(0, 0, 4)
            layer0_rank1_offset = self.expert_mapping.get_local_expert_indices_offset(0, 1, 4)
            layer1_rank0_offset = self.expert_mapping.get_local_expert_indices_offset(1, 0, 4)
            layer1_rank1_offset = self.expert_mapping.get_local_expert_indices_offset(1, 1, 4)
            self.assertEqual(layer0_rank0_offset, 0)
            self.assertEqual(layer0_rank1_offset, 4)
            self.assertEqual(layer1_rank0_offset, 0)
            self.assertEqual(layer1_rank1_offset, 4)

    def test_get_deployed_experts_per_layer_basic_case(self):
        """测试基本情况：2个rank，3个layer，4个expert"""
        placement_pattern = torch.tensor([
            # rank 0
            [
                [1, 0, 1, 0],  # layer 0
                [0, 1, 0, 1],  # layer 1
                [1, 1, 1, 0]   # layer 2
            ],
            # rank 1
            [
                [0, 1, 0, 1],  # layer 0
                [1, 0, 1, 0],  # layer 1
                [0, 1, 0, 1]   # layer 2
            ]
        ])
        with patch.object(self.expert_mapping, 'placement_pattern', placement_pattern):
            result = self.expert_mapping.get_deployed_experts_per_layer()

            expected = torch.tensor([4, 4, 5])  # layer 0: 4, layer 1: 4, layer 2: 5
            self.assertEqual(result, expected.tolist(), f"Expected {expected}, but got {result}")

    def test_get_deployed_experts_per_layer_all_zeros(self):
        """测试全0情况"""
        placement_pattern = torch.zeros((2, 3, 4), dtype=torch.int)
        with patch.object(self.expert_mapping, 'placement_pattern', placement_pattern):
            result = self.expert_mapping.get_deployed_experts_per_layer()

            expected = torch.tensor([0, 0, 0])
            self.assertEqual(result, expected.tolist(), f"Expected {expected}, but got {result}")

    def test_get_deployed_experts_per_layer_all_ones(self):
        """测试全1情况"""
        placement_pattern = torch.ones((2, 3, 4), dtype=torch.int)
        with patch.object(self.expert_mapping, 'placement_pattern', placement_pattern):
            result = self.expert_mapping.get_deployed_experts_per_layer()

            expected = torch.tensor([8, 8, 8])  # 2 ranks * 4 experts = 8 per layer
            self.assertEqual(result, expected.tolist(), f"Expected {expected}, but got {result}")

    def test_get_deployed_experts_per_layer_single_rank(self):
        """测试只有一个rank的情况"""
        placement_pattern = torch.tensor([
            [
                [1, 1, 0, 0],  # layer 0
                [0, 0, 1, 1]   # layer 1
            ]
        ])
        with patch.object(self.expert_mapping, 'placement_pattern', placement_pattern):
            result = self.expert_mapping.get_deployed_experts_per_layer()

            expected = torch.tensor([2, 2])
            self.assertEqual(result, expected.tolist(), f"Expected {expected}, but got {result}")

    def test_get_deployed_experts_per_layer_empty_experts(self):
        """测试expert维度为0的情况"""
        placement_pattern = torch.tensor([[[], [], []], [[], [], []]])  # (2, 3, 0)
        with patch.object(self.expert_mapping, 'placement_pattern', placement_pattern):
            result = self.expert_mapping.get_deployed_experts_per_layer()

            expected = torch.tensor([0, 0, 0])
            self.assertEqual(result, expected.tolist(), f"Expected {expected}, but got {result}")

    def test_get_redundant_enable_per_layer(self):
        """测试只有一个rank的情况"""
        placement_pattern = torch.tensor([
            [
                [1, 1, 0, 1],  # layer 0
                [0, 0, 1, 1]   # layer 1
            ], # Rank0
            [
                [0, 1, 1, 1],  # layer 0
                [1, 1, 0, 0]   # layer 1
            ] # Rank1
        ])
        with patch.object(self.expert_mapping, 'placement_pattern', placement_pattern):
            result = self.expert_mapping.get_redundant_enable_per_layer()

        expected = [True,False]
        self.assertEqual(result, expected, f"Expected {expected}, but got {result}")



    def test_construct_global_expert_mapping(self):

        temp_dir = tempfile.TemporaryDirectory()
        pattern_file = os.path.join(temp_dir.name, "pattern.npy")

        """构造pattern"""
        pattern_data = torch.tensor(
            [
                [[1, 0, 1, 0], [0, 1, 0, 1]], # rank 0
                [[1, 1, 0, 0], [0, 0, 1, 1]]  # rank 1
            ])
        np.save(pattern_file, pattern_data)

        max_redundant_count = 2
        em = ExpertMapping(pattern_path=pattern_file, device="npu", max_redundants_per_expert=max_redundant_count)

        # Test parameters (adjust these based on actual values or make them configurable)
        num_layers = em.get_total_num_layers()
        num_experts = em.get_total_num_expert()

        # Expected redundant count
        expected_redundant_count = [2, 1, 1, 0, 0, 1, 1, 2]

        # Test redundant_count_per_expert_
        for layer in range(num_layers):
            for expert in range(num_experts):
                value = em.redundant_count_per_expert[layer, expert]
                expected = expected_redundant_count[layer * num_experts + expert]
                assert value == expected, f"Mismatch at layer {layer}, expert {expert}: expected {expected}, got {value}"

        # Expected global expert mapping
        expected_mapping = [0, 2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 3]

        # Test global_expert_mapping_
        for layer in range(num_layers):
            for expert in range(num_experts):
                for index in range(max_redundant_count):
                    value = em.global_expert_mapping[layer, expert, index]
                    expected_index = (layer * num_experts + expert) * max_redundant_count + index
                    expected = expected_mapping[expected_index]
                    assert value == expected, (
                        f"Mismatch at layer {layer}, expert {expert}, index {index}: "
                        f"expected {expected}, got {value}"
                    )
        temp_dir.cleanup()

    def test_all_layers_valid(self):
        """
        Test case where all layers satisfy the default placement requirements.
        """
        # Setup: world_size=2, num_layers=3, num_experts=4
        # Experts 0,1 on rank 0; Experts 2,3 on rank 1
        placement_pattern = torch.tensor([
            [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]],  # Rank 0
            [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]   # Rank 1
        ], dtype=torch.long)


        with patch.object(self.expert_mapping, 'placement_pattern', placement_pattern):
            result = self.expert_mapping.get_default_placement_layers()

            expected = [True, True, True]
            self.assertEqual(result, expected, "All layers should be valid")

    def test_some_layers_invalid(self):
        """
        Test case where some layers do not satisfy the default placement requirements.
        """
        # Setup: world_size=2, num_layers=3, num_experts=4
        # Layer 1 has incorrect placement for expert 0 (rank 0)
        placement_pattern = torch.tensor([
            [[1, 1, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]],  # Rank 0
            [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]   # Rank 1
        ], dtype=torch.long)

        with patch.object(self.expert_mapping, 'placement_pattern', placement_pattern):
            result = self.expert_mapping.get_default_placement_layers()

            expected = [True, False, True]
            self.assertEqual(result, expected, "Layer 1 should be invalid due to expert 0 placement")

    def test_empty_layers(self):
        """
        Test case with no layers (num_layers=0).
        """
        # Setup: world_size=2, num_layers=0, num_experts=4
        placement_pattern = torch.zeros((2, 0, 4), dtype=torch.long)

        with patch.object(self.expert_mapping, 'placement_pattern', placement_pattern):
            result = self.expert_mapping.get_default_placement_layers()

            expected = []
            self.assertEqual(result, expected, "Empty layers should return empty list")

    def test_full_layers(self):
        """
        Test case where expected ranks exceed world_size.
        """
        # Setup: world_size=2, num_layers=2, num_experts=8
        # This will cause expected_ranks to go beyond world_size
        placement_pattern = torch.ones((2, 2, 8), dtype=torch.long)

        with patch.object(self.expert_mapping, 'placement_pattern', placement_pattern):
                result = self.expert_mapping.get_default_placement_layers()
                expected = [False, False]
                self.assertEqual(result, expected, "All layers should be invalid due to expert 0 placement")

    def test_single_expert_per_rank(self):
        """
        Test case with one expert per rank.
        """
        # Setup: world_size=2, num_layers=2, num_experts=2
        placement_pattern = torch.tensor([
            [[1, 0], [1, 0]],  # Rank 0
            [[0, 1], [0, 1]]   # Rank 1
        ], dtype=torch.long)

        with patch.object(self.expert_mapping, 'placement_pattern', placement_pattern):
            result = self.expert_mapping.get_default_placement_layers()

            expected = [True, True]
            self.assertEqual(result, expected, "All layers should be valid with one expert per rank")

if __name__ == "__main__":
    unittest.main()