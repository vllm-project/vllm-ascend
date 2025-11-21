# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import random
import time
import unittest
import warnings

import numpy as np
import torch

from ..omni_placement.cluster_status import ClusterStatus
from ..omni_placement.expert_mapping import ExpertMapping
from ..omni_placement.optim.redundant_experts_optimizer import \
    Redundant2ExpertsBalancer

# Suppress warnings
warnings.filterwarnings('ignore')
# device = torch.device('npu' if torch.npu.is_available() else 'cpu')
device = 'cpu'


class TestTokenBalance(unittest.TestCase):

    def setUp(self):
        self.path = './patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy'
        expert_placement = np.load(self.path).astype(np.int32)

        self.placement_pattern = torch.tensor(expert_placement,
                                              dtype=torch.int64,
                                              device=device)
        print('Testing....shape of ', self.placement_pattern.shape)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        self.cluster_status = ClusterStatus(config=None,
                                            expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        self.optimizer = Redundant2ExpertsBalancer(self.cluster_status)

        # 创建一个TokenBalance实例用于测试
        self.optimizer_DEVSPEC = Redundant2ExpertsBalancer(
            self.cluster_status,
            rank=0,
            Device_specific=True,
            num_device_per_host=8)

    def test_expert_mapping_initialization(self):
        """测试 placement_pattern 是否正确初始化"""
        self.assertTrue(
            torch.equal(self.optimizer.placement_pattern,
                        self.placement_pattern),
            "Optimizer 的 placement_pattern 初始化不正确")

    def test_expert_mapping_initialization_DEVSPEC(self):
        """测试 placement_pattern 是否正确初始化"""
        self.assertTrue(
            torch.equal(self.optimizer_DEVSPEC.placement_pattern,
                        self.placement_pattern),
            "Optimizer_DEVSPEC 的 placement_pattern 初始化不正确")

    def test_build_expert_mapping_to_origin_pos(self):
        """测试 expert_position_A 和 expert_position_B 是否正确生成"""
        self.optimizer._build_expert_mapping()
        self.assertIsNotNone(self.optimizer.expert_position_A,
                             "expert_position_A 没有正确初始化")
        self.assertIsNotNone(self.optimizer.expert_position_B,
                             "expert_position_B 没有正确初始化")
        self.assertEqual(self.optimizer.expert_position_A.shape,
                         self.optimizer.expert_position_B.shape,
                         "expert_position_A 和 expert_position_B 形状应相同")

    def test_build_expert_mapping_to_origin_pos_DEVSPEC(self):
        """测试 expert_mapping_frozen 是否正确生成"""
        self.assertIsNotNone(self.optimizer_DEVSPEC.expert_mapping_frozen,
                             "expert_mapping_frozen 没有正确初始化")

    def test_optimize_output_shapes(self):
        """测试 optimize 方法的输出形状是否正确"""
        batch_size, seq_len, hidden_dim = 4, 10, 64
        top_k = 2
        layer_idx_moe = 0

        tokens = torch.rand((batch_size, seq_len, hidden_dim), device=device)
        token_expert_id = torch.randint(
            0,
            self.optimizer.placement_pattern.shape[2] - 1,
            (batch_size, seq_len, top_k),
            device=device)
        token_scores = torch.rand((batch_size, seq_len, top_k), device=device)

        tokens_out, token_expert_id_out, token_scores_out = self.optimizer.optimize(
            layer_idx_moe, tokens.clone(), token_expert_id.clone(),
            token_scores.clone(), [])

        self.assertEqual(tokens.shape, tokens_out.shape, "tokens 输出形状不匹配")
        self.assertEqual(token_expert_id.shape, token_expert_id_out.shape,
                         "token_expert_id 输出形状不匹配")
        self.assertEqual(token_scores.shape, token_scores_out.shape,
                         "token_scores 输出形状不匹配")

    def test_optimize_output_shapes_DEVSPEC(self):
        """测试 optimize 方法的输出形状是否正确"""
        batch_size, seq_len, hidden_dim = 4, 10, 64
        top_k = 2
        layer_idx_moe = 0

        tokens = torch.rand((batch_size, seq_len, hidden_dim), device=device)
        token_expert_id = torch.randint(
            0,
            self.optimizer.placement_pattern.shape[2] - 1,
            (batch_size, seq_len, top_k),
            device=device)
        token_scores = torch.rand((batch_size, seq_len, top_k), device=device)

        tokens_out, token_expert_id_out, token_scores_out = self.optimizer_DEVSPEC.optimize(
            layer_idx_moe, tokens.clone(), token_expert_id.clone(),
            token_scores.clone(), self.cluster_status)

        self.assertEqual(tokens.shape, tokens_out.shape, "tokens 输出形状不匹配")
        self.assertEqual(token_expert_id.shape, token_expert_id_out.shape,
                         "token_expert_id 输出形状不匹配")
        self.assertEqual(token_scores.shape, token_scores_out.shape,
                         "token_scores 输出形状不匹配")

    def test_optimize_correctness(self):
        """测试 optimize 方法的 token_expert_id 是否正确映射"""
        batch_size, seq_len, top_k = 4, 10, 2
        layer_idx_moe = 0

        token_expert_id = torch.randint(
            0,
            self.optimizer.placement_pattern.shape[2] - 1,
            (batch_size, seq_len, top_k),
            device=device)
        token_expert_id_expected = token_expert_id.clone()

        midpoint = batch_size // 2
        token_expert_id_expected[:midpoint] = self.optimizer.expert_position_A[
            layer_idx_moe, token_expert_id[:midpoint]]
        token_expert_id_expected[midpoint:] = self.optimizer.expert_position_B[
            layer_idx_moe, token_expert_id[midpoint:]]

        _, token_expert_id_out, _ = self.optimizer.optimize(
            layer_idx_moe, None, token_expert_id.clone(), None, [])

        self.assertTrue(
            torch.equal(token_expert_id_out, token_expert_id_expected),
            "optimize 方法未正确更新 token_expert_id")

    def test_optimize_correctness_DEVSPEC(self):
        """测试 optimize 方法的 token_expert_id 是否正确映射"""
        batch_size, seq_len, top_k = 4, 10, 2
        layer_idx_moe = 0

        token_expert_id = torch.randint(
            0,
            self.optimizer_DEVSPEC.placement_pattern.shape[2] - 1,
            (batch_size, seq_len, top_k),
            device=device)
        token_expert_id_expected = token_expert_id.clone()

        token_expert_id_expected = self.optimizer_DEVSPEC.expert_mapping_frozen[
            layer_idx_moe, token_expert_id]

        _, token_expert_id_out, _ = self.optimizer_DEVSPEC.optimize(
            layer_idx_moe, None, token_expert_id.clone(), None,
            self.cluster_status)

        self.assertTrue(
            torch.equal(token_expert_id_out, token_expert_id_expected),
            "optimize 方法未正确更新 token_expert_id")

    def test_empty_input_tokens(self):
        """测试 optimize 方法在空 tokens 输入时的行为"""
        layer_idx_moe = 0
        tokens = torch.tensor([], dtype=torch.float32,
                              device=device)  # 空 tokens
        token_expert_id = torch.randint(
            0,
            self.optimizer.placement_pattern.shape[2], (1, 1, 1),
            device=device)
        token_scores = torch.randn((1, 1, 1), device=device)

        tokens_out, token_expert_id_out, token_scores_out = self.optimizer.optimize(
            layer_idx_moe, tokens, token_expert_id, token_scores, [])

        self.assertEqual(tokens_out.shape[0], 0, "空 tokens 输入时应返回空 tokens")
        self.assertEqual(token_expert_id_out.shape, token_expert_id.shape,
                         "空 tokens 不应影响 token_expert_id 形状")
        self.assertEqual(token_scores_out.shape, token_scores.shape,
                         "空 tokens 不应影响 token_scores 形状")

    def test_empty_input_tokens_DEVSPEC(self):
        """测试 optimize_DEVSPEC 方法在空 tokens 输入时的行为"""
        layer_idx_moe = 0
        tokens = torch.tensor([], dtype=torch.float32,
                              device=device)  # 空 tokens
        token_expert_id = torch.randint(
            0,
            self.optimizer_DEVSPEC.placement_pattern.shape[2], (1, 1, 1),
            device=device)
        token_scores = torch.randn((1, 1, 1), device=device)

        tokens_out, token_expert_id_out, token_scores_out = self.optimizer_DEVSPEC.optimize(
            layer_idx_moe, tokens, token_expert_id, token_scores,
            self.cluster_status)

        self.assertEqual(tokens_out.shape[0], 0, "空 tokens 输入时应返回空 tokens")
        self.assertEqual(token_expert_id_out.shape, token_expert_id.shape,
                         "空 tokens 不应影响 token_expert_id 形状")
        self.assertEqual(token_scores_out.shape, token_scores.shape,
                         "空 tokens 不应影响 token_scores 形状")

    def test_single_token_case(self):
        """测试 optimize 方法在单个 token 输入时的行为"""
        layer_idx_moe = 0
        tokens = torch.rand((1, 1, 64), device=device)  # 1 个 token
        token_expert_id = torch.randint(
            0,
            self.optimizer.placement_pattern.shape[2], (1, 1, 1),
            device=device)
        token_scores = torch.rand((1, 1, 1), device=device)

        tokens_out, token_expert_id_out, token_scores_out = self.optimizer.optimize(
            layer_idx_moe, tokens, token_expert_id, token_scores, [])

        self.assertEqual(tokens_out.shape, tokens.shape, "单 token 处理后形状应保持不变")
        self.assertEqual(token_expert_id_out.shape, token_expert_id.shape,
                         "单 token 的 expert ID 应保持形状不变")
        self.assertEqual(token_scores_out.shape, token_scores.shape,
                         "单 token 的 scores 应保持形状不变")

    def test_single_token_case_DEVSPEC(self):
        """测试 optimize_DEVSPEC 方法在单个 token 输入时的行为"""
        layer_idx_moe = 0
        tokens = torch.rand((1, 1, 64), device=device)  # 1 个 token
        token_expert_id = torch.randint(
            0,
            self.optimizer_DEVSPEC.placement_pattern.shape[2], (1, 1, 1),
            device=device)
        token_scores = torch.rand((1, 1, 1), device=device)

        tokens_out, token_expert_id_out, token_scores_out = self.optimizer_DEVSPEC.optimize(
            layer_idx_moe, tokens, token_expert_id, token_scores,
            self.cluster_status)

        self.assertEqual(tokens_out.shape, tokens.shape, "单 token 处理后形状应保持不变")
        self.assertEqual(token_expert_id_out.shape, token_expert_id.shape,
                         "单 token 的 expert ID 应保持形状不变")
        self.assertEqual(token_scores_out.shape, token_scores.shape,
                         "单 token 的 scores 应保持形状不变")

    def test_invalid_expert_id(self):
        """测试 optimize 方法在 token_expert_id 超出范围时的行为"""
        layer_idx_moe = 0
        batch_size, seq_len, top_k = 4, 10, 2

        tokens = torch.rand((batch_size, seq_len, 64), device=device)
        token_expert_id = torch.full((batch_size, seq_len, top_k),
                                     fill_value=9999,
                                     dtype=torch.int64,
                                     device=device)  # 非法 ID
        token_scores = torch.rand((batch_size, seq_len, top_k), device=device)

        with self.assertRaises(IndexError):
            self.optimizer.optimize(layer_idx_moe, tokens, token_expert_id,
                                    token_scores, [])

    def test_random_mapping_stability(self):
        """测试 _build_expert_mapping_to_origin_pos 是否在相同输入下稳定"""
        self.optimizer._build_expert_mapping()
        expert_position_A_1 = self.optimizer.expert_position_A.clone()
        expert_position_B_1 = self.optimizer.expert_position_B.clone()

        # 重新初始化 optimizer 来强制重新计算映射
        self.optimizer = Redundant2ExpertsBalancer(self.cluster_status)
        self.optimizer._build_expert_mapping()
        expert_position_A_2 = self.optimizer.expert_position_A.clone()
        expert_position_B_2 = self.optimizer.expert_position_B.clone()

        self.assertTrue(torch.equal(expert_position_A_1, expert_position_A_2),
                        "expert_position_A 在相同输入下应保持稳定")
        self.assertTrue(torch.equal(expert_position_B_1, expert_position_B_2),
                        "expert_position_B 在相同输入下应保持稳定")

    def test_matrix_contains_only_zero_and_one(self):
        # 加载矩阵
        expert_placement = np.load(self.path).astype(np.int32)

        # 打印矩阵形状信息，帮助调试
        print(f"矩阵形状: {expert_placement.shape}")

        # 方法1：使用numpy的unique函数检查唯一值
        unique_values = np.unique(expert_placement)
        self.assertTrue(
            np.array_equal(unique_values, np.array([0, 1]))
            or np.array_equal(unique_values, np.array([0]))
            or np.array_equal(unique_values, np.array([1])),
            f"矩阵包含0和1以外的值: {unique_values}")

    def test_matrix_contains_only_zero_and_one_2(self):
        expert_placement = np.load(self.path).astype(np.int32)
        # 方法2：通过逻辑运算检查每个元素是0或1
        is_zero_or_one = np.logical_or(expert_placement == 0,
                                       expert_placement == 1)
        self.assertTrue(np.all(is_zero_or_one), "矩阵包含非0非1的元素")

        # 测试转换为PyTorch张量后的结果
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        placement_pattern = torch.tensor(expert_placement,
                                         dtype=torch.int64,
                                         device=device)
        print(f"PyTorch张量形状: {placement_pattern.shape}")

        # 检查PyTorch张量中的唯一值
        torch_unique = torch.unique(placement_pattern).cpu().numpy()
        self.assertTrue(
            np.array_equal(torch_unique, np.array([0, 1]))
            or np.array_equal(torch_unique, np.array([0]))
            or np.array_equal(torch_unique, np.array([1])),
            f"PyTorch张量包含0和1以外的值: {torch_unique}")

    def test_position_mapping(self):
        # 创建一个示例expert_mapping的矩阵
        # 假设有2个设备，3层网络，每层有4个专家位置

        print("\nTesting position_mapping...")
        count = 0
        # 收集所有有专家的位置坐标
        expert_positions = []
        for device_id in range(self.placement_pattern.shape[0]):
            for layer_id in range(self.placement_pattern.shape[1]):
                for ep_id in range(self.placement_pattern.shape[2]):
                    if self.placement_pattern[device_id, layer_id, ep_id] == 1:
                        # expert_positions.append((device_id, layer_id, ep_id))
                        expert_positions.append(
                            torch.tensor((device_id, layer_id, ep_id),
                                         dtype=torch.int64,
                                         device=device))
                        count += 1

        # 验证position_mapping的计算是否正确
        # for device_id, layer_id, ep_id in expert_positions:
        tt_tt = 0
        for xx in expert_positions:
            device_id, layer_id, ep_id = xx[0], xx[1], xx[2]
            # 手动计算预期的position_mapping值
            expected_value = self.optimizer.expert_per_device * device_id + torch.sum(
                self.placement_pattern[device_id, layer_id, :ep_id]).item()

            # 开始计时
            start_time = time.time()

            # 执行需要计时的代码
            # actual_value = self.optimizer.position_mapping[device_id, layer_id, ep_id]
            actual_value1 = self.optimizer.expert_position_A[layer_id, ep_id]
            actual_value2 = self.optimizer.expert_position_B[layer_id, ep_id]
            PD = (expected_value == actual_value1) or (expected_value
                                                       == actual_value2)
            # 结束计时
            end_time = time.time()

            tt_tt += end_time - start_time

            self.assertEqual(PD, True)

        print('Total time:', tt_tt)

    def test_optimize_function_AB_mapping_equal_when_redundant(self):
        # 检查self.expert_mapping沿第一个维度求和是否有值为2
        if self.placement_pattern is not None:
            # 沿第一个维度求和
            summed_mapping = torch.sum(self.placement_pattern, dim=0)

            # 检查是否有值为2
            has_value_2 = torch.any(summed_mapping >= 2)

            # 比较expert_position_A和expert_position_B
            A_eq_B = torch.equal(self.optimizer.expert_position_A,
                                 self.optimizer.expert_position_B)

            # 如果有值为2，则A和B应该不同
            if has_value_2:
                self.assertFalse(
                    A_eq_B,
                    "当expert_mapping沿第一维求和有值为2时，expert_position_A和expert_position_B应不相等"
                )
            else:
                # 如果没有值为2，则可能相等也可能不相等，取决于你的实现逻辑
                # 如果有特定期望，可以在这里添加相应的断言
                pass

    def test_optimize_function_AB_mapping_equal_when_unique(self):
        # 检查self.expert_mapping沿第一个维度求和是否有值为2
        if self.placement_pattern is not None:
            # 沿第一个维度求和
            summed_mapping = torch.sum(self.placement_pattern, dim=0)
            # 检查是否有值为2
            has_value_larger_1 = torch.any(summed_mapping >= 1)
            # 比较expert_position_A和expert_position_B
            A_eq_B = torch.equal(self.optimizer.expert_position_A,
                                 self.optimizer.expert_position_B)
            # 如果有值为2，则A和B应该不同
            if has_value_larger_1:
                pass
            else:
                # 如果没有值为2，则可能相等也可能不相等，取决于你的实现逻辑
                # 如果有特定期望，可以在这里添加相应的断言
                self.assertTrue(
                    A_eq_B,
                    "当expert_mapping沿第一维求和全是1或0时，expert_position_A和expert_position_B应相等"
                )

    def test_optimize_function_AB_mapping_equal_when_noEP(self):
        # 检查self.expert_mapping沿第一个维度求和是否有值为0
        if self.placement_pattern is not None:
            # 沿第一个维度求和
            summed_mapping = torch.sum(self.placement_pattern, dim=0)
            # 检查是否有值为0
            has_value_0 = torch.any(summed_mapping == 0)
            if has_value_0:
                # 检查expert_position_A和expert_position_B
                self.assertTrue(
                    torch.any(self.optimizer.expert_position_A == -1),
                    "有未分配的expert，但是A映射到了positionID")
                self.assertTrue(
                    torch.any(self.optimizer.expert_position_B == -1),
                    "有未分配的expert，但是B映射到了positionID")

    def test_optimize_output_shape_consistency(self):
        # 设置测试参数
        device = "cpu"
        topk = 4
        layer_id_moe = torch.randint(0, self.placement_pattern.shape[1],
                                     (1, )).item()
        input_token = torch.randint(0, 1000, (1, 10), device=device)
        origin_topk = torch.randint(-1,
                                    self.placement_pattern.shape[2], (1, topk),
                                    device=device)
        token_expert_scores = torch.rand((1, 10, topk), device=device)

        # 调用优化函数
        _, optimized_mapping, _ = self.optimizer.optimize(
            layer_id_moe, input_token, origin_topk, token_expert_scores,
            self.cluster_status)

        # 测试形状一致性
        self.assertEqual(origin_topk.shape, optimized_mapping.shape,
                         "origin_topk和optimized_mapping的形状应该一致")

    def test_optimize_output_shape_consistency_DEVSPEC(self):
        # 设置测试参数
        device = "cpu"
        topk = 4
        layer_id_moe = torch.randint(0, self.placement_pattern.shape[1],
                                     (1, )).item()
        input_token = torch.randint(0, 1000, (1, 10), device=device)
        origin_topk = torch.randint(-1,
                                    self.placement_pattern.shape[2], (1, topk),
                                    device=device)
        token_expert_scores = torch.rand((1, 10, topk), device=device)

        # 调用优化函数
        _, optimized_mapping, _ = self.optimizer_DEVSPEC.optimize(
            layer_id_moe, input_token, origin_topk, token_expert_scores,
            self.cluster_status)

        # 测试形状一致性
        self.assertEqual(origin_topk.shape, optimized_mapping.shape,
                         "origin_topk和optimized_mapping的形状应该一致")

    def test_optimize_output_data_type_consistency(self):
        # 设置测试参数
        device = "cpu"
        topk = 4
        layer_id_moe = torch.randint(0, self.placement_pattern.shape[1],
                                     (1, )).item()
        input_token = torch.randint(0, 1000, (1, 10), device=device)
        origin_topk = torch.randint(-1,
                                    self.placement_pattern.shape[2], (1, topk),
                                    device=device,
                                    dtype=torch.int)
        token_expert_scores = torch.rand((1, 10, topk), device=device)

        # 调用优化函数
        _, optimized_mapping, _ = self.optimizer.optimize(
            layer_id_moe, input_token, origin_topk, token_expert_scores,
            self.cluster_status)

        # 测试数据类型一致性
        self.assertEqual(origin_topk.dtype, optimized_mapping.dtype,
                         "origin_topk和optimized_mapping的数据类型应该一致")

    def test_optimize_output_data_type_consistency_DEVSPEC(self):
        # 设置测试参数
        device = "cpu"
        topk = 4
        layer_id_moe = torch.randint(0, self.placement_pattern.shape[1],
                                     (1, )).item()
        input_token = torch.randint(0, 1000, (1, 10), device=device)
        origin_topk = torch.randint(-1,
                                    self.placement_pattern.shape[2], (1, topk),
                                    device=device,
                                    dtype=torch.int)
        token_expert_scores = torch.rand((1, 10, topk), device=device)

        # 调用优化函数
        _, optimized_mapping, _ = self.optimizer_DEVSPEC.optimize(
            layer_id_moe, input_token, origin_topk, token_expert_scores,
            self.cluster_status)

        # 测试数据类型一致性
        self.assertEqual(origin_topk.dtype, optimized_mapping.dtype,
                         "origin_topk和optimized_mapping的数据类型应该一致")

    @unittest.skip("not for our strategy.")
    def test_optimize_output_range_consistency(self):
        # 设置测试参数
        device = torch.device("cpu")
        topk = 8
        layer_id_moe = torch.randint(0, self.placement_pattern.shape[1],
                                     (1, )).item()
        input_token = torch.randint(0, 1000, (1, 10), device=device)
        origin_topk = torch.randint(-1,
                                    self.placement_pattern.shape[2] - 1,
                                    (100, topk),
                                    device=device)
        token_expert_scores = torch.rand((1, 10, topk), device=device)

        # 调用优化函数
        _, optimized_mapping, _ = self.optimizer.optimize(
            layer_id_moe, input_token, origin_topk, token_expert_scores,
            self.cluster_status)

        # 测试数值范围一致性
        self.assertLessEqual(
            optimized_mapping.max().item(),
            self.placement_pattern.shape[0] * self.optimizer.expert_per_device
            - 1, "optimized_mapping的最大值不应超过position总量")
        self.assertGreaterEqual(optimized_mapping.min().item(), -1,
                                "optimized_mapping的最小值不应小于-1")

    @unittest.skip("not for our strategy.")
    def test_optimize_output_range_consistency_DEVSPEC(self):
        # 设置测试参数
        device = torch.device("cpu")
        topk = 8
        layer_id_moe = torch.randint(0, self.placement_pattern.shape[1],
                                     (1, )).item()
        input_token = torch.randint(0, 1000, (1, 10), device=device)
        origin_topk = torch.randint(-1,
                                    self.placement_pattern.shape[2] - 1,
                                    (100, topk),
                                    device=device)
        token_expert_scores = torch.rand((1, 10, topk), device=device)

        # 调用优化函数
        _, optimized_mapping, _ = self.optimizer_DEVSPEC.optimize(
            layer_id_moe, input_token, origin_topk, token_expert_scores,
            self.cluster_status)

        # 测试数值范围一致性
        self.assertLessEqual(
            optimized_mapping.max().item(), self.placement_pattern.shape[0] *
            self.optimizer_DEVSPEC.expert_per_device - 1,
            "optimized_mapping的最大值不应超过position总量")
        self.assertGreaterEqual(optimized_mapping.min().item(), -1,
                                "optimized_mapping的最小值不应小于-1")

    def test_optimize_function(self):
        # 设置随机种子以确保结果可重现
        # random.seed(42)
        # torch.manual_seed(42)
        # device = torch.device('npu' if torch.npu.is_available() else 'cpu')
        device = 'cpu'
        ep_per_device = self.optimizer.expert_per_device
        print(f"\nExperts per device: {ep_per_device}")

        # 设置测试参数
        num_tokens = 100  # 输入tokens数量
        topk = 32  # 每个token选择的专家数

        # 随机生成layer_id (范围在0到expert_mapping.shape[1]-1之间)
        layer_id_moe = random.randint(0, self.placement_pattern.shape[1] -
                                      1)  # 注意API中layer_id从1开始
        print(f"\nRandom layer_id: {layer_id_moe}")

        # 随机生成origin_topk矩阵 (形状为num_tokens x topk)
        # 值的范围在-1到expert_mapping.shape[2]-1之间

        origin_topk = torch.randint(0,
                                    self.placement_pattern.shape[2] - 1,
                                    (num_tokens, topk),
                                    device=device)
        print(f"\nRandom origin_topk shape: {origin_topk.shape}")

        # 创建虚拟的token和token_expert_scores (值不重要，因为在当前实现中未使用)
        input_token = torch.zeros(num_tokens)
        token_expert_scores = torch.rand(num_tokens, topk)

        # 调用optimize函数获取优化后的映射, 是位置id
        print("\nCalling optimize function...")
        layer_id_moe = torch.tensor(layer_id_moe,
                                    dtype=torch.int64,
                                    device=device)

        start_time = time.time()

        _, optimized_mapping, _ = self.optimizer.optimize(
            layer_id_moe, input_token, origin_topk.clone(),
            token_expert_scores, self.cluster_status)

        end_time = time.time()

        print('Total time:', end_time - start_time, optimized_mapping.shape)

        print(f"\nOptimized mapping shape: {optimized_mapping.shape}")

        # 逐个检查每个值是否满足条件
        for i in range(num_tokens):
            for j in range(topk):

                value_ori = origin_topk[i][j]

                pos_id = optimized_mapping[i][j]
                if pos_id == -1:
                    self.assertEqual(
                        torch.sum(self.placement_pattern[:, layer_id_moe,
                                                         value_ori]), 0)
                    continue

                device_id = pos_id // ep_per_device
                pos_this_device = pos_id % ep_per_device

                self.assertEqual(
                    self.placement_pattern[device_id, layer_id_moe, value_ori],
                    1)
                self.assertEqual(
                    torch.sum(
                        self.placement_pattern[device_id,
                                               layer_id_moe, :value_ori]),
                    pos_this_device)

    @unittest.skip("not ready for pos -1.")
    def test_optimize_function_DEVSPEC(self):
        # 设置随机种子以确保结果可重现
        # random.seed(42)
        # torch.manual_seed(42)
        # device = torch.device('npu' if torch.npu.is_available() else 'cpu')
        device = 'cpu'
        ep_per_device = self.optimizer_DEVSPEC.expert_per_device
        print(f"\nExperts per device: {ep_per_device}")

        # 设置测试参数
        num_tokens = 100  # 输入tokens数量
        topk = 32  # 每个token选择的专家数

        # 随机生成layer_id (范围在0到expert_mapping.shape[1]-1之间)
        layer_id_moe = random.randint(0, self.placement_pattern.shape[1] -
                                      1)  # 注意API中layer_id从1开始
        print(f"\nRandom layer_id: {layer_id_moe}")

        # 随机生成origin_topk矩阵 (形状为num_tokens x topk)
        # 值的范围在-1到expert_mapping.shape[2]-1之间

        origin_topk = torch.randint(0,
                                    self.placement_pattern.shape[2] - 1,
                                    (num_tokens, topk),
                                    device=device)
        print(f"\nRandom origin_topk shape: {origin_topk.shape}")

        # 创建虚拟的token和token_expert_scores (值不重要，因为在当前实现中未使用)
        input_token = torch.zeros(num_tokens)
        token_expert_scores = torch.rand(num_tokens, topk)

        # 调用optimize函数获取优化后的映射, 是位置id
        print("\nCalling optimize function...")
        layer_id_moe = torch.tensor(layer_id_moe,
                                    dtype=torch.int64,
                                    device=device)

        start_time = time.time()

        _, optimized_mapping, _ = self.optimizer_DEVSPEC.optimize(
            layer_id_moe, input_token, origin_topk.clone(),
            token_expert_scores, self.cluster_status)

        end_time = time.time()

        print('Total time:', end_time - start_time, optimized_mapping.shape)

        print(f"\nOptimized mapping shape: {optimized_mapping.shape}")

        # 逐个检查每个值是否满足条件
        for i in range(num_tokens):
            for j in range(topk):

                value_ori = origin_topk[i][j]

                pos_id = optimized_mapping[i][j]
                if pos_id == -1:
                    self.assertEqual(
                        torch.sum(self.placement_pattern[:, layer_id_moe,
                                                         value_ori]), 0)
                    continue

                device_id = pos_id // ep_per_device
                pos_this_device = pos_id % ep_per_device

                self.assertEqual(
                    self.placement_pattern[device_id, layer_id_moe, value_ori],
                    1)
                self.assertEqual(
                    torch.sum(
                        self.placement_pattern[device_id,
                                               layer_id_moe, :value_ori]),
                    pos_this_device)

    def test_optimize_function_2(self):
        device = 'cpu'
        expert_mapping_tmp = torch.zeros(10,
                                         3,
                                         30,
                                         dtype=torch.int32,
                                         device=device)
        num_devices, num_layers, num_epids = expert_mapping_tmp.shape
        ep_per_device = num_epids // num_devices

        for ep_id in range(num_epids):
            expert_mapping_tmp[ep_id // ep_per_device, :, ep_id] = 1

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp = Redundant2ExpertsBalancer(cluster_status_tmp)

        origin_topk = torch.randint(0,
                                    expert_mapping_tmp.shape[2] - 1, (50, 50),
                                    dtype=torch.int32,
                                    device=device)

        _, optimized_mapping, _ = optimizer_tmp.optimize(
            0, expert_mapping_tmp, origin_topk.clone(), expert_mapping_tmp,
            self.cluster_status)

        self.assertEqual(True, torch.equal(optimized_mapping, origin_topk))

    def test_optimize_function_22(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0], [1, 0, 1, 0]], [[0, 0, 1, 1], [0, 1, 0, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp = Redundant2ExpertsBalancer(cluster_status_tmp)

        origin_topk = torch.randint(0,
                                    expert_mapping_tmp.shape[2] - 1, (10, 10),
                                    dtype=torch.int32,
                                    device=device)

        _, optimized_mapping, _ = optimizer_tmp.optimize(
            0, expert_mapping_tmp, origin_topk.clone(), expert_mapping_tmp,
            self.cluster_status)
        # print(optimized_mapping,  origin_topk)
        self.assertEqual(True, torch.equal(optimized_mapping, origin_topk))

    def test_optimize_function_3(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0], [1, 0, 1, 0]], [[0, 0, 1, 1], [0, 1, 0, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp = Redundant2ExpertsBalancer(cluster_status_tmp)

        origin_topk = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]],
                                   dtype=torch.int32,
                                   device=device)

        _, optimized_mapping, _ = optimizer_tmp.optimize(
            0, expert_mapping_tmp, origin_topk.clone(), expert_mapping_tmp,
            self.cluster_status)
        # print(optimized_mapping,  origin_topk)
        self.assertEqual(True, torch.equal(optimized_mapping, origin_topk))

    def test_optimize_function_4(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0], [1, 0, 1, 0]], [[0, 0, 1, 1], [0, 1, 0, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp = Redundant2ExpertsBalancer(cluster_status_tmp)

        origin_topk = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]],
                                   dtype=torch.int32,
                                   device=device)

        check_topk = torch.tensor([[0, 2, 1, 3], [0, 2, 1, 3]],
                                  dtype=torch.int32,
                                  device=device)

        _, optimized_mapping, _ = optimizer_tmp.optimize(
            1, expert_mapping_tmp, origin_topk.clone(), expert_mapping_tmp,
            self.cluster_status)
        # print(optimized_mapping,  origin_topk)
        self.assertEqual(True, torch.equal(check_topk, optimized_mapping))

    def test_optimize_function_5(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 1, 1], [1, 0, 1, 0]], [[1, 1, 1, 1], [0, 1, 0, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp = Redundant2ExpertsBalancer(cluster_status_tmp)

        origin_topk = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]],
                                   dtype=torch.int32,
                                   device=device)

        check_topk = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]],
                                  dtype=torch.int32,
                                  device=device)

        _, optimized_mapping, _ = optimizer_tmp.optimize(
            0, expert_mapping_tmp, origin_topk.clone(), expert_mapping_tmp,
            self.cluster_status)
        # print(optimized_mapping,  origin_topk)
        self.assertEqual(True, torch.equal(check_topk, optimized_mapping))

    def test_optimize_function_6(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 1, 0], [1, 0, 1, 0]], [[1, 1, 1, 0], [0, 1, 0, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp = Redundant2ExpertsBalancer(cluster_status_tmp)

        origin_topk = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]],
                                   dtype=torch.int32,
                                   device=device)

        check_topk = torch.tensor([[0, 1, 2, -1], [3, 4, 5, -1]],
                                  dtype=torch.int32,
                                  device=device)

        _, optimized_mapping, _ = optimizer_tmp.optimize(
            0, expert_mapping_tmp, origin_topk.clone(), expert_mapping_tmp,
            self.cluster_status)
        # print(optimized_mapping,  origin_topk)
        self.assertEqual(True, torch.equal(check_topk, optimized_mapping))

    def test_large_input_performance(self):
        """测试 optimize 方法在大规模输入下的性能"""
        device = 'cpu'

        batch_size, seq_len, hidden_dim = 1024, 512, 64  # 大输入
        top_k = 2
        layer_idx_moe = 0

        tokens = torch.rand((batch_size, seq_len, hidden_dim), device=device)
        token_expert_id = torch.randint(
            0,
            self.optimizer.placement_pattern.shape[2] - 1,
            (batch_size, seq_len, top_k),
            device=device)
        token_scores = torch.rand((batch_size, seq_len, top_k), device=device)

        start_time = time.time()
        self.optimizer.optimize(layer_idx_moe, tokens, token_expert_id,
                                token_scores, [])
        end_time = time.time()

        print(
            f"Large input performance test completed in {end_time - start_time:.4f} seconds"
        )

        self.assertLess(end_time - start_time, 2.0, "optimize 方法在大规模输入下运行时间过长")

    def test_device_compatibility(self):
        """测试 optimize 方法是否支持 CPU/NPU 运行"""
        test_device = 'cpu'
        # if torch.npu.is_available():
        #     test_device = 'npu'

        batch_size, seq_len, hidden_dim = 4, 10, 64
        top_k = 2
        layer_idx_moe = 0

        tokens = torch.rand((batch_size, seq_len, hidden_dim),
                            device=test_device)
        token_expert_id = torch.randint(
            0,
            self.optimizer.placement_pattern.shape[2] - 1,
            (batch_size, seq_len, top_k),
            device=test_device)
        token_scores = torch.rand((batch_size, seq_len, top_k),
                                  device=test_device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        tmp_cluster_status = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        tmp_optimizer = Redundant2ExpertsBalancer(tmp_cluster_status)

        # else:
        #     tmp_optimizer = self.optimizer

        tokens_out, token_expert_id_out, token_scores_out = tmp_optimizer.optimize(
            layer_idx_moe, tokens, token_expert_id, token_scores, [])

        self.assertEqual(tokens_out.device, tokens.device, "tokens 输出设备错误")
        self.assertEqual(token_expert_id_out.device, token_expert_id.device,
                         "token_expert_id 输出设备错误")
        self.assertEqual(token_scores_out.device, token_scores.device,
                         "token_scores 输出设备错误")

    def test_optimize_function_DEVICESPECIFIC_0(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0]],
             [[0, 0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0]],
             [[1, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0]],
             [[0, 1, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 1, 0]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                  rank=0,
                                                  Device_specific=True,
                                                  num_device_per_host=2)

        contains_minus_one = torch.any(
            optimizer_tmp.expert_mapping_frozen == -1)
        self.assertTrue(contains_minus_one, "位置映射矩阵应该不含有-1")

    def test_optimize_function_DEVICESPECIFIC_1(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0]],
             [[0, 0, 1, 1, 0, 0, 0, 1], [1, 1, 0, 1, 0, 0, 0, 0]],
             [[1, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0]],
             [[0, 1, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                  rank=0,
                                                  Device_specific=True,
                                                  num_device_per_host=2)

        contains_minus_one = torch.any(
            optimizer_tmp.expert_mapping_frozen == -1)
        self.assertFalse(contains_minus_one, "位置映射矩阵应该不含有-1")

    @unittest.skip("not for our strategy.")
    def test_optimize_function_DEVICESPECIFIC_2(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0, 0, 1, 1, 0]], [[0, 0, 1, 1, 0, 1, 0, 1]],
             [[1, 0, 0, 0, 1, 1, 0, 0]], [[0, 1, 0, 0, 0, 0, 1, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp1 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=0,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        optimizer_tmp2 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=1,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        A_eq_A = torch.equal(optimizer_tmp1.expert_mapping_frozen,
                             optimizer_tmp2.expert_mapping_frozen)
        # 如果有值为2，则A和B应该不同
        self.assertFalse(
            A_eq_A,
            " device0 和device1在ep5选择不同的映射,device 1 的映射和device 2 的映射应不同")

    def test_optimize_function_DEVICESPECIFIC_3(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 1, 0]], [[0, 0, 1, 1, 0, 0, 0, 1]],
             [[1, 0, 0, 0, 1, 1, 0, 0]], [[0, 1, 0, 0, 0, 0, 1, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp1 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=0,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        optimizer_tmp2 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=1,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        A_eq_A = torch.equal(optimizer_tmp1.expert_mapping_frozen,
                             optimizer_tmp2.expert_mapping_frozen)
        # 如果有值为2，则A和B应该不同
        self.assertTrue(
            A_eq_A,
            "当expert_mappinghost内和host外情况一致的时候, device 1 的映射和device 2 的映射应相等")

    @unittest.skip("not for our strategy.")
    def test_optimize_function_DEVICESPECIFIC_4(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 1, 0]], [[0, 0, 1, 1, 0, 0, 1, 1]],
             [[1, 0, 0, 0, 1, 1, 1, 0]], [[0, 1, 0, 0, 0, 0, 1, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp1 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=0,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        optimizer_tmp2 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=3,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        B_eq_B = torch.equal(optimizer_tmp1.expert_mapping_frozen,
                             optimizer_tmp2.expert_mapping_frozen)
        # 如果有值为2，则A和B应该不同
        self.assertFalse(
            B_eq_B,
            "当expert_mapping沿第一维求和全是1或0时, expert_position_A和expert_position_B应相等"
        )

    @unittest.skip("not for our strategy.")
    def test_optimize_function_DEVICESPECIFIC_5(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 1, 1]], [[0, 0, 1, 1, 0, 0, 1, 1]],
             [[1, 0, 0, 0, 1, 1, 1, 0]], [[1, 1, 0, 0, 0, 0, 1, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp1 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=0,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        optimizer_tmp2 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=3,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        # B_eq_B = torch.equal(optimizer_tmp1.expert_position_A, optimizer_tmp2.expert_position_A)

        ori_topk = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7],
                                dtype=torch.int32,
                                device=device)
        tmp1_topk = optimizer_tmp1.expert_mapping_frozen[0, ori_topk]
        tmp1_topk_device = tmp1_topk // optimizer_tmp1.expert_per_device

        self.assertEqual(int(torch.sum(tmp1_topk_device < 1.5)), 6,
                         "device 分配错误")
        self.assertEqual(int(torch.sum(tmp1_topk_device > 1.5)), 2,
                         "device 分配错误")

    @unittest.skip("not for our strategy.")
    def test_optimize_function_DEVICESPECIFIC_6(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 1, 1]], [[0, 0, 1, 1, 0, 0, 1, 1]],
             [[1, 0, 0, 0, 1, 1, 1, 0]], [[1, 1, 0, 0, 0, 0, 1, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp1 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=0,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        optimizer_tmp2 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=3,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        # B_eq_B = torch.equal(optimizer_tmp1.expert_position_A, optimizer_tmp2.expert_position_A)

        ori_topk = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7],
                                dtype=torch.int32,
                                device=device)
        tmp1_topk = optimizer_tmp1.expert_mapping_frozen[0, ori_topk]
        tmp1_topk_device = tmp1_topk // optimizer_tmp1.expert_per_device

        self.assertEqual(int(torch.sum(tmp1_topk_device < 1.5)), 6,
                         "device 分配错误")
        self.assertEqual(int(torch.sum(tmp1_topk_device > 1.5)), 2,
                         "device 分配错误")

    @unittest.skip("not for our strategy.")
    def test_optimize_function_DEVICESPECIFIC_7(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 1, 1]], [[0, 0, 1, 1, 0, 0, 1, 1]],
             [[1, 0, 0, 0, 1, 1, 1, 0]], [[1, 1, 0, 0, 0, 0, 1, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp1 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=0,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        optimizer_tmp2 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=3,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        # B_eq_B = torch.equal(optimizer_tmp1.expert_position_A, optimizer_tmp2.expert_position_A)

        ori_topk = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7],
                                dtype=torch.int32,
                                device=device)
        tmp2_topk = optimizer_tmp2.expert_mapping_frozen[0, ori_topk]
        tmp2_topk_device = tmp2_topk // optimizer_tmp2.expert_per_device

        self.assertEqual(int(torch.sum(tmp2_topk_device < 1.5)), 2,
                         "device 分配错误")
        self.assertEqual(int(torch.sum(tmp2_topk_device > 1.5)), 6,
                         "device 分配错误")

    @unittest.skip("not for our strategy.")
    def test_optimize_function_DEVICESPECIFIC_8(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 1, 1]], [[0, 0, 1, 1, 0, 0, 1, 1]],
             [[1, 0, 0, 0, 1, 1, 1, 0]], [[1, 1, 0, 0, 0, 0, 1, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试   cluster_status, rank = 0, capacity_factor=1.0, Device_specific = False, num_device_per_host = 8
        optimizer_tmp1 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=0,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        optimizer_tmp2 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=3,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        # B_eq_B = torch.equal(optimizer_tmp1.expert_position_A, optimizer_tmp2.expert_position_A)

        ori_topk = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7],
                                dtype=torch.int32,
                                device=device)

        _, tmp2_topk, _ = optimizer_tmp2.optimize(0, None, ori_topk, None, [])

        tmp2_topk_device = tmp2_topk // optimizer_tmp2.expert_per_device

        self.assertEqual(int(torch.sum(tmp2_topk_device < 1.5)), 2,
                         "device 分配错误")
        self.assertEqual(int(torch.sum(tmp2_topk_device > 1.5)), 6,
                         "device 分配错误")

    @unittest.skip("not for our strategy.")
    def test_optimize_function_DEVICESPECIFIC_9(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 1, 1]], [[0, 0, 1, 1, 0, 0, 1, 1]],
             [[1, 0, 0, 0, 1, 1, 1, 0]], [[1, 1, 0, 0, 0, 0, 1, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp1 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=0,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        optimizer_tmp2 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=1,
                                                   Device_specific=True,
                                                   num_device_per_host=2)
        C_eq_C = torch.equal(optimizer_tmp1.expert_mapping_frozen,
                             optimizer_tmp2.expert_mapping_frozen)

        self.assertFalse(C_eq_C, "不同的device有共同的device 则应该有不同的mapping")

    @unittest.skip("not for our strategy.")
    def test_optimize_function_DEVICESPECIFIC_10(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 1, 1, 1, 1, 1, 1]], [[1, 1, 1, 1, 1, 1, 1, 1]],
             [[1, 1, 1, 1, 1, 1, 1, 1]], [[1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp1 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=0,
                                                   Device_specific=True,
                                                   num_device_per_host=2)

        ori_topk = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7],
                                dtype=torch.int32,
                                device=device)
        tmp_topk = optimizer_tmp1.expert_mapping_frozen[0, ori_topk]

        self.assertTrue(torch.equal(ori_topk, tmp_topk), "device 分配错误")

    def test_optimize_function_DEVICESPECIFIC_11(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 0, 0]], [[0, 0, 1, 1, 0, 0, 0, 0]],
             [[0, 0, 0, 0, 1, 1, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp1 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=0,
                                                   Device_specific=True,
                                                   num_device_per_host=2)

        ori_topk = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7],
                                dtype=torch.int32,
                                device=device)
        tmp_topk = optimizer_tmp1.expert_mapping_frozen[0, ori_topk]

        self.assertTrue(torch.equal(ori_topk, tmp_topk), "device 分配错误")

    @unittest.skip("not for our strategy.")
    def test_optimize_function_DEVICESPECIFIC_12(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 0, 1]], [[0, 0, 1, 1, 0, 0, 0, 1]],
             [[0, 0, 0, 0, 1, 1, 0, 1]], [[1, 0, 0, 0, 0, 0, 1, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp1 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=0,
                                                   Device_specific=True,
                                                   num_device_per_host=2)

        ori_topk = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7],
                                dtype=torch.int32,
                                device=device)
        tmp_topk = optimizer_tmp1.expert_mapping_frozen[0, ori_topk]
        checked_topk = torch.tensor([0, 1, 3, 4, 6, 7, 10, 2],
                                    dtype=torch.int32,
                                    device=device)

        self.assertTrue(torch.equal(checked_topk, tmp_topk), "device 分配错误")

    def test_optimize_function_DEVICESPECIFIC_13(self):
        device = 'cpu'
        expert_mapping_tmp = torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 0, 1]], [[0, 0, 1, 1, 0, 0, 0, 0]],
             [[0, 0, 0, 0, 1, 1, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 1]]],
            dtype=torch.int32,
            device=device)

        expert_mapping = ExpertMapping(
            "../patterns/placement_pattern_3d_v3_indevrrori03152041_58moe_16devices_SC.npy",
            device)
        expert_mapping.placement_pattern = expert_mapping_tmp
        expert_mapping.local_expert_mapping = torch.zeros(
            len(expert_mapping_tmp[0]),  # num_of_devices
            len(expert_mapping_tmp[0][0]),
            dtype=torch.int32,
            device=device)
        cluster_status_tmp = ClusterStatus(config=None,
                                           expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        optimizer_tmp1 = Redundant2ExpertsBalancer(cluster_status_tmp,
                                                   rank=0,
                                                   Device_specific=True,
                                                   num_device_per_host=2)

        ori_topk = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7],
                                dtype=torch.int32,
                                device=device)
        tmp_topk = optimizer_tmp1.expert_mapping_frozen[0, ori_topk]
        checked_topk = torch.tensor([0, 1, 3, 4, 6, 7, 9, 2],
                                    dtype=torch.int32,
                                    device=device)

        self.assertTrue(torch.equal(checked_topk, tmp_topk), "device 分配错误")


if __name__ == "__main__":
    unittest.main()
