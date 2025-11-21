# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# import torch_npu
import unittest

import torch
import torch.nn.functional as F

from ..omni_placement.optim.ada_router_optimizer import AdaRouter


class TestAdaRouter(unittest.TestCase):

    def setUp(self):
        # 创建一个AdaRouter实例用于测试
        self.router = AdaRouter(cluster_status=None)
        self.router.count = 0  # 初始化count属性
        self.ex_num = 4
        # 创建测试数据
        self.token_expert_id = torch.tensor([
            [0, 1, 2],  # 第一个token的专家ID
            [2, 0, 1],  # 第二个token的专家ID
            [1, 2, 0],  # 第三个token的专家ID
            [0, 2, 1],  # 第四个token的专家ID
        ]).to(torch.int32)

        # 创建专家分数，按降序排列
        self.token_scores = torch.tensor([
            [0.6, 0.3, 0.1],  # 第一个token的专家分数
            [0.5, 0.3, 0.2],  # 第二个token的专家分数
            [0.7, 0.2, 0.1],  # 第三个token的专家分数
            [0.4, 0.4, 0.2],  # 第四个token的专家分数
        ])
        self.input = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])

    def test_optimize(self):
        """测试optimize方法, 包括select_experts_t的间接测试"""
        threshold = 0.8
        self.router._threshold = 1 - threshold

        layer_id = 0
        token = None

        # 调用optimize方法
        _, topk_ids, topk_weights = self.router.optimize(layer_id,
                                                         token,
                                                         self.token_expert_id,
                                                         self.token_scores,
                                                         cluster_status=None)

        token_expect_expert_id = torch.tensor([[0, 1, -1], [2, 0, -1],
                                               [1, -1, -1],
                                               [0, 2, -1]]).to(torch.int32)
        expected_topk_weights = topk_weights * (token_expect_expert_id > -1)
        expected_topk_weights = expected_topk_weights / (
            torch.sum(expected_topk_weights, dim=-1, keepdim=True) + 1e-12)
        self.assertTrue(torch.equal(topk_ids, token_expect_expert_id))
        self.assertEqual(topk_ids.dtype, token_expect_expert_id.dtype)
        torch.testing.assert_close(expected_topk_weights, topk_weights)

    def test_optimize_method(self):
        self.router._threshold = 0.2
        self.router._method = "fuckkk"  # 方法不存在
        layer_id = 0
        token = None
        _, topk_ids, _ = self.router.optimize(layer_id,
                                              token,
                                              self.token_expert_id,
                                              self.token_scores,
                                              cluster_status=None)
        self.assertTrue(torch.equal(topk_ids,
                                    self.token_expert_id))  # topk_ids不变


class TestSelectExpertsByThreshold(unittest.TestCase):

    def setUp(self):
        # 创建一个包含 _select_experts_by_threshold 方法的模拟类
        self.router = AdaRouter(cluster_status=None)
        self.router._threshold = 0.1

    def test_basic_functionality(self):
        """测试基本功能是否正常工作"""
        # 准备输入数据
        batch_size, seq_len, top_k = 2, 3, 4
        token_expert_ids = torch.tensor([
            [0, 1, 2, 3],  # 都保留
            [4, 5, 6, 7],  # 部分保留
            [8, 9, 10, 11],  # 部分保留
            [12, 13, 14, 15],  # 都保留
            [16, 17, 18, 19],  # 都不保留
            [20, 21, 22, 23],  # 部分保留
        ])

        token_scores = torch.rand((6, 4), dtype=torch.float32)
        token_scores = token_scores / torch.sum(
            token_scores, dim=-1, keepdim=True)

        # 调用函数
        new_scores, new_ids = self.router._select_experts_by_threshold(
            token_expert_ids, token_scores)

        # 断言输出形状与输入相同
        self.assertEqual(new_scores.shape, token_scores.shape)
        self.assertEqual(new_ids.shape, token_expert_ids.shape)

        # 检查掩码应用是否正确
        expected_mask = token_scores > self.router._threshold
        expected_ids = token_expert_ids * expected_mask + (~expected_mask) * -1
        torch.testing.assert_close(new_ids, expected_ids)

        # 检查权重是否正确归一化
        for i in range(len(token_scores)):
            if torch.any(token_scores[i] > self.router._threshold):
                # 如果该行有大于阈值的元素，检查归一化结果
                expected_row_sum = 1.0
                actual_row_sum = torch.sum(new_scores[i]).item()
                self.assertAlmostEqual(actual_row_sum,
                                       expected_row_sum,
                                       places=5)
            else:
                # 如果该行没有大于阈值的元素，所有权重应为0
                torch.testing.assert_close(new_scores[i],
                                           torch.zeros_like(new_scores[i]))

    def test_shape_and_dtype_ids(self):
        """测试不同形状和数据类型的输入"""
        for bs, sl, top_k in [(1, 1, 1), (2, 3, 4), (5, 10, 8)]:
            for dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
                # 创建输入数据
                token_expert_ids = torch.randint(0, 100,
                                                 (bs * sl, top_k)).to(dtype)
                token_scores = torch.rand((bs * sl, top_k),
                                          dtype=torch.float32)
                # 调用函数
                _, new_ids = self.router._select_experts_by_threshold(
                    token_expert_ids, token_scores)
                # 检查输出形状
                self.assertEqual(new_ids.shape, token_expert_ids.shape)
                # 检查输出数据类型
                self.assertEqual(new_ids.dtype, dtype)

    def test_shape_and_dtype_scores(self):
        """测试不同形状和数据类型的输入"""
        for bs, sl, top_k in [(1, 1, 1), (2, 3, 4), (5, 10, 8)]:
            for dtype in [torch.float32, torch.float64, torch.float16]:
                # 创建输入数据
                token_expert_ids = torch.randint(0, 100, (bs * sl, top_k)).to(
                    torch.int32)
                token_scores = torch.rand((bs * sl, top_k), dtype=dtype)
                # 调用函数
                new_scores, _ = self.router._select_experts_by_threshold(
                    token_expert_ids, token_scores)
                # 检查输出形状
                self.assertEqual(new_scores.shape, token_scores.shape)
                # 检查输出数据类型
                self.assertEqual(new_scores.dtype, dtype)

    def test_empty_inputs(self):
        """测试空输入的情况"""
        # 创建空张量
        token_expert_ids = torch.tensor([], dtype=torch.long).reshape(0, 4)
        token_scores = torch.tensor([], dtype=torch.float32).reshape(0, 4)

        # 调用函数
        new_scores, new_ids = self.router._select_experts_by_threshold(
            token_expert_ids, token_scores)

        # 检查输出形状
        self.assertEqual(new_scores.shape, (0, 4))
        self.assertEqual(new_ids.shape, (0, 4))

    def test_threshold_edge_cases(self):
        """测试阈值边界情况"""
        # 准备输入数据
        token_expert_ids = torch.tensor([[1, 2, 3, 4]])
        # 测试所有值等于阈值
        token_scores = torch.tensor([[0.25, 0.25, 0.25, 0.25]],
                                    dtype=torch.float32)
        self.router._threshold = 0.25
        new_scores, new_ids = self.router._select_experts_by_threshold(
            token_expert_ids, token_scores)
        torch.testing.assert_close(new_ids,
                                   torch.full_like(token_expert_ids,
                                                   -1))  # 所有ID都应被替换为-1

    def test_threshold_equals_zero(self):
        token_expert_ids = torch.tensor([[1, 2, 3, 4]])
        # 测试阈值为0的情况
        token_scores = torch.tensor([[0.4, 0.3, 0.2, 0.1]],
                                    dtype=torch.float32)
        self.router._threshold = 0.0
        new_scores, new_ids = self.router._select_experts_by_threshold(
            token_expert_ids, token_scores)
        torch.testing.assert_close(new_ids, token_expert_ids)  # 所有ID都应保留

    def test_normalization(self):
        """测试权重归一化"""
        # 准备输入数据
        token_expert_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8],
                                         [9, 10, 11, 12]])

        token_scores = torch.tensor(
            [
                [0.8, 0.7, 0.6, 0.5],  # 所有分数大于阈值
                [0.6, 0.4, 0.3, 0.2],  # 只有一个分数大于阈值
                [0.4, 0.3, 0.2, 0.1]  # 所有分数小于阈值
            ],
            dtype=torch.float32)

        self.router._threshold = 0.5
        new_scores, new_ids = self.router._select_experts_by_threshold(
            token_expert_ids, token_scores)

        # 第一行，所有元素应归一化
        expected_sum_1 = 1.0
        self.assertAlmostEqual(torch.sum(new_scores[0]).item(),
                               expected_sum_1,
                               places=5)

        # 第二行，只有一个元素大于阈值，因此归一化后该元素值为1
        expected_2 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        torch.testing.assert_close(new_scores[1], expected_2)

        # 第三行，所有元素小于阈值，因此所有元素应为0
        expected_3 = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        torch.testing.assert_close(new_scores[2], expected_3)

    def test_zero_sum_division(self):
        """测试除以零的情况（当所有分数都低于阈值时）"""
        token_expert_ids = torch.tensor([[1, 2, 3, 4]])
        token_scores = torch.tensor([[0.4, 0.3, 0.2, 0.1]],
                                    dtype=torch.float32)
        self.router._threshold = 0.5
        new_scores, new_ids = self.router._select_experts_by_threshold(
            token_expert_ids, token_scores)
        torch.testing.assert_close(new_ids,
                                   torch.full_like(token_expert_ids,
                                                   -1))  # 所有ID都应被替换为-1
        torch.testing.assert_close(new_scores,
                                   torch.zeros_like(token_scores))  # 所有分数都应为0


class TestUnpadding(unittest.TestCase):

    def test_filter_minus_one_values(self):
        """测试过滤掉等于-1的元素"""
        # 准备测试数据
        expanded_expert_idx = torch.tensor([-1, -1, -1, 2, 5, 3, 0])
        sorted_local_tokens = torch.tensor([100, 101, 102, 103, 104, 105, 106])

        # 执行被测试的代码
        mask_minus = expanded_expert_idx.eq(-1)  # 等价于 tensor == -1
        num_minus = mask_minus.sum()
        filtered_tokens = sorted_local_tokens[num_minus:]

        # 验证结果
        self.assertEqual(mask_minus.tolist(),
                         [True, True, True, False, False, False, False])
        self.assertEqual(num_minus.item(), 3)  # 有3个-1元素
        self.assertEqual(filtered_tokens.tolist(),
                         [103, 104, 105, 106])  # 应该剩下最后4个元素

    def test_no_minus_one_values(self):
        """测试没有-1元素的情况"""
        expanded_expert_idx = torch.tensor([2, 1, 5, 3, 0])
        sorted_local_tokens = torch.tensor([100, 101, 102, 103, 104])

        mask_minus = expanded_expert_idx.eq(-1)
        num_minus = mask_minus.sum()
        filtered_tokens = sorted_local_tokens[num_minus:]

        self.assertEqual(mask_minus.tolist(),
                         [False, False, False, False, False])
        self.assertEqual(num_minus.item(), 0)
        self.assertEqual(filtered_tokens.tolist(),
                         [100, 101, 102, 103, 104])  # 应保留所有元素

    def test_all_minus_one_values(self):
        """测试全部是-1元素的情况"""
        expanded_expert_idx = torch.tensor([-1, -1, -1, -1])
        sorted_local_tokens = torch.tensor([100, 101, 102, 103])

        mask_minus = expanded_expert_idx.eq(-1)
        num_minus = mask_minus.sum()
        filtered_tokens = sorted_local_tokens[num_minus:]

        self.assertEqual(mask_minus.tolist(), [True, True, True, True])
        self.assertEqual(num_minus.item(), 4)
        self.assertEqual(filtered_tokens.tolist(), [])  # 应该是空张量

    def test_empty_tensor(self):
        """测试空张量的情况"""
        expanded_expert_idx = torch.tensor([])
        sorted_local_tokens = torch.tensor([])

        mask_minus = expanded_expert_idx.eq(-1)
        num_minus = mask_minus.sum()
        filtered_tokens = sorted_local_tokens[num_minus:]

        self.assertEqual(mask_minus.tolist(), [])
        self.assertEqual(num_minus.item(), 0)
        self.assertEqual(filtered_tokens.tolist(), [])

    def test_different_tensor_types(self):
        """测试不同数据类型的张量"""
        # 测试浮点型
        expanded_expert_idx = torch.tensor([-1.0, -1.0, 2.0, 5.0, 3.0],
                                           dtype=torch.float)
        sorted_local_tokens = torch.tensor([100.0, 101.0, 102.0, 103.0, 104.0],
                                           dtype=torch.float)

        mask_minus = expanded_expert_idx.eq(-1)
        num_minus = mask_minus.sum()
        filtered_tokens = sorted_local_tokens[num_minus:]

        self.assertEqual(mask_minus.tolist(),
                         [True, True, False, False, False])
        self.assertEqual(num_minus.item(), 2)
        self.assertEqual(filtered_tokens.tolist(), [102.0, 103.0, 104.0])

    def test_multi_dimensional_tensor(self):
        """测试多维张量的情况"""
        expanded_expert_idx = torch.tensor([-1, 2, 3])
        sorted_local_tokens = torch.tensor([[1, 2], [3, 4], [5, 6]])

        mask_minus = expanded_expert_idx.eq(-1)
        num_minus = mask_minus.sum()
        filtered_tokens = sorted_local_tokens[num_minus:]

        self.assertEqual(mask_minus.sum().item(), 1)  # 总共有4个-1元素
        self.assertEqual(num_minus.item(), 1)
        # 假设按行裁剪，保留最后一行
        self.assertEqual(filtered_tokens.tolist(), [[3, 4], [5, 6]])


class TestPadding(unittest.TestCase):

    def test_pad_shape(self):
        # 创建一个示例输入张量
        # 假设形状为 [batch_size, channels, height, width]
        batch_size, channels, height, width = 2, 3, 4, 5
        input_tensor = torch.randn(batch_size, channels, height, width)
        # 定义填充参数
        num_minus = 2
        # 应用填充
        output_tensor = F.pad(input_tensor, (0, 0, num_minus, 0),
                              mode='constant',
                              value=0)
        # 检查输出形状
        expected_shape = (batch_size, channels, height + num_minus, width)
        self.assertEqual(output_tensor.shape, expected_shape)

    def test_pad_value(self):
        # 创建一个示例输入张量
        # 假设形状为 [batch_size, channels, height, width]
        batch_size, channels, height, width = 2, 3, 4, 5
        input_tensor = torch.randn(batch_size, channels, height, width)
        # 定义填充参数
        num_minus = 2
        # 应用填充
        output_tensor = F.pad(input_tensor, (0, 0, num_minus, 0),
                              mode='constant',
                              value=0)
        # 检查新添加的部分是否为0
        zeros_part = output_tensor[:, :, :num_minus, :]
        self.assertTrue(torch.all(zeros_part == 0))

    def test_unpad_part_equal(self):
        # 创建一个示例输入张量
        # 假设形状为 [batch_size, channels, height, width]
        batch_size, channels, height, width = 2, 3, 4, 5
        input_tensor = torch.randn(batch_size, channels, height, width)
        # 定义填充参数
        num_minus = 2
        # 应用填充
        output_tensor = F.pad(input_tensor, (0, 0, num_minus, 0),
                              mode='constant',
                              value=0)
        # 检查原始数据是否保持不变
        original_part = output_tensor[:, :, num_minus:, :]
        self.assertTrue(torch.all(torch.isclose(original_part, input_tensor)))


if __name__ == '__main__':
    unittest.main()
