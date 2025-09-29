# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import unittest
import torch
import torch_npu
import numpy as np
import sys
import os
import random
import time
from unittest.mock import MagicMock, patch

from ..omni_placement.cluster_status import ClusterStatus
from ..omni_placement.expert_mapping import ExpertMapping
from ..omni_placement.optim.token_balance_optimizer import TokenBalance

# device = torch.device('npu' if torch.npu.is_available() else 'cpu')
device = 'cpu'

class TestTokenBalance(unittest.TestCase):
    def setUp(self):
        expert_placement = np.load('./patterns/placement_pattern_3d_v3_naivebalance3_16devices_HE.npy').astype(np.int32)
        self.placement_pattern = torch.tensor(expert_placement, dtype=torch.int64, device=device)
        print('Testing....shape of ',self.placement_pattern.shape)

        expert_mapping = ExpertMapping('../patterns/placement_pattern_3d_v3_naivebalance3_16devices_HE.npy', device)
        self.cluster_status = ClusterStatus(config=None, expert_mapping=expert_mapping, rank=0)
        # 创建一个TokenBalance实例用于测试
        self.optimizer = TokenBalance(self.cluster_status)

        # 创建测试数据
        self.token_expert_id = torch.tensor([
            [0, 1, 2],  # 第一个token的专家ID
            [2, 0, 1],  # 第二个token的专家ID
            [1, 2, 0],  # 第三个token的专家ID
            [0, 2, 1],  # 第四个token的专家ID
        ])

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
                        expert_positions.append(torch.tensor((device_id, layer_id, ep_id), dtype=torch.int64, device=device))
                        count += 1

        # 验证position_mapping的计算是否正确
        # for device_id, layer_id, ep_id in expert_positions:
        tt_tt = 0
        for xx in expert_positions:
            device_id, layer_id, ep_id = xx[0], xx[1], xx[2]
            # 手动计算预期的position_mapping值
            expected_value = self.optimizer.expert_per_device * device_id + torch.sum(self.placement_pattern[device_id, layer_id, :ep_id]).item()

            # 开始计时
            start_time = time.time()

            # 执行需要计时的代码
            # actual_value = self.optimizer.position_mapping[device_id, layer_id, ep_id]
            actual_value = self.optimizer.expert_position[layer_id, ep_id]
            # 结束计时
            end_time = time.time()

            actual_value = actual_value.item()
            tt_tt += end_time - start_time

            self.assertEqual(expected_value, actual_value)

        print('Total time:', tt_tt)


    def test_optimize_function(self):
        # 设置随机种子以确保结果可重现
        # random.seed(42)
        # torch.manual_seed(42)
        # device = torch.device('npu' if torch.npu.is_available() else 'cpu')
        device = 'cpu'
        ep_per_device = self.optimizer.expert_per_device
        print(f"\nExperts per device: {ep_per_device}")

        # 设置测试参数
        num_tokens = 1000  # 输入tokens数量
        topk = 32          # 每个token选择的专家数

        # 随机生成layer_id (范围在0到expert_mapping.shape[1]-1之间)
        layer_id_moe = random.randint(0, self.placement_pattern.shape[1]-1)  # 注意API中layer_id从1开始
        print(f"\nRandom layer_id: {layer_id_moe}")

        # 随机生成origin_topk矩阵 (形状为num_tokens x topk)
        # 值的范围在-1到expert_mapping.shape[2]-1之间
        # origin_topk = torch.randint(-1, self.placement_pattern.shape[2], (num_tokens, topk))

        origin_topk = torch.randint(-1, self.placement_pattern.shape[2], (num_tokens, topk), device=device)
        print(f"\nRandom origin_topk shape: {origin_topk.shape}")

        # 创建虚拟的token和token_expert_scores (值不重要，因为在当前实现中未使用)
        input_token = torch.zeros(num_tokens)
        token_expert_scores = torch.rand(num_tokens, topk)

        # 调用optimize函数获取优化后的映射, 是位置id
        print("\nCalling optimize function...")
        layer_id_moe = torch.tensor(layer_id_moe, dtype=torch.int64, device=device)

        start_time = time.time()

        _, optimized_mapping, _ = self.optimizer.optimize(
            layer_id_moe, input_token, origin_topk.clone(), token_expert_scores, self.cluster_status
        )

        end_time = time.time()

        print('Total time:', end_time - start_time, optimized_mapping.shape)

        print(f"\nOptimized mapping shape: {optimized_mapping.shape}")

        # 逐个检查每个值是否满足条件
        for i in range(num_tokens):
            for j in range(topk):

                value_ori = origin_topk[i][j].item()

                pos_id = optimized_mapping[i][j].item()

                if pos_id==-1:  continue

                device_id = pos_id // ep_per_device
                pos_this_device = pos_id % ep_per_device

                self.assertEqual(self.placement_pattern[device_id, layer_id_moe, value_ori], 1)
                self.assertEqual(torch.sum(self.placement_pattern[device_id, layer_id_moe, :value_ori]), pos_this_device)


if __name__ == "__main__":
    unittest.main()