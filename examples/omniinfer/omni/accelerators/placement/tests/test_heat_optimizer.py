# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import unittest
import torch
import torch_npu
import numpy as np
import sys
import os
import random
from unittest.mock import MagicMock, patch
import time
from ..omni_placement.cluster_status import ClusterStatus
from ..omni_placement.expert_mapping import ExpertMapping
from ..omni_placement.optim.heat_optimizer import HEAT_ExpertsBalancer
import warnings
from pathlib import Path
import random

def _convert_pattern_path(path: str) -> str:
    # Check if the path is a relative path
    if not os.path.isabs(path):
        # Get the absolute path of the current script file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Create a Path object and resolve the relative path
        absolute_path = (Path(current_dir) / path).resolve()
        return str(absolute_path)
    else:
        # If it's already an absolute path, return it directly
        return path
# Suppress warnings
warnings.filterwarnings('ignore')
# device = torch.device('npu' if torch.npu.is_available() else 'cpu')
device = 'cpu'
class TestTokenBalance(unittest.TestCase):
    def setUp(self):
        self.path = '../patterns/DSV3_SupRXFlex_+15_58_MoELayers_64_dies_32_devices.npy'
        self.path2 = '../patterns/DSV3_SupRXFlex_+15_58_MoELayers_64_dies_32_devices.npy'

        # 策略 1
        self.path2  = _convert_pattern_path(self.path2)
        expert_placement2 = np.load(self.path2).astype(np.int32)
        self.placement_pattern2 = torch.tensor(expert_placement2, dtype=torch.int64, device=device)
        print('Testing....shape of ',self.placement_pattern2.shape)
        expert_mapping2 = ExpertMapping(self.path2, device)
        self.cluster_status2 = ClusterStatus(config=None, expert_mapping=expert_mapping2)
        # 创建一个TokenBalance实例用于测试
        self.optimizer_DEVSPEC = HEAT_ExpertsBalancer(self.cluster_status2, rank=0,   num_device_per_host=8)



    def test_expert_mapping_initialization_DEVSPEC(self):
            """测试 placement_pattern 是否正确初始化"""
            self.assertTrue(torch.equal(self.optimizer_DEVSPEC.placement_pattern, self.placement_pattern2),
                            "Optimizer_DEVSPEC 的 placement_pattern 初始化不正确")



    def test_build_expert_mapping_to_origin_pos_DEVSPEC(self):
        """测试 expert_mapping_ 是否正确生成"""
        self.assertIsNotNone(self.optimizer_DEVSPEC.expert_mapping_, "expert_mapping_ 没有正确初始化")





    def test_optimize_correctness_DEVSPEC(self):
            """测试 optimize 方法的 token_expert_id 是否正确映射"""
            batch_size, seq_len, top_k = 4, 10, 2
            layer_idx_moe = 0

            token_expert_id = torch.randint(0, self.optimizer_DEVSPEC.placement_pattern.shape[2]-1, (batch_size, seq_len, top_k), device=device)
            token_expert_id_expected = token_expert_id.clone()

            token_expert_id_expected = self.optimizer_DEVSPEC.expert_mapping_[layer_idx_moe, token_expert_id]

            _, token_expert_id_out, _ = self.optimizer_DEVSPEC.optimize(layer_idx_moe, None, token_expert_id.clone(), None, self.cluster_status2)

            self.assertTrue(torch.equal(token_expert_id_out, token_expert_id_expected),
                            "optimize 方法未正确更新 token_expert_id")


    def test_empty_input_tokens_DEVSPEC(self):
            """测试 optimize_DEVSPEC 方法在空 tokens 输入时的行为"""
            layer_idx_moe = 0
            tokens = torch.tensor([], dtype=torch.float32, device=device)  # 空 tokens
            token_expert_id = torch.randint(0, self.optimizer_DEVSPEC.placement_pattern.shape[2], (1, 1, 1), device=device)
            token_scores = torch.randn((1, 1, 1), device=device)

            tokens_out, token_expert_id_out, token_scores_out = self.optimizer_DEVSPEC.optimize(
                layer_idx_moe, tokens, token_expert_id, token_scores, self.cluster_status2
            )

            self.assertEqual(tokens_out.shape[0], 0, "空 tokens 输入时应返回空 tokens")
            self.assertEqual(token_expert_id_out.shape, token_expert_id.shape, "空 tokens 不应影响 token_expert_id 形状")
            self.assertEqual(token_scores_out.shape, token_scores.shape, "空 tokens 不应影响 token_scores 形状")


    def test_single_token_case_DEVSPEC(self):
            """测试 optimize_DEVSPEC 方法在单个 token 输入时的行为"""
            layer_idx_moe = 0
            tokens = torch.rand((1, 1, 64), device=device)  # 1 个 token
            token_expert_id = torch.randint(0, self.optimizer_DEVSPEC.placement_pattern.shape[2], (1, 1, 1), device=device)
            token_scores = torch.rand((1, 1, 1), device=device)

            tokens_out, token_expert_id_out, token_scores_out = self.optimizer_DEVSPEC.optimize(
                layer_idx_moe, tokens, token_expert_id, token_scores, self.cluster_status2
            )

            self.assertEqual(tokens_out.shape, tokens.shape, "单 token 处理后形状应保持不变")
            self.assertEqual(token_expert_id_out.shape, token_expert_id.shape, "单 token 的 expert ID 应保持形状不变")
            self.assertEqual(token_scores_out.shape, token_scores.shape, "单 token 的 scores 应保持形状不变")


    def test_optimize_output_shape_consistency_DEVSPEC_0(self):
        # 设置测试参数
        device =  "cpu"
        topk = 4
        layer_id_moe = torch.randint(0, self.placement_pattern2.shape[1], (1,)).item()
        input_token = torch.randint(0, 1000, (1, 10), device=device)
        origin_topk = torch.randint(-1, self.placement_pattern2.shape[2], (1, topk), device=device)
        token_expert_scores = torch.rand((1, 10, topk), device=device)

        # 调用优化函数
        _, optimized_mapping, _ = self.optimizer_DEVSPEC.optimize(
            layer_id_moe, input_token, origin_topk, token_expert_scores, self.cluster_status2
        )

        # 测试形状一致性
        self.assertEqual(origin_topk.shape, optimized_mapping.shape,
                        "origin_topk和optimized_mapping的形状应该一致")


    def test_optimize_output_data_type_consistency_DEVSPEC(self):
        # 设置测试参数
        device =  "cpu"
        topk = 4
        layer_id_moe = torch.randint(0, self.placement_pattern2.shape[1], (1,)).item()
        input_token = torch.randint(0, 1000, (1, 10), device=device)
        origin_topk = torch.randint(-1, self.placement_pattern2.shape[2], (1, topk), device=device, dtype=torch.int)
        token_expert_scores = torch.rand((1, 10, topk), device=device)

        # 调用优化函数
        _, optimized_mapping, _ = self.optimizer_DEVSPEC.optimize(
            layer_id_moe, input_token, origin_topk, token_expert_scores, self.cluster_status2
        )

        # 测试数据类型一致性
        self.assertEqual(origin_topk.dtype, optimized_mapping.dtype,
                        "origin_topk和optimized_mapping的数据类型应该一致")



    def test_optimize_output_data_type_consistency_DEVSPEC_2(self):
        # 设置测试参数
        device =  "cpu"
        topk = 4
        layer_id_moe = torch.randint(0, self.placement_pattern2.shape[1], (1,)).item()
        input_token = torch.randint(0, 1000, (1, 10), device=device)
        origin_topk = torch.randint(-1, self.placement_pattern2.shape[2], (1, topk), device=device, dtype=torch.int)
        token_expert_scores = torch.rand((1, 10, topk), device=device)
        world_size, num_layers, num_eps = self.optimizer_DEVSPEC.placement_pattern.shape

        placement_pattern = self.optimizer_DEVSPEC.placement_pattern
        cumsum_local = torch.cumsum(placement_pattern, dim=2)
        local_indices = cumsum_local - 1
        experts_per_device_layer = torch.sum(placement_pattern, dim=2, dtype=torch.long)
        cumulative_experts_per_layer = torch.cumsum(experts_per_device_layer, dim=0)
        device_offset_layer = torch.zeros_like(cumulative_experts_per_layer)
        device_offset_layer[1:, :] = cumulative_experts_per_layer[:-1, :]

        for i in range(num_layers):
            device_offset_layer_here = cumulative_experts_per_layer[:,i]
            for j in range(num_eps):
                pos = self.optimizer_DEVSPEC.local_expert_mapping[i][j]
                for kk in range(world_size ):
                    if pos < device_offset_layer_here[kk]: break
                self.assertEqual(placement_pattern[kk][i][j] , 1,
                            "placement_pattern[kk][i][j] 应该有这个EP")
                self.assertEqual(   int( device_offset_layer[kk][i] +
                                         torch.sum(placement_pattern[kk][i][:j]) +0.5)
                                 , pos
                                 , "PoS应该正确~" )

    def test_optimize_function_DEVICESPECIFIC_3(self):

        world_size = self.placement_pattern2.shape[0]

        em_lis = []
        clus_lis = []
        op_lis = []

        random.seed(42)

        # 从所有 rank 中随机选择 10 个进行测试
        leng = min(10,world_size //3  )
        selected_ranks = random.sample(range(world_size), leng)
        count = 0
        # 遍历被选中的 ranks
        for rank_i in selected_ranks:
            em_lis.append( ExpertMapping(self.path2, device)  )
            clus_lis.append( ClusterStatus(config=None, expert_mapping=em_lis[count])  )
            op_lis.append( HEAT_ExpertsBalancer( clus_lis[count], rank=rank_i,   num_device_per_host=8))
            # 沿第一个维度（dim=0）求和
            sum_along_dim0 = op_lis[count].placement_pattern_super.sum(dim=0)
            # 判断是否全为 1（用 float 类型时要考虑数值精度问题）
            self.assertTrue( torch.allclose(sum_along_dim0, torch.ones_like(sum_along_dim0), atol=1e-6), "每个expert都有映射") # checked!
            count += 1



    def test_optimize_function_DEVICESPECIFIC_4(self):

        world_size = self.placement_pattern2.shape[0]
        em_lis = []
        clus_lis = []
        op_lis = []
        random.seed(42)

        selected_ranks = [0 ]
        count = 0
        # 遍历被选中的 ranks
        for rank_i in selected_ranks:
            em_lis.append( ExpertMapping(self.path2, device)  )

            em_lis[0].placement_pattern = torch.tensor([
                                                        [  [1,0,0       ] ,[ 0,0,1    ]  ],
                                                        [  [0,1,0       ] ,[ 0,1,0    ]  ],
                                                        [  [0,0,1       ] ,[ 1,0,0    ]  ]
                                                        ])

            em_lis[0].local_expert_mapping = torch.zeros(
                                                        len(em_lis[0].placement_pattern[0] ),  # num_of_devices
                                                        len(em_lis[0].placement_pattern[0][0]),
                                                        dtype = torch.int32
                                                        )

            clus_lis.append( ClusterStatus(config=None, expert_mapping=em_lis[count])  )
            op_lis.append( HEAT_ExpertsBalancer( clus_lis[count], rank=rank_i,   num_device_per_host=8))
            # 沿第一个维度（dim=0）求和
            sum_along_dim0 = op_lis[count].placement_pattern_super.sum(dim=0)
            # 判断是否全为 1（用 float 类型时要考虑数值精度问题）
            self.assertTrue( torch.allclose(sum_along_dim0, torch.ones_like(sum_along_dim0), atol=1e-6), "每个expert都有映射") # checked!
            count += 1

        local_expert_mapping = torch.tensor([[0,1,2],[2,1,0]])
        self.assertTrue(torch.equal(op_lis[0].local_expert_mapping, local_expert_mapping  ),
                            "Optimizer_DEVSPEC 的 placement_pattern 初始化不正确")


    def test_optimize_function_DEVICESPECIFIC_5(self):
        em_lis = []
        clus_lis = []
        op_lis = []
        selected_ranks = [0 ]
        count = 0
        # 遍历被选中的 ranks
        for rank_i in selected_ranks:
            em_lis.append( ExpertMapping(self.path2, device)  )

            em_lis[count].placement_pattern = torch.tensor([
                                                        [  [ 1,0,0,0,1 ] ,[ 0,0,1,1,1 ]  ],
                                                        [  [ 0,1,0,0,0 ] ,[ 0,1,0,0,0 ]  ],
                                                        [  [ 0,0,1,1,0 ] ,[ 1,0,0,0,0 ]  ]
                                                        ])

            em_lis[count].local_expert_mapping = torch.zeros(
                                                        len(em_lis[0].placement_pattern[0] ),  # num_of_devices
                                                        len(em_lis[0].placement_pattern[0][0]),
                                                        dtype = torch.int32
                                                        )

            clus_lis.append( ClusterStatus(config=None, expert_mapping=em_lis[count])  )
            op_lis.append( HEAT_ExpertsBalancer( clus_lis[count], rank=rank_i,   num_device_per_host=8))
            # 沿第一个维度（dim=0）求和
            sum_along_dim0 = op_lis[count].placement_pattern_super.sum(dim=0)
            # 判断是否全为 1（用 float 类型时要考虑数值精度问题）
            self.assertTrue( torch.allclose(sum_along_dim0, torch.ones_like(sum_along_dim0), atol=1e-6), "每个expert都有映射") # checked!
            count += 1

        local_expert_mapping = torch.tensor([[0,2,3,4,1],[4,3,0,1,2]])
        self.assertTrue(torch.equal(op_lis[0].local_expert_mapping, local_expert_mapping  ),
                            "Optimizer_DEVSPEC 的 placement_pattern 初始化不正确")


    def test_optimize_function_DEVICESPECIFIC_6(self):
        em_lis = []
        clus_lis = []
        op_lis = []
        selected_ranks = [0, 1, 2]
        count = 0
        # 遍历被选中的 ranks
        for rank_i in selected_ranks:
            em_lis.append( ExpertMapping(self.path2, device)  )

            em_lis[count].placement_pattern = torch.tensor([
                                                        [  [ 1,0,0,0,1 ], [ 1,0,0,0,1 ]   ],
                                                        [  [ 1,1,0,0,0 ], [ 0,1,0,0,0 ]   ],
                                                        [  [ 1,0,1,1,0 ], [ 0,0,1,1,0 ] ]
                                                        ])

            em_lis[count].local_expert_mapping = torch.zeros(
                                                        len(em_lis[0].placement_pattern[0] ),
                                                        len(em_lis[0].placement_pattern[0][0]),
                                                        dtype = torch.int32
                                                        )

            clus_lis.append( ClusterStatus(config=None, expert_mapping=em_lis[count])  )
            op_lis.append( HEAT_ExpertsBalancer( clus_lis[count], rank=rank_i,   num_device_per_host=8))
            # 沿第一个维度（dim=0）求和
            sum_along_dim0 = op_lis[count].placement_pattern_super.sum(dim=0)
            # 判断是否全为 1（用 float 类型时要考虑数值精度问题）
            self.assertTrue( torch.allclose(sum_along_dim0, torch.ones_like(sum_along_dim0), atol=1e-6), "每个expert都有映射") # checked!
            count += 1

        self.assertFalse(torch.equal(op_lis[0].local_expert_mapping, op_lis[1].local_expert_mapping  ),
                            "Optimizer_DEVSPEC 的 local_expert_mapping 初始化不正确")

        self.assertFalse(torch.equal(op_lis[1].local_expert_mapping, op_lis[2].local_expert_mapping  ),
                            "Optimizer_DEVSPEC 的 local_expert_mapping 初始化不正确")

        self.assertFalse(torch.equal(op_lis[0].local_expert_mapping, op_lis[2].local_expert_mapping  ),
                            "Optimizer_DEVSPEC 的 local_expert_mapping 初始化不正确")

        self.assertTrue(torch.equal(op_lis[0].local_expert_mapping[1], op_lis[1].local_expert_mapping[1]  ),
                            "Optimizer_DEVSPEC 的 local_expert_mapping 初始化不正确")

        self.assertTrue(torch.equal(op_lis[1].local_expert_mapping[1], op_lis[2].local_expert_mapping[1]  ),
                            "Optimizer_DEVSPEC 的 local_expert_mapping 初始化不正确")

        self.assertTrue(torch.equal(op_lis[0].local_expert_mapping[1], op_lis[2].local_expert_mapping[1]  ),
                            "Optimizer_DEVSPEC 的 local_expert_mapping 初始化不正确")



if __name__ == "__main__":
    unittest.main()






