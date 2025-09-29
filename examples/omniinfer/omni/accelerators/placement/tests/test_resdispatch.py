# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import tempfile
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
from ..omni_placement.optim.resdispatch_optimizer import ResDis_ExpertsBalancer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
# device = torch.device('npu' if torch.npu.is_available() else 'cpu')

class TestTokenBalance(unittest.TestCase):
    def setUp(self):
        self.device = "npu"
        self.path = './patterns/DSV3_RedOri_+1_16_devices_58_MoE_Layers_16_dies_per_host.npy'
        self.path2 = '../patterns/DSV3_RedOri_+1_16_devices_58_MoE_Layers_16_dies_per_host.npy'
        expert_placement = np.load(self.path).astype(np.int32)

        self.placement_pattern = torch.tensor(expert_placement, dtype=torch.int64, device= self.device)
        print('Testing....shape of ',self.placement_pattern.shape)

        expert_mapping = ExpertMapping("../patterns/DSV3_RedOri_+1_16_devices_58_MoE_Layers_16_dies_per_host.npy",  self.device)
        self.cluster_status = ClusterStatus(config=None, expert_mapping=expert_mapping)
        # 创建一个TokenBalance实例用于测试
        self.optimizer = ResDis_ExpertsBalancer(self.cluster_status, rank=0)

        # 创建一个TokenBalance实例用于测试
        self.optimizer_rand = ResDis_ExpertsBalancer(self.cluster_status, rank=0, is_rand_op= True )


    def test_expert_mapping_initialization(self):
            """测试 placement_pattern 是否正确初始化"""
            self.assertTrue(torch.equal(self.optimizer.placement_pattern, self.placement_pattern),
                            "Optimizer 的 placement_pattern 初始化不正确")

    def test_expert_mapping_initialization_DEVSPEC(self):
            """测试 placement_pattern 是否正确初始化"""
            self.assertTrue(torch.equal(self.optimizer_rand.placement_pattern, self.placement_pattern),
                            "optimizer_rand 的 placement_pattern 初始化不正确")



    def test_optimize_output_shapes(self):
        """测试 optimize 方法的输出形状是否正确"""
        batch_size, seq_len, hidden_dim = 4, 10, 64
        top_k = 2
        layer_idx_moe = 0

        tokens = torch.rand((batch_size, seq_len, hidden_dim), device= self.device)
        token_expert_id = torch.randint(0, self.optimizer.placement_pattern.shape[2]-1, (batch_size, seq_len, top_k), device= self.device)
        token_scores = torch.rand((batch_size, seq_len, top_k), device= self.device)

        tokens_out, token_expert_id_out, token_scores_out = self.optimizer.optimize(
            layer_idx_moe, tokens.clone(), token_expert_id.clone(), token_scores.clone(), []
        )

        self.assertEqual(tokens.shape, tokens_out.shape, "tokens 输出形状不匹配")
        self.assertEqual(token_expert_id.shape, token_expert_id_out.shape, "token_expert_id 输出形状不匹配")
        self.assertEqual(token_scores.shape, token_scores_out.shape, "token_scores 输出形状不匹配")

    def test_optimize_output_shapes_DEVSPEC(self):
        """测试 optimize 方法的输出形状是否正确"""
        batch_size, seq_len, hidden_dim = 4, 10, 64
        top_k = 2
        layer_idx_moe = 0

        tokens = torch.rand((batch_size, seq_len, hidden_dim), device= self.device)
        token_expert_id = torch.randint(0, self.optimizer.placement_pattern.shape[2]-1, (batch_size, seq_len, top_k), device= self.device)
        token_scores = torch.rand((batch_size, seq_len, top_k), device= self.device)

        tokens_out, token_expert_id_out, token_scores_out = self.optimizer_rand.optimize(
            layer_idx_moe, tokens.clone(), token_expert_id.clone(), token_scores.clone(), self.cluster_status
        )

        self.assertEqual(tokens.shape, tokens_out.shape, "tokens 输出形状不匹配")
        self.assertEqual(token_expert_id.shape, token_expert_id_out.shape, "token_expert_id 输出形状不匹配")
        self.assertEqual(token_scores.shape, token_scores_out.shape, "token_scores 输出形状不匹配")





    def test_empty_input_tokens(self):
            """测试 optimize 方法在空 tokens 输入时的行为"""
            layer_idx_moe = 0
            tokens = torch.tensor([], dtype=torch.float32, device= self.device)  # 空 tokens
            token_expert_id = torch.randint(0, self.optimizer.placement_pattern.shape[2], (1, 1, 1), device= self.device)
            token_scores = torch.randn((1, 1, 1), device= self.device)

            tokens_out, token_expert_id_out, token_scores_out = self.optimizer.optimize(
                layer_idx_moe, tokens, token_expert_id, token_scores, []
            )

            self.assertEqual(tokens_out.shape[0], 0, "空 tokens 输入时应返回空 tokens")
            self.assertEqual(token_expert_id_out.shape, token_expert_id.shape, "空 tokens 不应影响 token_expert_id 形状")
            self.assertEqual(token_scores_out.shape, token_scores.shape, "空 tokens 不应影响 token_scores 形状")

    def test_empty_input_tokens_DEVSPEC(self):
            """测试 optimize_DEVSPEC 方法在空 tokens 输入时的行为"""
            layer_idx_moe = 0
            tokens = torch.tensor([], dtype=torch.float32, device= self.device)  # 空 tokens
            token_expert_id = torch.randint(0, self.optimizer_rand.placement_pattern.shape[2], (1, 1, 1), device= self.device)
            token_scores = torch.randn((1, 1, 1), device= self.device)

            tokens_out, token_expert_id_out, token_scores_out = self.optimizer_rand.optimize(
                layer_idx_moe, tokens, token_expert_id, token_scores, self.cluster_status
            )

            self.assertEqual(tokens_out.shape[0], 0, "空 tokens 输入时应返回空 tokens")
            self.assertEqual(token_expert_id_out.shape, token_expert_id.shape, "空 tokens 不应影响 token_expert_id 形状")
            self.assertEqual(token_scores_out.shape, token_scores.shape, "空 tokens 不应影响 token_scores 形状")

    def test_single_token_case(self):
            """测试 optimize 方法在单个 token 输入时的行为"""
            layer_idx_moe = 0
            tokens = torch.rand((1, 1, 64), device= self.device)  # 1 个 token
            token_expert_id = torch.randint(0, self.optimizer.placement_pattern.shape[2], (1, 1, 1), device= self.device)
            token_scores = torch.rand((1, 1, 1), device= self.device)

            tokens_out, token_expert_id_out, token_scores_out = self.optimizer.optimize(
                layer_idx_moe, tokens, token_expert_id, token_scores, []
            )

            self.assertEqual(tokens_out.shape, tokens.shape, "单 token 处理后形状应保持不变")
            self.assertEqual(token_expert_id_out.shape, token_expert_id.shape, "单 token 的 expert ID 应保持形状不变")
            self.assertEqual(token_scores_out.shape, token_scores.shape, "单 token 的 scores 应保持形状不变")


    def test_single_token_case_DEVSPEC(self):
            """测试 optimize_DEVSPEC 方法在单个 token 输入时的行为"""
            layer_idx_moe = 0
            tokens = torch.rand((1, 1, 64), device= self.device)  # 1 个 token
            token_expert_id = torch.randint(0, self.optimizer_rand.placement_pattern.shape[2], (1, 1, 1), device= self.device)
            token_scores = torch.rand((1, 1, 1), device= self.device)

            tokens_out, token_expert_id_out, token_scores_out = self.optimizer_rand.optimize(
                layer_idx_moe, tokens, token_expert_id, token_scores, self.cluster_status
            )

            self.assertEqual(tokens_out.shape, tokens.shape, "单 token 处理后形状应保持不变")
            self.assertEqual(token_expert_id_out.shape, token_expert_id.shape, "单 token 的 expert ID 应保持形状不变")
            self.assertEqual(token_scores_out.shape, token_scores.shape, "单 token 的 scores 应保持形状不变")

    def test_invalid_expert_id(self):
            """测试 optimize 方法在 token_expert_id 超出范围时的行为"""
            layer_idx_moe = 0
            batch_size, seq_len, top_k = 4, 10, 2

            tokens = torch.rand((batch_size, seq_len, 64), device= self.device)
            token_expert_id = torch.full((batch_size, seq_len, top_k), fill_value=9999, dtype=torch.int64, device= self.device)  # 非法 ID
            token_scores = torch.rand((batch_size, seq_len, top_k), device= self.device)

            #with self.assertRaises(IndexError):
            # npu tensor no IndexError, IndexError can be found only when cpu tensor
            self.optimizer.optimize(layer_idx_moe, tokens, token_expert_id, token_scores, [])



    def test_matrix_contains_only_zero_and_one(self):
        # 加载矩阵
        expert_placement = np.load(self.path).astype(np.int32)

        # 打印矩阵形状信息，帮助调试
        print(f"矩阵形状: {expert_placement.shape}")

        # 方法1：使用numpy的unique函数检查唯一值
        unique_values = np.unique(expert_placement)
        self.assertTrue(
            np.array_equal(unique_values, np.array([0, 1])) or
            np.array_equal(unique_values, np.array([0])) or
            np.array_equal(unique_values, np.array([1])),
            f"矩阵包含0和1以外的值: {unique_values}"
        )

    def test_matrix_contains_only_zero_and_one_2(self):
        expert_placement = np.load(self.path).astype(np.int32)
        # 方法2：通过逻辑运算检查每个元素是0或1
        is_zero_or_one = np.logical_or(expert_placement == 0, expert_placement == 1)
        self.assertTrue(
            np.all(is_zero_or_one),
            "矩阵包含非0非1的元素"
        )

        # 测试转换为PyTorch张量后的结果
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        placement_pattern = torch.tensor(expert_placement, dtype=torch.int64, device= self.device)
        print(f"PyTorch张量形状: {placement_pattern.shape}")

        # 检查PyTorch张量中的唯一值
        torch_unique = torch.unique(placement_pattern).cpu().numpy()
        self.assertTrue(
            np.array_equal(torch_unique, np.array([0, 1])) or
            np.array_equal(torch_unique, np.array([0])) or
            np.array_equal(torch_unique, np.array([1])),
            f"PyTorch张量包含0和1以外的值: {torch_unique}"
        )




    def test_optimize_output_shape_consistency(self):
        # 设置测试参数
        # device =  "cpu"
        topk = 4
        layer_id_moe = torch.randint(0, self.placement_pattern.shape[1], (1,)).item()
        input_token = torch.randint(0, 1000, (1, 10), device= self.device)
        origin_topk = torch.randint(-1, self.placement_pattern.shape[2], (1, topk), device= self.device)
        token_expert_scores = torch.rand((1, 10, topk), device= self.device)

        # 调用优化函数
        _, optimized_mapping, _ = self.optimizer.optimize(
            layer_id_moe, input_token, origin_topk, token_expert_scores, self.cluster_status
        )

        # 测试形状一致性
        self.assertEqual(origin_topk.shape, optimized_mapping.shape,
                        "origin_topk和optimized_mapping的形状应该一致")


    def test_optimize_output_shape_consistency_DEVSPEC(self):
        # 设置测试参数
        # device =  "cpu"
        topk = 4
        layer_id_moe = torch.randint(0, self.placement_pattern.shape[1], (1,)).item()
        input_token = torch.randint(0, 1000, (1, 10), device= self.device)
        origin_topk = torch.randint(-1, self.placement_pattern.shape[2], (1, topk), device= self.device)
        token_expert_scores = torch.rand((1, 10, topk), device= self.device)

        # 调用优化函数
        _, optimized_mapping, _ = self.optimizer_rand.optimize(
            layer_id_moe, input_token, origin_topk, token_expert_scores, self.cluster_status
        )

        # 测试形状一致性
        self.assertEqual(origin_topk.shape, optimized_mapping.shape,
                        "origin_topk和optimized_mapping的形状应该一致")




    def test_optimize_output_data_type_consistency(self):
        # 设置测试参数
        # device =  "cpu"
        topk = 4
        layer_id_moe = torch.randint(0, self.placement_pattern.shape[1], (1,)).item()
        input_token = torch.randint(0, 1000, (1, 10), device= self.device)
        origin_topk = torch.randint(-1, self.placement_pattern.shape[2], (1, topk), device= self.device, dtype=torch.int)
        token_expert_scores = torch.rand((1, 10, topk), device= self.device)

        # 调用优化函数
        _, optimized_mapping, _ = self.optimizer.optimize(
            layer_id_moe, input_token, origin_topk, token_expert_scores, self.cluster_status
        )

        # 测试数据类型一致性
        self.assertEqual(origin_topk.dtype, optimized_mapping.dtype,
                        "origin_topk和optimized_mapping的数据类型应该一致")



    def test_optimize_output_data_type_consistency_DEVSPEC(self):
        # 设置测试参数
        # device =  "cpu"
        topk = 4
        layer_id_moe = torch.randint(0, self.placement_pattern.shape[1], (1,)).item()
        input_token = torch.randint(0, 1000, (1, 10), device= self.device)
        origin_topk = torch.randint(-1, self.placement_pattern.shape[2], (1, topk), device= self.device, dtype=torch.int)
        token_expert_scores = torch.rand((1, 10, topk), device= self.device)

        # 调用优化函数
        _, optimized_mapping, _ = self.optimizer_rand.optimize(
            layer_id_moe, input_token, origin_topk, token_expert_scores, self.cluster_status
        )

        # 测试数据类型一致性
        self.assertEqual(origin_topk.dtype, optimized_mapping.dtype,
                        "origin_topk和optimized_mapping的数据类型应该一致")




    def test_optimize_function(self):
        # 设置随机种子以确保结果可重现
        # random.seed(42)
        # torch.manual_seed(42)
        # device = torch.device('npu' if torch.npu.is_available() else 'cpu')
        # device = 'cpu'
        expert_sums = torch.sum(self.placement_pattern, dim=-1)  # 对最后一维求和
        ep_per_device = int(torch.max(expert_sums).item() + 0.5)  # 求所有和的最大值


        # 设置测试参数
        num_tokens = 100  # 输入tokens数量
        topk = 5          # 每个token选择的专家数

        # 随机生成layer_id (范围在0到expert_mapping.shape[1]-1之间)
        layer_id_moe = random.randint(0, self.placement_pattern.shape[1]-1)  # 注意API中layer_id从1开始
        print(f"\nRandom layer_id: {layer_id_moe}")

        # 随机生成origin_topk矩阵 (形状为num_tokens x topk)
        # 值的范围在-1到expert_mapping.shape[2]-1之间

        origin_topk = torch.randint(0, self.placement_pattern.shape[2]-1, (num_tokens, topk), device= self.device)
        print(f"\nRandom origin_topk shape: {origin_topk.shape}")

        # 创建虚拟的token和token_expert_scores (值不重要，因为在当前实现中未使用)
        input_token = torch.zeros(num_tokens)
        token_expert_scores = torch.rand(num_tokens, topk)

        # 调用optimize函数获取优化后的映射, 是位置id
        print("\nCalling optimize function...")
        layer_id_moe = torch.tensor(layer_id_moe, dtype=torch.int64, device= self.device)

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
                value_ori = origin_topk[i][j]
                pos_id = optimized_mapping[i][j]
                if pos_id==-1:
                    self.assertEqual( torch.sum(self.placement_pattern[:,layer_id_moe,value_ori]) , 0 )
                    continue
                device_id = pos_id // ep_per_device
                pos_this_device = pos_id % ep_per_device
                self.assertEqual(self.placement_pattern[device_id, layer_id_moe, value_ori], 1)
                self.assertEqual(torch.sum(self.placement_pattern[device_id, layer_id_moe, :value_ori]), pos_this_device)



    def test_large_input_performance(self):
            """测试 optimize 方法在大规模输入下的性能"""
            # device = 'cpu'

            batch_size, seq_len, hidden_dim = 1024, 512, 64  # 大输入
            top_k = 2
            layer_idx_moe = 0

            tokens = torch.rand((batch_size, seq_len, hidden_dim), device= self.device)
            token_expert_id = torch.randint(0, self.optimizer.placement_pattern.shape[2] -1, (batch_size, seq_len, top_k), device= self.device)
            token_scores = torch.rand((batch_size, seq_len, top_k), device= self.device)

            start_time = time.time()
            self.optimizer.optimize(layer_idx_moe, tokens, token_expert_id, token_scores, [])
            end_time = time.time()

            print(f"Large input performance test completed in {end_time - start_time:.4f} seconds")

            self.assertLess(end_time - start_time, 2.0, "optimize 方法在大规模输入下运行时间过长")

    def test_npu_device_compatibility(self):
            """测试 optimize 方法是否支持 CPU/NPU 运行"""
            test_device = 'npu'
            # if torch.npu.is_available():
            #     test_device = 'npu'
            batch_size, seq_len, hidden_dim = 4, 10, 64
            top_k = 2
            layer_idx_moe = 0

            tokens = torch.rand((batch_size, seq_len, hidden_dim), device=test_device)
            token_expert_id = torch.randint(0, self.optimizer.placement_pattern.shape[2] -1 , (batch_size, seq_len, top_k), device=test_device)
            token_scores = torch.rand((batch_size, seq_len, top_k), device=test_device)

            # 创建一个TokenBalance实例用于测试
            tmp_optimizer =  self.optimizer


            tokens_out, token_expert_id_out, token_scores_out = tmp_optimizer.optimize(
                layer_idx_moe, tokens, token_expert_id, token_scores, []
            )
            self.assertEqual(tokens_out.device, tokens.device, "tokens 输出设备错误")
            self.assertEqual(token_expert_id_out.device, token_expert_id.device, "token_expert_id 输出设备错误")
            self.assertEqual(token_scores_out.device, token_scores.device, "token_scores 输出设备错误")
    
    # def test_cpu_device_compatibility(self):
    #         """测试 optimize 方法是否支持 CPU/NPU 运行"""
    #         test_device = 'cpu'
    #         # if torch.npu.is_available():
    #         #     test_device = 'npu'
    #         batch_size, seq_len, hidden_dim = 4, 10, 64
    #         top_k = 2
    #         layer_idx_moe = 0

    #         tokens = torch.rand((batch_size, seq_len, hidden_dim), device=test_device)
    #         token_expert_id = torch.randint(0, self.optimizer.placement_pattern.shape[2] -1 , (batch_size, seq_len, top_k), device=test_device)
    #         token_scores = torch.rand((batch_size, seq_len, top_k), device=test_device)

    #         # 创建一个TokenBalance实例用于测试
    #         tmp_optimizer =  self.optimizer


    #         tokens_out, token_expert_id_out, token_scores_out = tmp_optimizer.optimize(
    #             layer_idx_moe, tokens, token_expert_id, token_scores, []
    #         )
    #         self.assertEqual(tokens_out.device, tokens.device, "tokens 输出设备错误")
    #         self.assertEqual(token_expert_id_out.device, token_expert_id.device, "token_expert_id 输出设备错误")
    #         self.assertEqual(token_scores_out.device, token_scores.device, "token_scores 输出设备错误")


    def test_optimize_function_topkid_unordered_nored(self):
        em_lis = []
        clus_lis = []
        op_lis = []
        selected_ranks = [0, 1]
        count = 0
        placement_pattern = torch.tensor([
                                            [  [ 0,1 ]   ],
                                            [  [ 1,0 ]   ]
                                            ])

        origin_topk = torch.tensor([[0, 0, 0, 0 , 1, 1, 1, 1]] , dtype=torch.int32)
        true_optimized_mapping1 = torch.tensor( [[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.int32,device=self.device)

        pattern_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"patterns/tmp.npy")
        np.save(pattern_path,placement_pattern)
        """构造pattern"""

        # 遍历被选中的 ranks
        for rank_i in selected_ranks:
            em_lis.append( ExpertMapping(pattern_path,  self.device)  )
            # em_lis[count].placement_pattern = placement_pattern

            clus_lis.append( ClusterStatus(config=None, expert_mapping=em_lis[count])  )
            op_lis.append( ResDis_ExpertsBalancer( clus_lis[count], rank=rank_i ) )

            _, optimized_mapping1, _ = op_lis[count].optimize(
            0, placement_pattern, origin_topk, placement_pattern, None
            )
            self.assertTrue(torch.equal(optimized_mapping1, true_optimized_mapping1  ),
                            "TopKID分配不正确")
            count += 1
        if os.path.exists(pattern_path):
            os.remove(pattern_path)

    def test_optimize_function_topkid_ordered_red(self):
        em_lis = []
        clus_lis = []
        op_lis = []
        selected_ranks = [0, 1]
        count = 0
        placement_pattern = torch.tensor([
                                            [  [ 1, 0, 1 ]    ],
                                            [  [ 0, 1, 1 ]   ]
                                            ])
        pattern_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"patterns/tmp.npy")
        np.save(pattern_path,placement_pattern)

        origin_topk = torch.tensor([[0, 0, 0, 0 , 1, 1, 1, 1, 2, 2, 2, 2]] , dtype=torch.int32)
        true_optimized_mapping1 = torch.tensor( [[0, 0, 0, 0 , 2, 2, 2, 2, 1, 3, 1, 3]], dtype=torch.int32, device=self.device)


        # 遍历被选中的 ranks
        for rank_i in selected_ranks:
            em_lis.append( ExpertMapping(pattern_path,  self.device)  )

            clus_lis.append( ClusterStatus(config=None, expert_mapping=em_lis[count])  )
            op_lis.append( ResDis_ExpertsBalancer( clus_lis[count], rank=rank_i ) )

            _, optimized_mapping1, _ = op_lis[count].optimize(
            0, placement_pattern, origin_topk, placement_pattern, self.cluster_status
            )
            print( 'optimized_mapping1: ', optimized_mapping1)
            self.assertTrue(torch.equal(optimized_mapping1, true_optimized_mapping1  ),
                            "TopKID分配不正确")
            count += 1
        if os.path.exists(pattern_path):
            os.remove(pattern_path)

    def test_optimize_function_topkid_ordered_nored(self):
        em_lis = []
        clus_lis = []
        op_lis = []
        selected_ranks = [0, 1]
        count = 0
        placement_pattern = torch.tensor([
                                            [  [ 0, 1, 1 ]    ],
                                            [  [ 1, 0, 1 ]   ]
                                            ]).numpy()

        pattern_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"patterns/tmp.npy")
        np.save(pattern_path,placement_pattern)

        origin_topk = torch.tensor([[0, 0, 0, 0 , 1, 1, 1, 1, 2, 2, 2, 2]] , dtype=torch.int32)
        true_optimized_mapping1 = torch.tensor( [[2, 2, 2, 2, 0, 0, 0, 0 , 1, 3, 1, 3]], dtype=torch.int32,device =self.device)


        # 遍历被选中的 ranks
        for rank_i in selected_ranks:
            em_lis.append( ExpertMapping(pattern_path,  self.device)  )
            # em_lis[count].placement_pattern = placement_pattern

            clus_lis.append( ClusterStatus(config=None, expert_mapping=em_lis[count])  )
            op_lis.append( ResDis_ExpertsBalancer( clus_lis[count], rank=rank_i ) )

            _, optimized_mapping1, _ = op_lis[count].optimize(
            0, placement_pattern, origin_topk, placement_pattern, self.cluster_status
            )
            print( 'optimized_mapping1: ', optimized_mapping1)
            self.assertTrue(torch.equal(optimized_mapping1, true_optimized_mapping1  ),
                            "TopKID分配不正确")
            count += 1

    def test_optimize_function_topkid_ordered_nored22(self):
        em_lis = []
        clus_lis = []
        op_lis = []
        selected_ranks = [0, 1]
        count = 0
        placement_pattern = torch.tensor([
                                            [  [ 1, 1, 1 ]    ],
                                            [  [ 1, 1, 1 ]   ]
                                            ])
        pattern_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"patterns/tmp.npy")
        np.save(pattern_path,placement_pattern)

        origin_topk = torch.tensor([[0, 0, 0, 0 , 1, 1, 1, 1, 2, 2, 2, 2]] , dtype=torch.int32)
        true_optimized_mapping1 = torch.tensor( [[0, 3, 0, 3, 1, 4, 1, 4, 2, 5, 2, 5]], dtype=torch.int32, device=self.device)


        # 遍历被选中的 ranks
        for rank_i in selected_ranks:
            em_lis.append( ExpertMapping(pattern_path,  self.device)  )

            clus_lis.append( ClusterStatus(config=None, expert_mapping=em_lis[count])  )
            op_lis.append( ResDis_ExpertsBalancer( clus_lis[count], rank=rank_i ) )

            _, optimized_mapping1, _ = op_lis[count].optimize(
            0, placement_pattern, origin_topk, placement_pattern, self.cluster_status
            )
            print( 'optimized_mapping1: ', optimized_mapping1.device)
            print( 'true_optimized_mapping1: ', true_optimized_mapping1.device)
            self.assertTrue(torch.equal(optimized_mapping1, true_optimized_mapping1  ),
                            "TopKID分配不正确")
            count += 1
        if os.path.exists(pattern_path):
            os.remove(pattern_path)

    def test_optimize_function_topkid_sample(self):
        em_lis = []
        clus_lis = []
        op_lis = []
        selected_ranks = [0, 1, 2, 3]
        count = 0
        placement_pattern = torch.tensor([
                                            [  [ 1,0,0,0,1 ] ,[ 1,0,1,1,1 ]  ],
                                            [  [ 0,1,0,0,1 ] ,[ 0,1,0,0,1 ]  ],
                                            [  [ 0,0,1,0,0 ] ,[ 1,0,0,0,1 ]  ],
                                            [  [ 0,0,0,1,0 ] ,[ 0,0,0,0,1 ]  ]
                                            ])
        pattern_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"patterns/tmp.npy")
        np.save(pattern_path,placement_pattern)

        origin_topk = torch.tensor([  [ 0, 0, 0, 0 , 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4 ,4  ]
                                        ])
        # without global max offset
        # true_optimized_mapping1 = torch.tensor([[0, 0, 0, 0, 2, 4, 5, 3, 1, 3, 1, 3, 1 ,3 ,1 ,3]], dtype=torch.int32, device=self.device)
        # true_optimized_mapping2 = torch.tensor([[0, 6, 0, 6, 4, 1, 2, 8, 3, 5, 7, 8, 3, 5, 7, 8]], dtype=torch.int32, device=self.device)

        # with global max offset
        true_optimized_mapping1 = torch.tensor([[0, 0, 0, 0, 4, 8, 12, 5, 1, 5, 1, 5, 1 , 5, 1, 5]], dtype=torch.int32, device=self.device)
        true_optimized_mapping2 = torch.tensor([[0, 8, 0, 8, 4, 1, 2, 12, 3, 5, 9, 12, 3, 5, 9, 12]], dtype=torch.int32, device=self.device)

        # 遍历被选中的 ranks
        for rank_i in selected_ranks:
            em_lis.append( ExpertMapping(pattern_path,  self.device)  )

            clus_lis.append( ClusterStatus(config=None, expert_mapping=em_lis[count])  )
            op_lis.append( ResDis_ExpertsBalancer( clus_lis[count], rank=rank_i ) )

            _, optimized_mapping1, _ = op_lis[count].optimize(
            0, placement_pattern, origin_topk, placement_pattern, self.cluster_status
            )

            _, optimized_mapping2, _ = op_lis[count].optimize(
            1, placement_pattern, origin_topk, placement_pattern, self.cluster_status
            )
            self.assertTrue(torch.equal(optimized_mapping1, true_optimized_mapping1  ),
                            "TopKID分配不正确")
            self.assertTrue(torch.equal(optimized_mapping2, true_optimized_mapping2  ),
                            "TopKID分配不正确")

            count += 1
        if os.path.exists(pattern_path):
            os.remove(pattern_path)






if __name__ == "__main__":
    unittest.main()
