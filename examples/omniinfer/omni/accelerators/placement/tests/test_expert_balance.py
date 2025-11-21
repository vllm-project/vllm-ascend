# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import unittest
import warnings

import numpy as np
import torch

from ..omni_placement.cluster_status import ClusterStatus
from ..omni_placement.expert_mapping import ExpertMapping
from ..omni_placement.optim.expert_balance_optimizer import \
    ExpertsBalanceOptimizer

# Suppress warnings
warnings.filterwarnings('ignore')
# device = torch.device('npu' if torch.npu.is_available() else 'cpu')


class TestExpertBalance(unittest.TestCase):

    def setUp(self):
        self.device = "npu"

    def test_select_redundant_2experts(self):
        em_lis = []
        clus_lis = []
        op_lis = []
        selected_ranks = [0, 1]
        count = 0
        placement_pattern = torch.tensor([[[1, 0, 1, 1]], [[0, 1, 1, 1]]])

        origin_topk = torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3]],
            dtype=torch.int32)
        true_optimized_mapping1 = torch.tensor(
            [[0, 3, 1, 2, 0, 3, 1, 2], [0, 3, 4, 5, 0, 3, 4, 5]],
            dtype=torch.int32,
            device=self.device)

        pattern_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "patterns/tmp.npy")
        np.save(pattern_path, placement_pattern)
        """构造pattern"""

        # 遍历被选中的 ranks
        for rank_i in selected_ranks:
            em_lis.append(ExpertMapping(pattern_path, self.device))
            # em_lis[count].placement_pattern = placement_pattern

            clus_lis.append(
                ClusterStatus(config=None, expert_mapping=em_lis[count]))
            op_lis.append(
                ExpertsBalanceOptimizer(clus_lis[count], origin_topk.shape[0],
                                        origin_topk.shape[1]))

            _, optimized_mapping1, _ = op_lis[count].optimize(
                0, placement_pattern, origin_topk, placement_pattern, None)
            self.assertTrue(
                torch.equal(optimized_mapping1, true_optimized_mapping1),
                "TopKID分配不正确")
            count += 1
        if os.path.exists(pattern_path):
            os.remove(pattern_path)

    def test_select_redundant_4experts(self):
        em_lis = []
        clus_lis = []
        op_lis = []
        selected_ranks = [0, 1]
        count = 0
        placement_pattern = torch.tensor([
            [[1, 0, 1, 1]],  #rank0 (0~2)
            [[0, 1, 1, 1]],  #rank1 (3~5)
            [[1, 0, 1, 1]],  #rank2 (6~8)
            [[0, 1, 1, 1]]  #rank3 (9~11)
        ])

        origin_topk = torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3],
             [0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3]],
            dtype=torch.int32)
        true_optimized_mapping1 = torch.tensor(
            [[0, 3, 1, 2, 0, 3, 1, 2], [6, 9, 4, 5, 6, 9, 4, 5],
             [0, 3, 7, 8, 0, 3, 7, 8], [6, 9, 10, 11, 6, 9, 10, 11]],
            dtype=torch.int32,
            device=self.device)

        pattern_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "patterns/tmp.npy")
        np.save(pattern_path, placement_pattern)
        """构造pattern"""

        # 遍历被选中的 ranks
        for rank_i in selected_ranks:
            em_lis.append(ExpertMapping(pattern_path, self.device))
            # em_lis[count].placement_pattern = placement_pattern

            clus_lis.append(
                ClusterStatus(config=None, expert_mapping=em_lis[count]))
            op_lis.append(
                ExpertsBalanceOptimizer(clus_lis[count], origin_topk.shape[0],
                                        origin_topk.shape[1]))

            _, optimized_mapping1, _ = op_lis[count].optimize(
                0, placement_pattern, origin_topk, placement_pattern, None)
            self.assertTrue(
                torch.equal(optimized_mapping1, true_optimized_mapping1),
                "TopKID分配不正确")
            count += 1
        if os.path.exists(pattern_path):
            os.remove(pattern_path)

    def test_select_redundant_expert_with_2_layers(self):
        em_lis = []
        clus_lis = []
        op_lis = []
        selected_ranks = [0, 1]
        count = 0
        placement_pattern = torch.tensor([[[1, 0, 1, 1], [0, 1, 1, 1]],
                                          [[0, 1, 1, 1], [1, 0, 1, 1]]])

        origin_topk = torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3]],
            dtype=torch.int32)
        true_optimized_mapping = torch.tensor(
            [[[0, 3, 1, 2, 0, 3, 1, 2], [0, 3, 4, 5, 0, 3, 4, 5]],
             [[3, 0, 1, 2, 3, 0, 1, 2], [3, 0, 4, 5, 3, 0, 4, 5]]],
            dtype=torch.int32,
            device=self.device)

        pattern_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "patterns/tmp.npy")
        np.save(pattern_path, placement_pattern)
        """构造pattern"""

        # 遍历被选中的 ranks
        for rank_i in selected_ranks:
            em_lis.append(ExpertMapping(pattern_path, self.device))
            # em_lis[count].placement_pattern = placement_pattern

            clus_lis.append(
                ClusterStatus(config=None, expert_mapping=em_lis[count]))
            op_lis.append(
                ExpertsBalanceOptimizer(clus_lis[count], origin_topk.shape[0],
                                        origin_topk.shape[1]))

            for layer in range(2):
                _, optimized_mapping1, _ = op_lis[count].optimize(
                    layer, placement_pattern, origin_topk, placement_pattern,
                    None)
                self.assertTrue(
                    torch.equal(optimized_mapping1,
                                true_optimized_mapping[layer]), "TopKID分配不正确")
            count += 1
        if os.path.exists(pattern_path):
            os.remove(pattern_path)


if __name__ == "__main__":
    unittest.main()
