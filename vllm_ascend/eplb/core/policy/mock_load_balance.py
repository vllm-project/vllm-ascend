# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import copy
import random
import torch

from .eplb_policy import EplbPolicy, DynamicConfig

random.seed(42)

class MockLoadBalance(EplbPolicy):
    def __init__(self, config: DynamicConfig):
        super().__init__(config)

    def rebalance_experts(self, current_expert_table, expert_workload):
        new_table = copy.deepcopy(current_expert_table)
        num_layers = len(current_expert_table)
        num_card = len(current_expert_table[0])

        for i in range(num_layers):
            # select two NPUs randomly
            idx1, idx2 = random.sample(range(num_card), 2)

            # exchange physical expert 0 on selected NPU pair
            pos1 = torch.where(new_table[layer_idx][idx1] == 0)
            pos2 = torch.where(new_table[layer_idx][idx2] == 0)

            if pos1 is not None and pos2 is not None:
                new_table[layer_idx][idx1][pos1] = -1
                new_table[layer_idx][idx1][pos2] = 0
                new_table[layer_idx][idx2][pos2] = -1
                new_table[layer_idx][idx2][pos1] = 0

        return 1, [-i for i in range(num_layers)], new_table
