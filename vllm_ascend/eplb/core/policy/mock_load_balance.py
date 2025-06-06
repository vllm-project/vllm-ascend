# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import copy
import random

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
            # 随机选两个卡
            idx1, idx2 = random.sample(range(num_card), 2)

            def find_last_valid(expert_list):
                for j in range(len(expert_list) - 1, -1, -1):
                    if expert_list[j] != -1:
                        return j
                return None

            pos1 = find_last_valid(new_table[layer_idx][idx1])
            pos2 = find_last_valid(new_table[layer_idx][idx2])

            if pos1 is not None and pos2 is not None:
                new_table[layer_idx][idx1][pos1], new_table[layer_idx][idx2][pos2] = (
                    new_table[layer_idx][idx2][pos2],
                    new_table[layer_idx][idx1][pos1]
                )
                
        return 1, [-i for i in range(num_layers)], new_table
