# Copyright # Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
            # 随机选两个卡
            indices = random.sample(range(num_card), 2)

            # 交换冗余专家
            new_table[i][indices[0]][-1], new_table[i][indices[1]][-1] = (
                new_table[i][indices[1]][-1],
                new_table[i][indices[0]][-1]
            )
        return 1, [-i for i in range(num_layers)], new_tableHuawei Technologies Co., Ltd. 2023-2024. All rights reserved.

