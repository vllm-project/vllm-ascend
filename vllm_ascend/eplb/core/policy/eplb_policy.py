# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from abc import abstractmethod


class DynamicConfig:
    placement_policy = None

    max_transferred_expert_per_layer = 100
    # 一台机器上，一层最多搬运多少专家

    ep_worldsize = 64  # 整个集群上所有的专家分布在多少个die上
    num_die_per_host = 8  # 每台机器上有几个die


class EplbPolicy:
    def __init__(self, config: DynamicConfig):
        self.config = config

    @abstractmethod
    def rebalance_experts(self, current_expert_table, expert_workload):
        """
        传入weight并返回相关限制条件下的专家复制和放置
        INPUT:
        current_expert_table: [layerId, rankId, expert_num_i]
        expert_workload = expert_table[layer0][rankId][expert_num_i]

        RETURNED: (res, expert_table)
        res:
        1 -- table_changed
        0 -- not_changed

        expert_table: [layerId, rankId, expert_num_i]
        expert_num_i --- [0, MaxExpertPerRank]
        expertID = expert_table[layer0][rankId][expert_num_i]
        array_values:
        [0, 1, 2, 3, 248]
        [4, 5, 6, 7, 254]
        [8, 9, 10, 11, 71]
        ...
        [252, 253, 254, 255, 0]
        """
        pass
