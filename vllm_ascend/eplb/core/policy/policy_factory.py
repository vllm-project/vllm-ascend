# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from .eplb_policy import EplbPolicy, DynamicConfig
from .mock_load_balance import MockLoadBalance
from .dynamic_ep import DynamicEP



class PolicyFactory:
    @staticmethod
    def generate_policy(policy_type: int, config: DynamicConfig) -> EplbPolicy:
        policy = {
            0:MockLoadBalance ,  # MockLoadBalance with greedy d2d expert weight composing
            1:DynamicEP,         # Dynamic EPLB policy with greedy d2d expert weight composing
            2:DynamicEP,         # Dynamic EPLB policy with bipartite d2d expert weight composing
        }
        return policy.get(policy_type, MockLoadBalance)(config)
