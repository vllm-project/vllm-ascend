# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from .eplb_policy import EplbPolicy, DynamicConfig
from .mock_load_balance import MockLoadBalance
from .mock_dynamic_ep import DynamicEP



class PolicyFactory:
    @staticmethod
    def generate_policy(policy_type: int, config: DynamicConfig) -> EplbPolicy:
        policy = {
            0:MockLoadBalance ,  # MockLoadBalance
            1:DynamicEP,  # DynamicEP
        }
        return policy.get(policy_type, MockLoadBalance)(config)
