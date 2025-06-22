# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from .eplb_policy import EplbPolicy, DynamicConfig
from .mock_load_balance import MockLoadBalance
from .dynamic_ep import DynamicEplb
from .dynamic_ep_v2 import DynamicEplbV2



class PolicyFactory:
    @staticmethod
    def generate_policy(policy_type: int, config: DynamicConfig) -> EplbPolicy:
        policy = {
            """
                Constraint applying Dynamic EPLB policy V2:
                If there exists redundant expert,
                only one redundant expert can be placed in one NPU and its physical expert index must be 0
            """
            # Applying bipartite d2d expert weight update composing
            0:MockLoadBalance,     # MockLoadBalance
            1:DynamicEplb,         # Dynamic EPLB policy
            2:DynamicEplbV2,       # Dynamic EPLB policy V2

            # Applying greedy d2d expert weight update composing
            3:MockLoadBalance,   # MockLoadBalance
            4:DynamicEplb,       # Dynamic EPLB policy
            5:DynamicEplbV2,     # Dynamic EPLB policy V2
        }
        return policy.get(policy_type, MockLoadBalance)(config)
