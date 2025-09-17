# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Todo: Once https://github.com/vllm-project/vllm/pull/24069 is merged in vllm. Remove this factory.
from .policy_abstract import DynamicConfig, EplbPolicy
from .policy_dynamic_ep import DynamicEplb
from .policy_dynamic_ep_v2 import DynamicEplbV2
from .policy_random import RandomLoadBalance


class PolicyFactory:

    @staticmethod
    def generate_policy(policy_type: int, config: DynamicConfig) -> EplbPolicy:
        policy = {
            # Constraint applying Dynamic EPLB policy V2:
            # If there exists redundant expert:
            # only one redundant expert can be placed in one NPU and its physical expert index must be 0

            # Applying greedy d2d expert weight update composing
            0:
            RandomLoadBalance,  # RandomLoadBalance: shuffle last physical expert on NPU 1 and 3
            1:
            DynamicEplb,  # Dynamic EPLB policy: overall expert replacement based on current moe load
            2:
            DynamicEplbV2,  # Dynamic EPLB policy V2:  expert replacement with constrained number of expert shuffle
        }
        return policy.get(policy_type, RandomLoadBalance)(config)
