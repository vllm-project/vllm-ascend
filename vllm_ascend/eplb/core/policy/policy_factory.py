# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Todo: Once https://github.com/vllm-project/vllm/pull/24069 is merged in vllm. Remove this factory.
from .policy_abstract import DynamicConfig, EplbPolicy
from .policy_default_eplb import DefaultEplb
from .policy_flashlb import FlashLB, warm_up
from .policy_random import RandomLoadBalance
from .policy_swift_balancer import SwiftBalanceEplb


class PolicyFactory:
    @staticmethod
    def generate_policy(policy_type: int, config: DynamicConfig) -> EplbPolicy:
        policy = {
            # Constraint applying Dynamic EPLB policy V2:
            # If there exists redundant expert:
            # only one redundant expert can be placed in one NPU and its physical expert index must be 0
            # Applying greedy d2d expert weight update composing
            0: RandomLoadBalance,  # RandomLoadBalance: shuffle last physical expert on NPU 1 and 3
            1: DefaultEplb,  # Dynamic EPLB policy: overall expert replacement based on current moe load
            # Dynamic EPLB policy V2: expert replacement with constrained number of expert shuffle
            2: SwiftBalanceEplb,
            # FlashLB EPLB policy: expert replacement based on Joint Optimization,
            # Multi-Shot Enhancement and Incremental Adjustment
            3: FlashLB,
        }
        policy_class = policy.get(policy_type, RandomLoadBalance)
        policy_instance = policy_class(config)
        if policy_type == 3:
            warm_up()
        return policy_instance
