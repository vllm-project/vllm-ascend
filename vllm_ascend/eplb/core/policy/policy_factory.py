# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Todo: Once https://github.com/vllm-project/vllm/pull/24069 is merged in vllm. Remove this factory.
from importlib import import_module

from vllm.logger import logger

from .policy_abstract import EplbPolicy


class PolicyFactory:
    @staticmethod
    def generate_policy(policy_type: int) -> EplbPolicy:
        policy: dict[int, tuple[str, str]] = {
            # Constraint applying Dynamic EPLB policy V2:
            # If there exists redundant expert:
            # only one redundant expert can be placed in one NPU and its physical expert index must be 0
            # Applying greedy d2d expert weight update composing
            # RandomLoadBalance: shuffle last physical expert on NPU 1 and 3
            0: (".policy_random", "RandomLoadBalance"),
            # Dynamic EPLB policy: overall expert replacement based on current moe load
            1: (".policy_default_eplb", "DefaultEplb"),
            # Dynamic EPLB policy V2: expert replacement with constrained number of expert shuffle
            2: (".policy_swift_balancer", "SwiftBalanceEplb"),
            # FlashLB EPLB policy: expert replacement based on Joint Optimization,
            # Multi-Shot Enhancement and Incremental Adjustment
            3: (".policy_flashlb", "FlashLB"),
        }
        policy_entry = policy.get(policy_type)
        fallback = policy_entry is None
        if policy_entry is None:
            policy_entry = policy[0]
        module_name, class_name = policy_entry
        policy_module = import_module(module_name, package=__package__)
        policy_class = getattr(policy_module, class_name)
        if fallback:
            logger.warning(
                "[eplb/policy] Unrecognized policy_type=%s, falling back to %s",
                policy_type,
                policy_class.__name__,
            )
        else:
            logger.info("[eplb/policy] Policy: %s (type=%s)", policy_class.__name__, policy_type)
        policy_instance = policy_class()
        if policy_type == 3:
            policy_module.warm_up()
        return policy_instance
