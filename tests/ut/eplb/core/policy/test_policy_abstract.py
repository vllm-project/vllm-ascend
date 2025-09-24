# test_policy_abstract.py
from vllm_ascend.eplb.core.policy.policy_abstract import (DynamicConfig,
                                                          EplbPolicy)


class DummyPolicy(EplbPolicy):

    def rebalance_experts(self, current_expert_table, expert_workload):
        return 1, current_expert_table


def test_dynamic_config_attributes():
    config = DynamicConfig()
    assert config.placement_policy is None
    assert config.max_transferred_expert_per_layer == 100
    assert config.ep_worldsize == 64
    assert config.num_die_per_host == 8


def test_eplb_policy_init_and_method():
    config = DynamicConfig()
    policy = DummyPolicy(config)

    assert policy.config == config

    expert_table = [[0, 1, 2]]
    workload = [10]
    res, new_table = policy.rebalance_experts(expert_table, workload)

    assert res == 1
    assert new_table == expert_table
