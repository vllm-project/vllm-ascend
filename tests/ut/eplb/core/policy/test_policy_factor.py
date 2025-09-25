import pytest

from vllm_ascend.eplb.core.policy.policy_abstract import DynamicConfig
from vllm_ascend.eplb.core.policy.policy_dynamic_ep import DynamicEplb
from vllm_ascend.eplb.core.policy.policy_dynamic_ep_v2 import DynamicEplbV2
from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory
from vllm_ascend.eplb.core.policy.policy_random import RandomLoadBalance


@pytest.fixture
def dummy_config():
    return DynamicConfig()


@pytest.mark.parametrize("policy_type, expected_class", [
    (0, RandomLoadBalance),
    (1, DynamicEplb),
    (2, DynamicEplbV2),
    (999, RandomLoadBalance),
])
def test_generate_policy(policy_type, expected_class, dummy_config):
    policy_instance = PolicyFactory.generate_policy(policy_type, dummy_config)
    assert isinstance(policy_instance, expected_class)
