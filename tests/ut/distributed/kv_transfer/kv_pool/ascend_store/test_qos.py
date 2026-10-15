# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.qos import KvQosPolicy, validate_qos_mode


@pytest.mark.parametrize("priority,qos", [(0, 0), (1, 3), (2, 7)])
def test_request_mapping(priority, qos):
    policy = KvQosPolicy.from_config({"priority_to_qos": {"0": 0, "1": 3, "2": 7}})
    assert policy.select(priority) == qos


@pytest.mark.parametrize("qos", [-1, 8, True, "3"])
def test_invalid_qos(qos):
    with pytest.raises(ValueError):
        KvQosPolicy.from_config({"priority_to_qos": {"0": qos}})


def test_metadata_missing_is_not_defaulted():
    policy = KvQosPolicy.from_config({"priority_to_qos": {"0": 0}})
    with pytest.raises(ValueError):
        policy.select(None)


def test_layerwise_requires_separate_implementation():
    with pytest.raises(ValueError):
        validate_qos_mode({"kv_qos": {"priority_to_qos": {"0": 0}}, "use_layerwise": True})
