# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from tests.ut.kvpp_utils import make_kvpp_config
from vllm_ascend.ascend_config import KVPPConfig


def config():
    c = make_kvpp_config(2)
    c.speculative_config = None
    c.model_config.get_num_layers = lambda _: 17
    c.use_v2_model_runner = False
    c.kv_transfer_config = SimpleNamespace(
        kv_connector="AscendStoreConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={
            "backend": "memcache",
            "use_layerwise": True,
            "layerwise_num_shared_buffers": 3,
        },
    )
    return c


@pytest.mark.parametrize("buffers,prefetch", [(3, 3), (4, 4), (3, 5), (5, 3)])
def test_accepts_larger_shared_buffers_and_prefetch(buffers, prefetch):
    c = config()
    c.kv_transfer_config.kv_connector_extra_config.update(
        layerwise_num_shared_buffers=buffers, layerwise_prefetch_layers=prefetch
    )
    KVPPConfig.from_vllm_config(c).validate(c)


def test_default_layerwise_layout():
    c = config()
    del c.kv_transfer_config.kv_connector_extra_config["layerwise_num_shared_buffers"]
    KVPPConfig.from_vllm_config(c).validate(c)


@pytest.mark.parametrize(
    "key,value",
    [
        ("layerwise_num_shared_buffers", 2),
        ("layerwise_num_shared_buffers", True),
        ("layerwise_prefetch_layers", 2),
        ("backend", "mooncake"),
    ],
)
def test_reject_unsupported_layout(key, value):
    c = config()
    c.kv_transfer_config.kv_connector_extra_config[key] = value
    with pytest.raises((ValueError, TypeError)):
        KVPPConfig.from_vllm_config(c).validate(c)


@pytest.mark.parametrize("case", ["consumer", "pcp", "pp", "v2", "mtp"])
def test_reject_unsupported_execution(case):
    c = config()
    if case == "consumer":
        c.kv_transfer_config.kv_role = "kv_consumer"
    if case == "pcp":
        c.parallel_config.prefill_context_parallel_size = 2
    if case == "pp":
        c.parallel_config.pipeline_parallel_size = 2
    if case == "v2":
        c.use_v2_model_runner = True
    if case == "mtp":
        c.speculative_config = SimpleNamespace(method="mtp")
    with pytest.raises(ValueError, match="KVPP layerwise offload"):
        KVPPConfig.from_vllm_config(c).validate(c)


def test_preserves_offload_independent_layers():
    c = config()
    c.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = [0, 4]
    KVPPConfig.from_vllm_config(c).validate(c)
