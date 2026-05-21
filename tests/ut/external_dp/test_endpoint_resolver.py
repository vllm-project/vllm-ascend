import pytest

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    EndpointResolver,
    ExternalDPConfigLoader,
)
from tests.ut.external_dp.conftest import GENERIC_EXTERNAL_DP_YAML, write_config


def test_resolve_generic_dp_endpoints(generic_config):
    endpoints = EndpointResolver(generic_config).resolve()
    assert len(endpoints) == 4
    assert [endpoint.role for endpoint in endpoints] == ["worker", "worker", "worker", "worker"]
    assert [(endpoint.node_index, endpoint.local_rank) for endpoint in endpoints] == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_resolve_pd_endpoints(pd_config):
    endpoints = EndpointResolver(pd_config).resolve()
    assert [endpoint.role for endpoint in endpoints] == ["prefiller", "prefiller", "decoder", "decoder"]


def test_dp_rank_auto_increment(generic_config):
    endpoints = EndpointResolver(generic_config).resolve()
    assert [endpoint.dp_rank for endpoint in endpoints] == [0, 1, 2, 3]


def test_port_auto_increment(generic_config):
    endpoints = EndpointResolver(generic_config).resolve()
    assert [endpoint.port for endpoint in endpoints] == [7100, 7101, 7100, 7101]


def test_visible_devices_auto_increment(generic_config):
    endpoints = EndpointResolver(generic_config).resolve()
    assert [endpoint.visible_devices for endpoint in endpoints] == ["0", "1", "0", "1"]


def test_detect_device_overflow(tmp_path):
    content = GENERIC_EXTERNAL_DP_YAML.replace("npu_per_node: 16", "npu_per_node: 1")
    with pytest.raises(ValueError, match="uses 2 NPUs"):
        ExternalDPConfigLoader.from_yaml(str(write_config(tmp_path, content)))


def test_detect_invalid_dp_rank_range(tmp_path):
    content = GENERIC_EXTERNAL_DP_YAML.replace("dp_rank_start: 2", "dp_rank_start: 3")
    with pytest.raises(ValueError, match="dp rank range exceeds"):
        ExternalDPConfigLoader.from_yaml(str(write_config(tmp_path, content)))
