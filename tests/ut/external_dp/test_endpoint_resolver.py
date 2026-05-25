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
    assert [(endpoint.config_index, endpoint.local_rank) for endpoint in endpoints] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


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


def test_visible_devices_include_tp_cp_sp_pp(tmp_path):
    content = GENERIC_EXTERNAL_DP_YAML.replace(
        "    tp_size: 1\n    pp_size: 1\n",
        "    tp_size: 2\n    cp_size: 2\n    sp_size: 2\n    pp_size: 1\n",
    )
    config = ExternalDPConfigLoader.from_yaml(str(write_config(tmp_path, content)))
    endpoints = EndpointResolver(config).resolve()
    assert [endpoint.visible_devices for endpoint in endpoints] == [
        "0,1,2,3,4,5,6,7",
        "8,9,10,11,12,13,14,15",
        "0,1,2,3,4,5,6,7",
        "8,9,10,11,12,13,14,15",
    ]


def test_parallel_sizes_default_to_one(tmp_path):
    content = GENERIC_EXTERNAL_DP_YAML.replace(
        "    dp_group: default\n"
        "    dp_size: 4\n"
        "    dp_size_local: 2\n"
        "    dp_rank_start: 0\n"
        "    tp_size: 1\n"
        "    pp_size: 1\n"
        '    dp_address: "${NODE_0_IP}"\n',
        '    dp_group: default\n    dp_address: "${NODE_0_IP}"\n',
        1,
    )
    config = ExternalDPConfigLoader.from_yaml(str(write_config(tmp_path, content)))
    node = config.node_configs[0]
    assert (node.dp_size, node.dp_size_local, node.tp_size, node.cp_size, node.sp_size, node.pp_size) == (
        1,
        1,
        1,
        1,
        1,
        1,
    )


def test_detect_device_overflow(tmp_path):
    content = GENERIC_EXTERNAL_DP_YAML.replace("npu_per_node: 16", "npu_per_node: 1")
    with pytest.raises(ValueError, match="uses 2 NPUs"):
        ExternalDPConfigLoader.from_yaml(str(write_config(tmp_path, content)))


def test_detect_invalid_dp_rank_range(tmp_path):
    content = GENERIC_EXTERNAL_DP_YAML.replace("dp_rank_start: 2", "dp_rank_start: 3")
    with pytest.raises(ValueError, match="dp rank range exceeds"):
        ExternalDPConfigLoader.from_yaml(str(write_config(tmp_path, content)))
