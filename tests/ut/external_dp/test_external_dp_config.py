import pytest

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import ExternalDPConfigLoader
from tests.ut.external_dp.conftest import GENERIC_EXTERNAL_DP_YAML, write_config


def test_parse_generic_dp_yaml(generic_config):
    assert generic_config.test_name == "generic external dp unit"
    assert generic_config.model == "Qwen/Qwen3-0.6B"
    assert generic_config.routing.type == "generic_dp"
    assert generic_config.routing.proxy_host == "10.0.0.1"
    assert generic_config.node_configs[1].dp_rank_start == 2
    assert generic_config.node_configs[1].dp_address == "10.0.0.1"
    assert generic_config.benchmark_cases[0]["case_name"] == "perf"
    assert not hasattr(generic_config.node_configs[1], "node_index")
    assert not hasattr(generic_config, "env_common")
    assert not hasattr(generic_config, "config_common")
    assert not hasattr(generic_config, "benchmarks")


def test_parse_disaggregated_prefill_yaml(pd_config):
    assert pd_config.routing.type == "disaggregated_prefill"
    assert pd_config.routing.groups["prefiller"] == [0]
    assert pd_config.routing.groups["decoder"] == [1]


def test_config_template_length_mismatch(tmp_path):
    content = GENERIC_EXTERNAL_DP_YAML.replace(
        "  - envs:\n"
        "      <<: *env_common\n"
        '      ASCEND_RT_VISIBLE_DEVICES: "${VISIBLE_DEVICES}"\n'
        '      SERVER_PORT: "${PORT}"\n'
        "    server_cmd_template: *cmd\n",
        "",
    )
    with pytest.raises(AssertionError, match="templates size"):
        ExternalDPConfigLoader.from_yaml(str(write_config(tmp_path, content)))


def test_invalid_routing_type(tmp_path):
    content = GENERIC_EXTERNAL_DP_YAML.replace('type: "generic_dp"', 'type: "unknown_dp"')
    with pytest.raises(ValueError, match="Unsupported routing.type"):
        ExternalDPConfigLoader.from_yaml(str(write_config(tmp_path, content)))


def test_invalid_group_index(tmp_path):
    content = GENERIC_EXTERNAL_DP_YAML.replace("worker: [0, 1]", "worker: [0, 2]")
    with pytest.raises(ValueError, match="out of range"):
        ExternalDPConfigLoader.from_yaml(str(write_config(tmp_path, content)))


def test_yaml_node_index_must_equal_config_index(tmp_path):
    content = GENERIC_EXTERNAL_DP_YAML.replace("node_index: 1", "node_index: 0", 1)
    with pytest.raises(ValueError, match=r"config\[1\]\.node_index must equal 1"):
        ExternalDPConfigLoader.from_yaml(str(write_config(tmp_path, content)))


def test_cluster_hosts_size_mismatch(tmp_path):
    content = GENERIC_EXTERNAL_DP_YAML.replace("  - 10.0.0.2\n", "")
    with pytest.raises(AssertionError, match="cluster_hosts size mismatch"):
        ExternalDPConfigLoader.from_yaml(str(write_config(tmp_path, content)))
