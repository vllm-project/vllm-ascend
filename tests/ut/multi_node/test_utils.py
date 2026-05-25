import pytest

from tests.e2e.nightly.multi_node.scripts import utils


def test_get_cluster_dns_list(monkeypatch):
    monkeypatch.setenv("LWS_LEADER_ADDRESS", "vllm-0.group.namespace")

    assert utils.get_cluster_dns_list(3) == [
        "vllm-0.group.namespace",
        "vllm-0-1.group.namespace",
        "vllm-0-2.group.namespace",
    ]


def test_resolve_cluster_ips_uses_cluster_hosts():
    raw_config = {"cluster_hosts": ["10.0.0.1", "10.0.0.2"]}

    assert utils.resolve_cluster_ips(raw_config, 2) == ["10.0.0.1", "10.0.0.2"]


def test_resolve_cluster_ips_detects_size_mismatch():
    raw_config = {"cluster_hosts": ["10.0.0.1"]}

    with pytest.raises(AssertionError, match="cluster_hosts size mismatch"):
        utils.resolve_cluster_ips(raw_config, 2)


def test_resolve_cluster_ips_uses_explicit_ips():
    raw_config = {"cluster_hosts": ["10.0.0.1"]}

    assert utils.resolve_cluster_ips(raw_config, 2, ["10.0.1.1", "10.0.1.2"]) == [
        "10.0.1.1",
        "10.0.1.2",
    ]


def test_resolve_current_node_index_from_env(monkeypatch):
    monkeypatch.setenv("LWS_WORKER_INDEX", "1")

    assert utils.resolve_current_node_index(["10.0.0.1", "10.0.0.2"]) == 1


def test_load_yaml_mapping_uses_base_path(tmp_path, monkeypatch):
    config_path = tmp_path / "case.yaml"
    config_path.write_text("model: test\nnum_nodes: 2\n", encoding="utf-8")
    monkeypatch.setenv("CONFIG_BASE_PATH", str(tmp_path))

    data = utils.load_yaml_mapping(
        "case.yaml",
        default_name="default.yaml",
        default_base_path="unused",
        description="test config",
    )

    assert data == {"model": "test", "num_nodes": 2}


def test_load_yaml_mapping_rejects_non_mapping(tmp_path):
    config_path = tmp_path / "case.yaml"
    config_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must be a mapping"):
        utils.load_yaml_mapping(
            str(config_path),
            default_name="default.yaml",
            default_base_path="unused",
            description="test config",
        )
