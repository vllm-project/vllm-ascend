import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.nightly.multi_node.external_dp.scripts import runtime
from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    ExternalDPConfig,
    ExternalDPConfigLoader,
    KVPoolConfig,
    RoutingConfig,
)


def _make_config(pool_type: str, pool_config: dict[str, Any]) -> ExternalDPConfig:
    return ExternalDPConfig(
        test_name="kv-pool-test",
        model="test/model",
        num_nodes=2,
        npu_per_node=1,
        cluster_hosts=None,
        cluster_ips=["192.0.2.10", "192.0.2.11"],
        routing=RoutingConfig(
            type="disaggregated_prefill",
            proxy_node_index=0,
            proxy_host="192.0.2.10",
            proxy_port=1999,
            proxy_script="proxy.py",
            groups={"prefiller": [0], "decoder": [1]},
        ),
        nodes=[],
        launch_templates=[],
        kv_pool=KVPoolConfig(type=pool_type, config=pool_config),
    )


def _mooncake_config() -> dict[str, Any]:
    return {
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "ascend",
        "device_name": "",
        "master_server_address": "wrong-host:1",
        "global_segment_size": "1GB",
        "local_hostname": "${LOCAL_IP}",
    }


def _memcache_config() -> dict[str, Any]:
    return {
        "meta": {"ock.mmc.log_level": "error"},
        "local": {
            "ock.mmc.log_level": "error",
            "ock.mmc.local_service.world_size": 256,
            "ock.mmc.local_service.protocol": "device_sdma",
            "ock.mmc.local_service.dram.size": "1GB",
        },
    }


def _mock_process_start(monkeypatch, launched: dict[str, object]):
    class FakeProcess:
        pid = 123
        returncode = None

        @staticmethod
        def poll():
            return None

    def fake_start_logged_process(cmd, env, log_file):
        launched.update(cmd=cmd, env=env, log_file=log_file)
        return FakeProcess()

    monkeypatch.setattr(runtime, "start_logged_process", fake_start_logged_process)


@pytest.mark.parametrize("pool_type", ["mooncake", "memcache"])
def test_parse_kv_pool_types(pool_type: str) -> None:
    pool_config = _mooncake_config() if pool_type == "mooncake" else _memcache_config()
    kv_pool = ExternalDPConfigLoader._parse_kv_pool(
        {"kv_pool": {"type": pool_type, "config": pool_config}}
    )

    assert kv_pool == KVPoolConfig(type=pool_type, config=pool_config)


def test_parse_kv_pool_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported kv_pool.type"):
        ExternalDPConfigLoader._parse_kv_pool(
            {"kv_pool": {"type": "unknown", "config": {}}}
        )


def test_validate_memcache_requires_meta_and_local_sections() -> None:
    config = _make_config("memcache", {"meta": {}})

    with pytest.raises(TypeError, match="kv_pool.config.local"):
        ExternalDPConfigLoader._validate_kv_pool(config)


def test_mooncake_manager_allocates_ports_and_starts_master(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config("mooncake", _mooncake_config())
    launched: dict[str, object] = {}
    terminated: list[int] = []
    ports = iter([41001, 41002])
    monkeypatch.setenv("LOG_PREFIX", str(tmp_path / "shared"))
    monkeypatch.setattr(runtime, "get_open_port", lambda: next(ports))
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(runtime, "terminate_process_tree", terminated.append)
    _mock_process_start(monkeypatch, launched)

    manager = runtime.create_kv_pool_manager(
        config=config,
        current_node_index=0,
        log_root=tmp_path,
    )
    with manager:
        assert isinstance(manager, runtime.ExternalDPMooncakeManager)
        generated = json.loads(manager.config_path.read_text(encoding="utf-8"))
        assert generated["master_server_address"] == "192.0.2.10:41001"
        assert generated["local_hostname"] == "192.0.2.10"
        assert manager.server_envs == {
            "MOONCAKE_CONFIG_PATH": str(manager.config_path),
            "MOONCAKE_MASTER": "192.0.2.10:41001",
        }
        assert launched["cmd"][0:5] == [
            "mooncake_master",
            "--port",
            "41001",
            "--metrics_port",
            "41002",
        ]

    assert terminated == [123]


def test_memcache_manager_allocates_ports_and_starts_meta_service(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config("memcache", _memcache_config())
    launched: dict[str, object] = {}
    ports = iter([42001, 42002])
    monkeypatch.setenv("LOG_PREFIX", str(tmp_path / "shared"))
    monkeypatch.setattr(runtime, "get_open_port", lambda: next(ports))
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(runtime, "terminate_process_tree", lambda _pid: None)
    _mock_process_start(monkeypatch, launched)

    manager = runtime.create_kv_pool_manager(
        config=config,
        current_node_index=0,
        log_root=tmp_path,
    )
    with manager:
        assert isinstance(manager, runtime.ExternalDPMemcacheManager)
        meta_config = manager.meta_config_path.read_text(encoding="utf-8")
        local_config = manager.local_config_path.read_text(encoding="utf-8")
        assert "ock.mmc.meta_service_url = tcp://192.0.2.10:42001" in meta_config
        assert "ock.mmc.meta_service.config_store_url = tcp://192.0.2.10:42002" in meta_config
        assert "ock.mmc.local_service.config_store_url = tcp://192.0.2.10:42002" in local_config
        assert manager.server_envs == {"MMC_LOCAL_CONFIG_PATH": str(manager.local_config_path)}
        assert launched["cmd"] == [
            runtime.sys.executable,
            "-c",
            "from memcache_hybrid import MetaService; MetaService.main()",
        ]
        assert launched["env"] == {"MMC_META_CONFIG_PATH": str(manager.meta_config_path)}


def test_non_primary_node_uses_shared_ports_without_starting_service(
    tmp_path: Path, monkeypatch
) -> None:
    shared_dir = tmp_path / "shared"
    coordination_file = shared_dir / "kv_pool_coord" / "kv-pool-test.json"
    coordination_file.parent.mkdir(parents=True)
    coordination_file.write_text(
        json.dumps({"type": "mooncake", "service": 43001, "auxiliary": 43002}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LOG_PREFIX", str(shared_dir))
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        runtime,
        "start_logged_process",
        lambda *args, **kwargs: pytest.fail("non-primary node must not start a KV pool service"),
    )

    manager = runtime.create_kv_pool_manager(
        config=_make_config("mooncake", _mooncake_config()),
        current_node_index=1,
        log_root=tmp_path,
    )
    with manager:
        assert isinstance(manager, runtime.ExternalDPMooncakeManager)
        generated = json.loads(manager.config_path.read_text(encoding="utf-8"))
        assert generated["master_server_address"] == "192.0.2.10:43001"
        assert generated["local_hostname"] == "192.0.2.11"
