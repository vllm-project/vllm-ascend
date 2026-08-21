import json
from contextlib import nullcontext
from pathlib import Path

from tests.e2e.common.kv_pool.config import (
    MemcacheKVPoolConfig,
    MooncakeKVPoolConfig,
)
from tests.e2e.nightly.single_node.models.scripts import kv_pool_runtime
from tests.e2e.nightly.single_node.models.scripts.single_node_config import (
    SingleNodeConfigLoader,
)


class FakeProcess:
    pid = 123
    returncode = None

    def __init__(self):
        self.stopped = False

    def poll(self):
        return 0 if self.stopped else None

    def terminate(self):
        self.stopped = True

    def kill(self):
        self.stopped = True

    def wait(self, timeout):
        self.stopped = True
        return 0


def _mock_runtime(monkeypatch, launched: dict[str, object]) -> None:
    def fake_popen(cmd, env, start_new_session):
        launched.update(cmd=cmd, env=env, start_new_session=start_new_session)
        return FakeProcess()

    monkeypatch.setattr(kv_pool_runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        kv_pool_runtime.socket,
        "create_connection",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(kv_pool_runtime.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(kv_pool_runtime.os, "killpg", lambda pgid, sig: None, raising=False)


def test_single_node_without_kv_pool_uses_noop_manager() -> None:
    manager = kv_pool_runtime.create_single_node_kv_pool_manager(None, "plain-case")

    with manager:
        assert manager.server_envs == {}


def test_single_node_config_loader_parses_mooncake() -> None:
    config = SingleNodeConfigLoader._parse_test_cases(
        [
            {
                "name": "mooncake-case",
                "model": "test/model",
                "envs": {"SERVER_PORT": "8000"},
                "server_cmd": ["--port", "$SERVER_PORT"],
                "kv_pool": {
                    "type": "mooncake",
                    "master_port": 50088,
                    "metrics_port": 50089,
                    "config": {"metadata_server": "P2PHANDSHAKE"},
                },
            }
        ]
    )[0]

    assert config.kv_pool == MooncakeKVPoolConfig(
        config={"metadata_server": "P2PHANDSHAKE"},
        master_port=50088,
        metrics_port=50089,
    )


def test_single_node_mooncake_manager(tmp_path: Path, monkeypatch) -> None:
    launched: dict[str, object] = {}
    _mock_runtime(monkeypatch, launched)
    monkeypatch.setattr(kv_pool_runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    manager = kv_pool_runtime.create_single_node_kv_pool_manager(
        MooncakeKVPoolConfig(
            config={
                "metadata_server": "P2PHANDSHAKE",
                "local_hostname": "${LOCAL_IP}",
            },
            master_port=50088,
            metrics_port=50089,
        ),
        "mooncake case",
    )

    with manager:
        assert isinstance(manager, kv_pool_runtime.SingleNodeMooncakeManager)
        generated = json.loads(manager.config_path.read_text(encoding="utf-8"))
        assert generated["master_server_address"] == "127.0.0.1:50088"
        assert generated["local_hostname"] == "127.0.0.1"
        assert manager.server_envs == {
            "MOONCAKE_CONFIG_PATH": str(manager.config_path),
            "MOONCAKE_MASTER": "127.0.0.1:50088",
        }
        assert launched["cmd"][0:5] == [
            "mooncake_master",
            "--port",
            "50088",
            "--metrics_port",
            "50089",
        ]


def test_single_node_memcache_manager(tmp_path: Path, monkeypatch) -> None:
    launched: dict[str, object] = {}
    _mock_runtime(monkeypatch, launched)
    monkeypatch.setattr(kv_pool_runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    manager = kv_pool_runtime.create_single_node_kv_pool_manager(
        MemcacheKVPoolConfig(
            config={
                "meta": {"ock.mmc.log_level": "error"},
                "local": {"ock.mmc.local_service.protocol": "device_sdma"},
            },
            meta_service_port=5000,
            config_store_port=6000,
        ),
        "memcache case",
    )

    with manager:
        assert isinstance(manager, kv_pool_runtime.SingleNodeMemcacheManager)
        meta_config = manager.meta_config_path.read_text(encoding="utf-8")
        local_config = manager.local_config_path.read_text(encoding="utf-8")
        assert "ock.mmc.meta_service_url = tcp://127.0.0.1:5000" in meta_config
        assert "ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:6000" in meta_config
        assert "ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6000" in local_config
        assert manager.server_envs == {"MMC_LOCAL_CONFIG_PATH": str(manager.local_config_path)}
        assert launched["env"]["MMC_META_CONFIG_PATH"] == str(manager.meta_config_path)
