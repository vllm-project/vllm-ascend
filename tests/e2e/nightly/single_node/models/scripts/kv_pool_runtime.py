import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import regex as re

from tests.e2e.common.kv_pool.config import (
    KVPoolConfig,
    MemcacheKVPoolConfig,
    MooncakeKVPoolConfig,
)

KV_POOL_READY_TIMEOUT_SECONDS = 300
KV_POOL_STOP_TIMEOUT_SECONDS = 10
MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO = 0.9
MOONCAKE_EVICTION_RATIO = 0.1
MOONCAKE_DEFAULT_KV_LEASE_TTL = 11000
MOONCAKE_LIBRARY_PATHS = (
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake",
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages",
)
SINGLE_NODE_POOL_HOST = "127.0.0.1"


def _replace_localhost(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _replace_localhost(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_localhost(item) for item in value]
    if isinstance(value, str):
        return value.replace("${LOCAL_IP}", SINGLE_NODE_POOL_HOST)
    return value


class SingleNodeKVPoolManager:
    """Manage one KV pool service for a single-node YAML test case."""

    def __init__(self, config: KVPoolConfig, case_name: str):
        self.config = config
        safe_case_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", case_name)
        self.runtime_dir = (Path(tempfile.gettempdir()) / "vllm_ascend_single_node_kv_pool" / safe_case_name).resolve()
        self.process: subprocess.Popen | None = None
        self.server_envs: dict[str, str] = {}

    def start(self) -> None:
        self._write_config()
        self._start_service()
        self._wait_ready()

    def _write_config(self) -> None:
        raise NotImplementedError

    def _start_service(self) -> None:
        raise NotImplementedError

    def _ready_ports(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _start_process(self, cmd: list[str], extra_env: dict[str, str]) -> None:
        env = {**os.environ, **extra_env}
        print(f"Starting {self.config.type} KV pool: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, env=env, start_new_session=True)

    def _wait_ready(self, timeout: int = KV_POOL_READY_TIMEOUT_SECONDS) -> None:
        pending_ports = set(self._ready_ports())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.config.type} service exited before becoming ready with code {self.process.returncode}"
                )
            for port in tuple(pending_ports):
                try:
                    with socket.create_connection((SINGLE_NODE_POOL_HOST, port), timeout=2):
                        pending_ports.remove(port)
                except OSError:
                    pass
            if not pending_ports:
                print(f"{self.config.type} KV pool is ready")
                return
            time.sleep(1)
        raise TimeoutError(f"Timed out waiting for {self.config.type} KV pool; ports={sorted(pending_ports)}")

    @staticmethod
    def _write_text_atomic(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)

    def cleanup(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.process = None
            return
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
            self.process.wait(timeout=KV_POOL_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
                self.process.wait(timeout=KV_POOL_STOP_TIMEOUT_SECONDS)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass
        finally:
            self.process = None

    def __enter__(self):
        try:
            self.start()
        except Exception:
            self.cleanup()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()


class SingleNodeMooncakeManager(SingleNodeKVPoolManager):
    def __init__(self, config: MooncakeKVPoolConfig, case_name: str):
        super().__init__(config, case_name)
        self.config = config
        self.config_path = self.runtime_dir / "mooncake.json"

    @property
    def mooncake_config(self) -> MooncakeKVPoolConfig:
        if not isinstance(self.config, MooncakeKVPoolConfig):
            raise TypeError("Mooncake manager requires MooncakeKVPoolConfig")
        return self.config

    @property
    def master_address(self) -> str:
        return f"{SINGLE_NODE_POOL_HOST}:{self.mooncake_config.master_port}"

    def _write_config(self) -> None:
        store_config = _replace_localhost(self.config.config)
        store_config["master_server_address"] = self.master_address
        self._write_text_atomic(
            self.config_path,
            json.dumps(store_config, indent=2, ensure_ascii=False),
        )
        self.server_envs = {
            "MOONCAKE_CONFIG_PATH": str(self.config_path),
            "MOONCAKE_MASTER": self.master_address,
        }

    def _start_service(self) -> None:
        cmd = [
            "mooncake_master",
            "--port",
            str(self.mooncake_config.master_port),
            "--metrics_port",
            str(self.mooncake_config.metrics_port),
            "--eviction_high_watermark_ratio",
            str(MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO),
            "--eviction_ratio",
            str(MOONCAKE_EVICTION_RATIO),
            "--default_kv_lease_ttl",
            str(MOONCAKE_DEFAULT_KV_LEASE_TTL),
        ]
        inherited_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        library_paths = [*MOONCAKE_LIBRARY_PATHS]
        if inherited_library_path:
            library_paths.append(inherited_library_path)
        self._start_process(
            cmd,
            {"LD_LIBRARY_PATH": os.pathsep.join(library_paths)},
        )

    def _ready_ports(self) -> tuple[int, ...]:
        return (self.mooncake_config.master_port,)


class SingleNodeMemcacheManager(SingleNodeKVPoolManager):
    def __init__(self, config: MemcacheKVPoolConfig, case_name: str):
        super().__init__(config, case_name)
        self.config = config
        self.meta_config_path = self.runtime_dir / "mmc-meta.conf"
        self.local_config_path = self.runtime_dir / "mmc-local.conf"

    @property
    def memcache_config(self) -> MemcacheKVPoolConfig:
        if not isinstance(self.config, MemcacheKVPoolConfig):
            raise TypeError("Memcache manager requires MemcacheKVPoolConfig")
        return self.config

    @staticmethod
    def _format_config(config: dict[str, Any]) -> str:
        def format_value(value: Any) -> str:
            return str(value).lower() if isinstance(value, bool) else str(value)

        return "".join(f"{key} = {format_value(value)}\n" for key, value in config.items())

    def _write_config(self) -> None:
        rendered_config = _replace_localhost(self.config.config)
        meta_config = dict(rendered_config["meta"])
        local_config = dict(rendered_config["local"])
        meta_service_url = f"tcp://{SINGLE_NODE_POOL_HOST}:{self.memcache_config.meta_service_port}"
        config_store_url = f"tcp://{SINGLE_NODE_POOL_HOST}:{self.memcache_config.config_store_port}"
        meta_config["ock.mmc.meta_service_url"] = meta_service_url
        meta_config["ock.mmc.meta_service.config_store_url"] = config_store_url
        local_config["ock.mmc.meta_service_url"] = meta_service_url
        local_config["ock.mmc.local_service.config_store_url"] = config_store_url
        self._write_text_atomic(self.meta_config_path, self._format_config(meta_config))
        self._write_text_atomic(self.local_config_path, self._format_config(local_config))
        self.server_envs = {"MMC_LOCAL_CONFIG_PATH": str(self.local_config_path)}

    def _start_service(self) -> None:
        self._start_process(
            [
                sys.executable,
                "-c",
                "from memcache_hybrid import MetaService; MetaService.main()",
            ],
            {"MMC_META_CONFIG_PATH": str(self.meta_config_path)},
        )

    def _ready_ports(self) -> tuple[int, ...]:
        return (
            self.memcache_config.meta_service_port,
            self.memcache_config.config_store_port,
        )


class NullSingleNodeKVPoolManager:
    def __init__(self):
        self.server_envs: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None


def create_single_node_kv_pool_manager(
    config: KVPoolConfig | None,
    case_name: str,
) -> SingleNodeKVPoolManager | NullSingleNodeKVPoolManager:
    if config is None:
        return NullSingleNodeKVPoolManager()
    if isinstance(config, MooncakeKVPoolConfig):
        return SingleNodeMooncakeManager(config, case_name)
    if isinstance(config, MemcacheKVPoolConfig):
        return SingleNodeMemcacheManager(config, case_name)
    raise TypeError(f"Unsupported KV pool config: {type(config).__name__}")
