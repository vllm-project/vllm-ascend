import json
import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

from tests.e2e.common.kv_pool.config import (
    MemcacheKVPoolConfig,
    MooncakeKVPoolConfig,
)
from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    ROUTING_DISAGGREGATED_PREFILL,
    ExternalDPConfig,
    NodeTemplate,
    RankInfo,
    replace_cluster_placeholders,
)
from tests.e2e.nightly.multi_node.external_dp.scripts.utils import (
    format_server_cmd,
    is_http_ready,
    start_logged_process,
    terminate_process_tree,
    wait_http_ready,
    wait_http_unready,
)
from tests.e2e.nightly.multi_node.scripts.utils import get_net_interface

logger = logging.getLogger(__name__)

SERVER_READY_TIMEOUT_SECONDS = 3600
KV_POOL_READY_TIMEOUT_SECONDS = 300
MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO = 0.9
MOONCAKE_EVICTION_RATIO = 0.1
MOONCAKE_DEFAULT_KV_LEASE_TTL = 11000
MOONCAKE_LIBRARY_PATHS = (
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake",
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages",
)
PRIMARY_NODE_INDEX = 0
TEMPLATE_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
ENV_VAR_RE = re.compile(r"(?<!\$)\$([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class ServerCommand:
    """Rendered command, env, and printable command line."""

    cmd: list[str]
    env: dict[str, str]
    display_cmd: str


RankProcess = tuple[subprocess.Popen, RankInfo, Path]


class ExternalDPKVPoolManager:
    """Common lifecycle for a KV pool service running on node 0."""

    def __init__(
        self,
        *,
        config: ExternalDPConfig,
        current_node_index: int,
        log_root: Path,
    ):
        self.config = config
        self.current_node_index = current_node_index
        self.log_root = log_root
        self.process: subprocess.Popen | None = None
        self.server_envs: dict[str, str] = {}

    @property
    def pool_type(self) -> str:
        if self.config.kv_pool is None:
            raise RuntimeError("KV pool is not configured")
        return self.config.kv_pool.type

    @property
    def pool_config(self) -> dict[str, Any]:
        if self.config.kv_pool is None:
            raise RuntimeError("KV pool is not configured")
        return self.config.kv_pool.config

    @property
    def service_host(self) -> str:
        return self.config.cluster_ips[PRIMARY_NODE_INDEX]

    def start(self) -> None:
        self._write_config()
        if self.current_node_index == PRIMARY_NODE_INDEX:
            self._start_service()
        self._wait_ready()

    def _write_config(self) -> None:
        raise NotImplementedError

    def _start_service(self) -> None:
        raise NotImplementedError

    def _ready_ports(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _wait_ready(self, timeout: int = KV_POOL_READY_TIMEOUT_SECONDS) -> None:
        pending_ports = set(self._ready_ports())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.pool_type} service exited before becoming ready with code {self.process.returncode}"
                )
            for port in tuple(pending_ports):
                try:
                    with socket.create_connection((self.service_host, port), timeout=2):
                        pending_ports.remove(port)
                except OSError:
                    pass
            if not pending_ports:
                logger.info("%s KV pool ready on node 0", self.pool_type)
                return
            time.sleep(1)
        raise TimeoutError(
            f"Timed out waiting for {self.pool_type} KV pool at {self.service_host}; ports={sorted(pending_ports)}"
        )

    @staticmethod
    def _write_text_atomic(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)

    def cleanup(self) -> None:
        if self.process is None:
            return
        logger.info("Stopping %s KV pool service pid=%d", self.pool_type, self.process.pid)
        terminate_process_tree(self.process.pid)
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


class ExternalDPMooncakeManager(ExternalDPKVPoolManager):
    """Materialize Mooncake config and manage one cluster-wide master."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_path = (self.log_root / f"node-{self.current_node_index}" / "runtime" / "mooncake.json").resolve()

    @property
    def mooncake_config(self) -> MooncakeKVPoolConfig:
        kv_pool = self.config.kv_pool
        if not isinstance(kv_pool, MooncakeKVPoolConfig):
            raise TypeError("Mooncake manager requires MooncakeKVPoolConfig")
        return kv_pool

    @property
    def master_address(self) -> str:
        return f"{self.service_host}:{self.mooncake_config.master_port}"

    def _write_config(self) -> None:
        local_ip = self.config.cluster_ips[self.current_node_index]
        store_config = replace_cluster_placeholders(
            self.pool_config,
            cluster_ips=self.config.cluster_ips,
            local_ip=local_ip,
            current_node_index=self.current_node_index,
        )
        store_config["master_server_address"] = self.master_address
        self._write_text_atomic(
            self.config_path,
            json.dumps(store_config, indent=2, ensure_ascii=False),
        )
        self.server_envs = {
            "MOONCAKE_CONFIG_PATH": str(self.config_path),
            "MOONCAKE_MASTER": self.master_address,
        }
        logger.info(
            "Generated Mooncake config for node %d: %s",
            self.current_node_index,
            self.config_path,
        )

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
        library_path_parts = [*MOONCAKE_LIBRARY_PATHS]
        if inherited_library_path:
            library_path_parts.append(inherited_library_path)
        env = {"LD_LIBRARY_PATH": os.pathsep.join(library_path_parts)}
        log_file = self.log_root / f"node-{self.current_node_index}" / "mooncake-master.log"
        self.process = start_logged_process(cmd, env, log_file)
        logger.info("Mooncake master launched at %s", self.master_address)

    def _ready_ports(self) -> tuple[int, ...]:
        return (self.mooncake_config.master_port,)


class ExternalDPMemcacheManager(ExternalDPKVPoolManager):
    """Materialize Memcache configs and manage one cluster-wide MetaService."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        runtime_dir = self.log_root / f"node-{self.current_node_index}" / "runtime"
        self.meta_config_path = (runtime_dir / "mmc-meta.conf").resolve()
        self.local_config_path = (runtime_dir / "mmc-local.conf").resolve()

    @property
    def memcache_config(self) -> MemcacheKVPoolConfig:
        kv_pool = self.config.kv_pool
        if not isinstance(kv_pool, MemcacheKVPoolConfig):
            raise TypeError("Memcache manager requires MemcacheKVPoolConfig")
        return kv_pool

    @staticmethod
    def _format_config(config: dict[str, Any]) -> str:
        def format_value(value: Any) -> str:
            if isinstance(value, bool):
                return str(value).lower()
            return str(value)

        return "".join(f"{key} = {format_value(value)}\n" for key, value in config.items())

    def _write_config(self) -> None:
        local_ip = self.config.cluster_ips[self.current_node_index]
        rendered_config = replace_cluster_placeholders(
            self.pool_config,
            cluster_ips=self.config.cluster_ips,
            local_ip=local_ip,
            current_node_index=self.current_node_index,
        )
        meta_config = dict(rendered_config["meta"])
        local_config = dict(rendered_config["local"])
        meta_service_url = f"tcp://{self.service_host}:{self.memcache_config.meta_service_port}"
        config_store_url = f"tcp://{self.service_host}:{self.memcache_config.config_store_port}"
        meta_config["ock.mmc.meta_service_url"] = meta_service_url
        meta_config["ock.mmc.meta_service.config_store_url"] = config_store_url
        local_config["ock.mmc.meta_service_url"] = meta_service_url
        local_config["ock.mmc.local_service.config_store_url"] = config_store_url
        self._write_text_atomic(self.meta_config_path, self._format_config(meta_config))
        self._write_text_atomic(self.local_config_path, self._format_config(local_config))
        self.server_envs = {"MMC_LOCAL_CONFIG_PATH": str(self.local_config_path)}
        logger.info("Generated Memcache configs for node %d", self.current_node_index)

    def _start_service(self) -> None:
        cmd = [
            sys.executable,
            "-c",
            "from memcache_hybrid import MetaService; MetaService.main()",
        ]
        env = {"MMC_META_CONFIG_PATH": str(self.meta_config_path)}
        log_file = self.log_root / f"node-{self.current_node_index}" / "memcache-meta-service.log"
        self.process = start_logged_process(cmd, env, log_file)
        logger.info("Memcache MetaService launched on node 0")

    def _ready_ports(self) -> tuple[int, ...]:
        return (
            self.memcache_config.meta_service_port,
            self.memcache_config.config_store_port,
        )


class NullKVPoolManager:
    """No-op context manager used when kv_pool is not configured."""

    def __init__(self):
        self.server_envs: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None


def create_kv_pool_manager(
    *,
    config: ExternalDPConfig,
    current_node_index: int,
    log_root: Path,
) -> ExternalDPKVPoolManager | NullKVPoolManager:
    kwargs = {
        "config": config,
        "current_node_index": current_node_index,
        "log_root": log_root,
    }
    if config.kv_pool is None:
        return NullKVPoolManager()
    if isinstance(config.kv_pool, MooncakeKVPoolConfig):
        return ExternalDPMooncakeManager(**kwargs)
    if isinstance(config.kv_pool, MemcacheKVPoolConfig):
        return ExternalDPMemcacheManager(**kwargs)
    raise TypeError(f"Unsupported KV pool config: {type(config.kv_pool).__name__}")


class ServerCommandBuilder:
    """Render rank templates into vLLM serve commands."""

    def __init__(self, config: ExternalDPConfig):
        self.config = config

    def build(self, rank: RankInfo, template: NodeTemplate) -> ServerCommand:
        variables = self._build_variables(rank)
        rendered_env = self._render_envs(template.envs, rank, variables)
        rendered_args = [
            self._render_string(
                arg,
                rank=rank,
                braced_variables=variables,
                unbraced_variables=rendered_env,
                allow_missing_unbraced=False,
            )
            for arg in template.server_cmd_template
        ]
        cmd = ["vllm", "serve", self.config.model, *rendered_args]

        env = {key: str(value) for key, value in rendered_env.items()}
        display_cmd = format_server_cmd(cmd, env)
        logger.info(
            "External DP server command node=%s rank=%s: %s",
            rank.node_index,
            rank.local_rank,
            display_cmd,
        )
        return ServerCommand(cmd=cmd, env=env, display_cmd=display_cmd)

    def build_all(self, ranks: list[RankInfo]) -> list[ServerCommand]:
        return [self.build(rank, self.config.launch_templates[rank.node_index]) for rank in ranks]

    def _build_variables(self, rank: RankInfo) -> dict[str, str]:
        return {
            "MODEL": self.config.model,
            "PORT_START": str(rank.port_start),
            "PORT": str(rank.port),
            "DP_SIZE": str(rank.dp_size),
            "DP_SIZE_LOCAL": str(rank.dp_size_local),
            "DP_RANK_START": str(rank.dp_rank - rank.local_rank),
            "DP_RANK": str(rank.dp_rank),
            "LOCAL_RANK": str(rank.local_rank),
            "TP_SIZE": str(rank.tp_size),
            "CP_SIZE": str(rank.cp_size),
            "SP_SIZE": str(rank.sp_size),
            "PP_SIZE": str(rank.pp_size),
            "DP_ADDRESS": rank.dp_address,
            "DP_RPC_PORT": str(rank.dp_rpc_port),
            "VISIBLE_DEVICES": rank.visible_devices,
            "NODE_INDEX": str(rank.node_index),
            "CONFIG_INDEX": str(rank.node_index),
        }

    def _render_envs(
        self,
        envs: dict[str, Any],
        rank: RankInfo,
        variables: dict[str, str],
    ) -> dict[str, str]:
        rendered_envs: dict[str, str] = {}
        for key, value in envs.items():
            if isinstance(value, str):
                value = self._render_string(
                    value,
                    rank=rank,
                    braced_variables=variables,
                    unbraced_variables={**os.environ, **rendered_envs},
                    allow_missing_unbraced=True,
                )
            rendered_envs[str(key)] = str(value)
        return rendered_envs

    def _render_string(
        self,
        value: str,
        *,
        rank: RankInfo,
        braced_variables: dict[str, str],
        unbraced_variables: dict[str, str],
        allow_missing_unbraced: bool,
    ) -> str:
        value = replace_cluster_placeholders(
            value,
            cluster_ips=self.config.cluster_ips,
            local_ip=rank.host,
            current_node_index=rank.node_index,
        )
        value = self._render_variables(
            value,
            braced_variables,
            pattern=TEMPLATE_VAR_RE,
            allow_missing=False,
        )
        return self._render_variables(
            value,
            unbraced_variables,
            pattern=ENV_VAR_RE,
            allow_missing=allow_missing_unbraced,
        )

    @staticmethod
    def _render_variables(
        value: str,
        variables: dict[str, str],
        *,
        pattern: re.Pattern[str],
        allow_missing: bool,
    ) -> str:
        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in variables:
                if allow_missing:
                    return ""
                raise KeyError(f"Unknown external DP template variable: {key}")
            return variables[key]

        return pattern.sub(repl, value)


class ExternalDPServerManager:
    """Start and stop the external DP ranks owned by the current node."""

    def __init__(
        self,
        *,
        config: ExternalDPConfig,
        ranks: list[RankInfo],
        current_node_index: int,
        log_root: Path,
        extra_envs: dict[str, str] | None = None,
    ):
        self.config = config
        self.ranks = ranks
        self.current_node_index = current_node_index
        self.log_root = log_root
        self.extra_envs = dict(extra_envs or {})
        self.command_builder = ServerCommandBuilder(config)
        self.dist_envs = build_dist_envs(
            config.cluster_ips[current_node_index],
            config.cluster_ips[0],
        )
        self.rank_processes: list[RankProcess] = []

    def start_current_node(self) -> None:
        local_ranks = [rank for rank in self.ranks if rank.node_index == self.current_node_index]
        logger.info("Starting %d external DP ranks on node %d", len(local_ranks), self.current_node_index)
        try:
            for rank in local_ranks:
                template = self.config.launch_templates[rank.node_index]
                template = type(template)(
                    envs={**template.envs, **self.dist_envs, **self.extra_envs},
                    server_cmd_template=template.server_cmd_template,
                )
                server_cmd = self.command_builder.build(rank, template)
                log_file = self._rank_log_file(rank)
                process = start_logged_process(server_cmd.cmd, server_cmd.env, log_file)
                self.rank_processes.append((process, rank, log_file))

            wait_ranks_ready(
                local_ranks,
                timeout=SERVER_READY_TIMEOUT_SECONDS,
                rank_processes=self.rank_processes,
            )
        except Exception:
            self.cleanup()
            raise

    def __enter__(self):
        self.start_current_node()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self) -> None:
        for process, rank, _log_file in reversed(self.rank_processes):
            logger.info(
                "Stopping external DP rank node=%d rank=%d pid=%d",
                rank.node_index,
                rank.local_rank,
                process.pid,
            )
            terminate_process_tree(process.pid)
        self.rank_processes.clear()

    def _rank_log_file(self, rank: RankInfo) -> Path:
        return self.log_root / f"node-{rank.node_index}" / f"rank-{rank.local_rank}.log"


class ExternalDPProxyLauncher:
    """Launch the external DP proxy on the configured proxy node."""

    def __init__(
        self,
        *,
        config: ExternalDPConfig,
        ranks: list[RankInfo],
        current_node_index: int,
        log_root: Path,
    ):
        self.config = config
        self.ranks = ranks
        self.current_node_index = current_node_index
        self.log_root = log_root
        self.pid: int | None = None

    def start(self) -> None:
        if self.current_node_index != self.config.routing.proxy_node_index:
            logger.info("Current node is not proxy node, skip launching external DP proxy")
            return

        cmd = build_proxy_server_cmd(self.config, self.ranks)
        log_file = self.log_root / f"node-{self.current_node_index}" / "proxy.log"
        process = start_logged_process(cmd, {}, log_file)
        self.pid = process.pid
        logger.info("External DP proxy launched: %s", proxy_server_health_url(self.config))

    def wait_ready(self, timeout: int = 300) -> None:
        wait_http_ready(proxy_server_health_url(self.config), timeout=timeout)
        logger.info("External DP proxy ready: %s", proxy_server_health_url(self.config))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self) -> None:
        if self.pid is None:
            return
        logger.info("Stopping external DP proxy pid=%d", self.pid)
        terminate_process_tree(self.pid)
        self.pid = None


def build_all_server_commands(config: ExternalDPConfig, ranks: list[RankInfo]) -> list[ServerCommand]:
    return ServerCommandBuilder(config).build_all(ranks)


def build_dist_envs(cur_ip: str, master_ip: str) -> dict[str, str]:
    nic_name = get_net_interface(cur_ip)
    return {
        "HCCL_IF_IP": cur_ip,
        "HCCL_SOCKET_IFNAME": nic_name,
        "GLOO_SOCKET_IFNAME": nic_name,
        "TP_SOCKET_IFNAME": nic_name,
        "LOCAL_IP": cur_ip,
        "NIC_NAME": nic_name,
        "MASTER_IP": master_ip,
    }


def build_proxy_server_cmd(config: ExternalDPConfig, ranks: list[RankInfo]) -> list[str]:
    routing = config.routing
    cmd = [sys.executable, routing.proxy_script, "--host", routing.proxy_host, "--port", str(routing.proxy_port)]

    if routing.type == ROUTING_DISAGGREGATED_PREFILL:
        prefiller_ranks = [rank for rank in ranks if rank.role == "prefiller"]
        decoder_ranks = [rank for rank in ranks if rank.role == "decoder"]
        if not prefiller_ranks or not decoder_ranks:
            raise ValueError("disaggregated_prefill proxy requires prefiller and decoder ranks")
        cmd.extend(["--prefiller-hosts", *[rank.host for rank in prefiller_ranks]])
        cmd.extend(["--prefiller-ports", *[str(rank.port) for rank in prefiller_ranks]])
        cmd.extend(["--decoder-hosts", *[rank.host for rank in decoder_ranks]])
        cmd.extend(["--decoder-ports", *[str(rank.port) for rank in decoder_ranks]])
        return cmd

    raise ValueError(f"Unsupported routing.type: {routing.type}")


def proxy_server_health_url(config: ExternalDPConfig) -> str:
    return f"http://{config.routing.proxy_host}:{config.routing.proxy_port}/healthcheck"


def rank_health_url(rank: RankInfo) -> str:
    return f"http://{rank.host}:{rank.port}/health"


def master_rank_health_url(ranks: list[RankInfo]) -> str:
    for rank in ranks:
        if rank.node_index == 0 and rank.local_rank == 0:
            return rank_health_url(rank)
    raise RuntimeError("External DP master rank was not found")


def rank_label(rank: RankInfo) -> str:
    return f"node={rank.node_index} rank={rank.local_rank} role={rank.role} url={rank_health_url(rank)}"


def format_http_status(label: str, url: str) -> str:
    status = "ready" if is_http_ready(url, timeout=1.0) else "waiting"
    return f"{label}={status} url={url}"


def _format_rank_statuses(
    ranks: list[RankInfo],
    rank_ready: dict[RankInfo, bool],
) -> str:
    parts = []
    for rank in ranks:
        status = "ready" if rank_ready[rank] else "waiting"
        parts.append(f"  {rank_label(rank)} status={status}")
    return "\n".join(parts)


def _raise_if_rank_process_exited(rank_processes: list[RankProcess] | None) -> None:
    if not rank_processes:
        return

    exited = []
    for process, rank, log_file in rank_processes:
        returncode = process.poll()
        if returncode is not None:
            exited.append(f"{rank_label(rank)} pid={process.pid} returncode={returncode} log={log_file}")

    if exited:
        raise RuntimeError("External DP rank process exited before ready: " + "; ".join(exited))


def wait_ranks_ready(
    ranks: Iterable[RankInfo],
    timeout: int,
    rank_processes: list[RankProcess] | None = None,
) -> None:
    ranks = list(ranks)
    rank_ready = {rank: False for rank in ranks}
    deadline = time.monotonic() + timeout
    last_log_time = 0.0

    while True:
        _raise_if_rank_process_exited(rank_processes)

        all_ready = True
        unhealthy_after_ready = []

        for rank in ranks:
            is_ready = is_http_ready(rank_health_url(rank), timeout=1.0)
            if is_ready:
                if not rank_ready[rank]:
                    logger.info("[READY] External DP rank %s", rank_label(rank))
                rank_ready[rank] = True
                continue

            all_ready = False
            if rank_ready[rank]:
                unhealthy_after_ready.append(rank)

        if unhealthy_after_ready:
            failed = "; ".join(rank_label(rank) for rank in unhealthy_after_ready)
            raise RuntimeError(f"External DP rank became unhealthy after ready: {failed}")

        if all_ready:
            return

        now = time.monotonic()
        if now - last_log_time >= 30:
            logger.info(
                "Polling external DP ranks: ready=%d/%d\n%s",
                sum(rank_ready.values()),
                len(ranks),
                _format_rank_statuses(ranks, rank_ready),
            )
            last_log_time = now

        if now >= deadline:
            pending = [rank for rank in ranks if not rank_ready[rank]]
            pending_labels = "; ".join(rank_label(rank) for rank in pending)
            raise TimeoutError(f"Timed out waiting for external DP ranks ready: {pending_labels}")

        time.sleep(5)


def wait_master_rank_stopped(ranks: list[RankInfo], timeout: int) -> None:
    url = master_rank_health_url(ranks)
    wait_http_ready(url, timeout=SERVER_READY_TIMEOUT_SECONDS)
    logger.info("Hanging until master external DP rank stops: %s", url)
    wait_http_unready(url, timeout=timeout)
