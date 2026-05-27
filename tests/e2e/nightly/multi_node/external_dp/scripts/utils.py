import logging
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    ROUTING_DISAGGREGATED_PREFILL,
    ROUTING_GENERIC_DP,
    ExternalDPConfig,
    NodeTemplate,
    RankInfo,
    replace_cluster_placeholders,
)
from tests.e2e.nightly.multi_node.scripts.benchmark_results import (
    build_task_entry,
    extract_hardware,
    filter_environment,
    get_vllm_version,
    write_results_json,
)
from tests.e2e.nightly.multi_node.scripts.utils import get_net_interface

logger = logging.getLogger(__name__)

TEMPLATE_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
ENV_VAR_RE = re.compile(r"(?<!\$)\$([A-Za-z_][A-Za-z0-9_]*)")
PROXY_HEALTH_PATH = "/healthcheck"
SENSITIVE_ENV_TOKENS = ("TOKEN", "SECRET", "PASSWORD", "ACCESS_KEY")


@dataclass(frozen=True)
class ServerCommand:
    """Rendered command, env, and printable command line."""

    cmd: list[str]
    env: dict[str, str]
    display_cmd: str


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


def format_server_cmd(cmd: list[str], env: dict[str, str] | None = None) -> str:
    env_parts: list[str] = []
    for key, value in sorted((env or {}).items()):
        display_value = "***" if any(token in key.upper() for token in SENSITIVE_ENV_TOKENS) else str(value)
        env_parts.append(f"{key}={shlex.quote(display_value)}")
    return " ".join([*env_parts, shlex.join(cmd)])


def start_logged_process(cmd: list[str], env: dict[str, str], log_file: Path) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    merged_env = {**os.environ, **env}
    with log_file.open("ab") as f:
        f.write(f"Starting command: {format_server_cmd(cmd, env)}\n".encode())
        f.flush()
        return subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=merged_env,
            start_new_session=True,
        )


def terminate_process_tree(pid: int, timeout: int = 30) -> None:
    try:
        import psutil
    except ModuleNotFoundError:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        return

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for process in children:
        process.terminate()
    parent.terminate()

    gone, alive = psutil.wait_procs([parent, *children], timeout=timeout)
    del gone
    for process in alive:
        process.kill()


def _is_http_ready(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_http_ready(url: str, timeout: int, interval: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for HTTP ready: {url}; last_error={last_error}")


def wait_http_unready(url: str, timeout: int, interval: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_http_ready(url):
            return
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for HTTP unready: {url}")


def collect_logs(src_dir: Path, output_tar: Path) -> None:
    if not src_dir.exists():
        return
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_tar, "w:gz") as tar:
        tar.add(src_dir, arcname=src_dir.name)


def build_proxy_server_cmd(config: ExternalDPConfig, ranks: list[RankInfo]) -> list[str]:
    routing = config.routing
    cmd = [sys.executable, routing.proxy_script, "--host", routing.proxy_host, "--port", str(routing.proxy_port)]

    if routing.type == ROUTING_GENERIC_DP:
        worker_ranks = [rank for rank in ranks if rank.role == "worker"]
        if not worker_ranks:
            raise ValueError("generic_dp proxy requires worker ranks")
        cmd.extend(["--dp-hosts", *[rank.host for rank in worker_ranks]])
        cmd.extend(["--dp-ports", *[str(rank.port) for rank in worker_ranks]])
        return cmd

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
    return f"http://{config.routing.proxy_host}:{config.routing.proxy_port}{PROXY_HEALTH_PATH}"


def _common_command_envs(commands: list[ServerCommand]) -> dict[str, str]:
    if not commands:
        return {}

    common_keys = set(commands[0].env)
    for command in commands[1:]:
        common_keys.intersection_update(command.env)

    common_envs: dict[str, str] = {}
    for key in sorted(common_keys):
        values = {command.env[key] for command in commands}
        if len(values) == 1:
            common_envs[key] = next(iter(values))
    return common_envs


def _extract_dtype(config: ExternalDPConfig, commands: list[ServerCommand]) -> str:
    has_w8a8 = "w8a8" in config.model.lower()
    has_quant_ascend = any("--quantization ascend" in command.display_cmd for command in commands)
    return "w8a8" if has_w8a8 and has_quant_ascend else "bf16"


def _extract_features(commands: list[ServerCommand]) -> list[str]:
    if not commands:
        return []
    features: list[str] = []
    command_args = [command.cmd for command in commands]
    command_displays = [" ".join(shlex.quote(arg) for arg in command.cmd) for command in commands]

    if any("--async-scheduling" in cmd for cmd in command_args):
        features.append("async_scheduling")
    if any("--enable-expert-parallel" in cmd for cmd in command_args):
        features.append("expert_parallel")
    if any("--speculative-config" in cmd for cmd in command_args):
        features.append("speculative")
    if any("cudagraph_mode" in display for display in command_displays):
        features.append("aclgraph")

    feature_envs = {
        "VLLM_ASCEND_ENABLE_FLASHCOMM": "flashcomm",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "flashcomm1",
        "VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE": "topk_optimize",
        "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "matmul_allreduce",
        "VLLM_ASCEND_ENABLE_MLAPO": "mlapo",
        "VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL": "context_parallel",
        "VLLM_ASCEND_ENABLE_FUSED_MC2": "fused_mc2",
    }
    for env_key, feature_name in feature_envs.items():
        values = [str(command.env.get(env_key, "0")) for command in commands]
        if any(value not in ("0", "", "false", "False") for value in values):
            features.append(feature_name)
    return features


def _build_serve_cmd(
    config: ExternalDPConfig,
    ranks: list[RankInfo],
    commands: list[ServerCommand],
) -> dict[str, Any]:
    entries: dict[str, str] = {}
    for rank, command in zip(ranks, commands):
        prefix = rank.role
        if config.routing.type == ROUTING_DISAGGREGATED_PREFILL:
            prefix = "prefill" if rank.role == "prefiller" else "decode"
        entries[f"{prefix}-node{rank.node_index}-rank{rank.local_rank}"] = command.display_cmd
    key = "external_dp_pd" if config.routing.type == ROUTING_DISAGGREGATED_PREFILL else "external_dp"
    return {key: entries}


def build_benchmark_results(
    *,
    config: ExternalDPConfig,
    ranks: list[RankInfo],
    commands: list[ServerCommand],
    results: list[Any],
) -> dict[str, Any]:
    valid_items = [(case["case_name"], case) for case in config.benchmark_cases]
    tasks = [build_task_entry(key, case, result) for (key, case), result in zip(valid_items, results)]
    runner = os.environ.get("VLLM_CI_RUNNER", "")
    common_envs = _common_command_envs(commands)

    return {
        "model_name": config.model,
        "hardware": extract_hardware(runner),
        "dtype": _extract_dtype(config, commands),
        "feature": _extract_features(commands),
        "vllm_version": get_vllm_version(),
        "vllm_ascend_version": os.environ.get("VLLM_ASCEND_REF", ""),
        "tasks": tasks,
        "serve_cmd": _build_serve_cmd(config, ranks, commands),
        "environment": filter_environment(common_envs),
    }


def write_benchmark_results_json(
    *,
    config: ExternalDPConfig,
    ranks: list[RankInfo],
    commands: list[ServerCommand],
    results: list[Any],
    output_dir: Path | None = None,
) -> Path:
    output = build_benchmark_results(config=config, ranks=ranks, commands=commands, results=results)
    job_name = os.environ.get("BENCHMARK_JOB_NAME", "") or config.test_name.replace(" ", "-")
    return write_results_json(output, job_name=job_name, output_dir=output_dir)
