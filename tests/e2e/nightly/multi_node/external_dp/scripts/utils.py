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
    ROUTING_GENERIC_DP,
    ROUTING_PD,
    ExternalDPConfig,
    ExternalDPEndpoint,
    NodeTemplate,
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

BRACED_VARIABLE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
UNBRACED_VARIABLE_RE = re.compile(r"(?<!\$)\$([A-Za-z_][A-Za-z0-9_]*)")
PROXY_HEALTHCHECK_PATH = "/healthcheck"
SENSITIVE_ENV_TOKENS = ("TOKEN", "SECRET", "PASSWORD", "ACCESS_KEY")


@dataclass(frozen=True)
class BuiltCommand:
    """Rendered command, env, and printable command line."""

    cmd: list[str]
    env: dict[str, str]
    display_cmd: str


class CommandBuilder:
    """Render endpoint templates into vLLM serve commands."""

    def __init__(self, config: ExternalDPConfig):
        self.config = config

    def build(self, endpoint: ExternalDPEndpoint, template: NodeTemplate) -> BuiltCommand:
        variables = self._build_variables(endpoint)
        rendered_env = self._render_envs(template.envs, endpoint, variables)
        rendered_args = [
            self._render_string(
                arg,
                endpoint=endpoint,
                braced_variables=variables,
                unbraced_variables=rendered_env,
                allow_missing_unbraced=False,
            )
            for arg in template.server_cmd_template
        ]
        cmd = ["vllm", "serve", self.config.model, *rendered_args]

        env = {key: str(value) for key, value in rendered_env.items()}
        display_cmd = format_command(cmd, env)
        logger.info(
            "External DP endpoint command node=%s rank=%s: %s",
            endpoint.config_index,
            endpoint.local_rank,
            display_cmd,
        )
        return BuiltCommand(cmd=cmd, env=env, display_cmd=display_cmd)

    def _build_variables(self, endpoint: ExternalDPEndpoint) -> dict[str, str]:
        return {
            "MODEL": self.config.model,
            "PORT_START": str(endpoint.port_start),
            "PORT": str(endpoint.port),
            "DP_SIZE": str(endpoint.dp_size),
            "DP_SIZE_LOCAL": str(endpoint.dp_size_local),
            "DP_RANK_START": str(endpoint.dp_rank - endpoint.local_rank),
            "DP_RANK": str(endpoint.dp_rank),
            "LOCAL_RANK": str(endpoint.local_rank),
            "TP_SIZE": str(endpoint.tp_size),
            "CP_SIZE": str(endpoint.cp_size),
            "SP_SIZE": str(endpoint.sp_size),
            "PP_SIZE": str(endpoint.pp_size),
            "DP_ADDRESS": endpoint.dp_address,
            "DP_RPC_PORT": str(endpoint.dp_rpc_port),
            "VISIBLE_DEVICES": endpoint.visible_devices,
            "NODE_INDEX": str(endpoint.config_index),
            "CONFIG_INDEX": str(endpoint.config_index),
        }

    def _render_envs(
        self,
        envs: dict[str, Any],
        endpoint: ExternalDPEndpoint,
        variables: dict[str, str],
    ) -> dict[str, str]:
        rendered_envs: dict[str, str] = {}
        for key, value in envs.items():
            if isinstance(value, str):
                value = self._render_string(
                    value,
                    endpoint=endpoint,
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
        endpoint: ExternalDPEndpoint,
        braced_variables: dict[str, str],
        unbraced_variables: dict[str, str],
        allow_missing_unbraced: bool,
    ) -> str:
        value = replace_cluster_placeholders(
            value,
            cluster_ips=self.config.cluster_ips,
            local_ip=endpoint.host,
            current_node_index=endpoint.config_index,
        )
        value = self._render_variables(
            value,
            braced_variables,
            pattern=BRACED_VARIABLE_RE,
            allow_missing=False,
        )
        return self._render_variables(
            value,
            unbraced_variables,
            pattern=UNBRACED_VARIABLE_RE,
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


def build_distributed_envs(cur_ip: str, master_ip: str) -> dict[str, str]:
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


def format_command(cmd: list[str], env: dict[str, str] | None = None) -> str:
    env_parts: list[str] = []
    for key, value in sorted((env or {}).items()):
        display_value = "***" if any(token in key.upper() for token in SENSITIVE_ENV_TOKENS) else str(value)
        env_parts.append(f"{key}={shlex.quote(display_value)}")
    return " ".join([*env_parts, shlex.join(cmd)])


def start_process(cmd: list[str], env: dict[str, str], log_file: Path) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    merged_env = {**os.environ, **env}
    with log_file.open("ab") as f:
        f.write(f"Starting command: {format_command(cmd, env)}\n".encode())
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


def build_proxy_command(config: ExternalDPConfig, endpoints: list[ExternalDPEndpoint]) -> list[str]:
    routing = config.routing
    cmd = [sys.executable, routing.proxy_script, "--host", routing.proxy_host, "--port", str(routing.proxy_port)]

    if routing.type == ROUTING_GENERIC_DP:
        worker_endpoints = [endpoint for endpoint in endpoints if endpoint.role == "worker"]
        if not worker_endpoints:
            raise ValueError("generic_dp proxy requires worker endpoints")
        cmd.extend(["--dp-hosts", *[endpoint.host for endpoint in worker_endpoints]])
        cmd.extend(["--dp-ports", *[str(endpoint.port) for endpoint in worker_endpoints]])
        return cmd

    if routing.type == ROUTING_PD:
        prefiller_endpoints = [endpoint for endpoint in endpoints if endpoint.role == "prefiller"]
        decoder_endpoints = [endpoint for endpoint in endpoints if endpoint.role == "decoder"]
        if not prefiller_endpoints or not decoder_endpoints:
            raise ValueError("disaggregated_prefill proxy requires prefiller and decoder endpoints")
        cmd.extend(["--prefiller-hosts", *[endpoint.host for endpoint in prefiller_endpoints]])
        cmd.extend(["--prefiller-ports", *[str(endpoint.port) for endpoint in prefiller_endpoints]])
        cmd.extend(["--decoder-hosts", *[endpoint.host for endpoint in decoder_endpoints]])
        cmd.extend(["--decoder-ports", *[str(endpoint.port) for endpoint in decoder_endpoints]])
        return cmd

    raise ValueError(f"Unsupported routing.type: {routing.type}")


def proxy_health_url(config: ExternalDPConfig) -> str:
    return f"http://{config.routing.proxy_host}:{config.routing.proxy_port}{PROXY_HEALTHCHECK_PATH}"


def _common_command_envs(commands: list[BuiltCommand]) -> dict[str, str]:
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


def _extract_dtype(config: ExternalDPConfig, commands: list[BuiltCommand]) -> str:
    has_w8a8 = "w8a8" in config.model.lower()
    has_quant_ascend = any("--quantization ascend" in command.display_cmd for command in commands)
    return "w8a8" if has_w8a8 and has_quant_ascend else "bf16"


def _extract_features(commands: list[BuiltCommand]) -> list[str]:
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
    endpoints: list[ExternalDPEndpoint],
    commands: list[BuiltCommand],
) -> dict[str, Any]:
    entries: dict[str, str] = {}
    for endpoint, command in zip(endpoints, commands):
        prefix = endpoint.role
        if config.routing.type == ROUTING_PD:
            prefix = "prefill" if endpoint.role == "prefiller" else "decode"
        entries[f"{prefix}-node{endpoint.config_index}-rank{endpoint.local_rank}"] = command.display_cmd
    key = "external_dp_pd" if config.routing.type == ROUTING_PD else "external_dp"
    return {key: entries}


def build_benchmark_results(
    *,
    config: ExternalDPConfig,
    endpoints: list[ExternalDPEndpoint],
    commands: list[BuiltCommand],
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
        "serve_cmd": _build_serve_cmd(config, endpoints, commands),
        "environment": filter_environment(common_envs),
    }


def write_benchmark_results_json(
    *,
    config: ExternalDPConfig,
    endpoints: list[ExternalDPEndpoint],
    commands: list[BuiltCommand],
    results: list[Any],
    output_dir: Path | None = None,
) -> Path:
    output = build_benchmark_results(config=config, endpoints=endpoints, commands=commands, results=results)
    job_name = os.environ.get("BENCHMARK_JOB_NAME", "") or config.test_name.replace(" ", "-")
    return write_results_json(output, job_name=job_name, output_dir=output_dir)
