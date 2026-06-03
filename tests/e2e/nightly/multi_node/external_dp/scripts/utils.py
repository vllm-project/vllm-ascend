import logging
import os
import shlex
import signal
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    ROUTING_DISAGGREGATED_PREFILL,
    ExternalDPConfig,
    RankInfo,
)
from tests.e2e.nightly.multi_node.scripts.benchmark_results import (
    build_task_entry,
    extract_hardware,
    filter_environment,
    get_vllm_version,
    write_results_json,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tests.e2e.nightly.multi_node.external_dp.scripts.runtime import ServerCommand

SENSITIVE_ENV_TOKENS = ("TOKEN", "SECRET", "PASSWORD", "ACCESS_KEY")


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


def is_http_ready(url: str, timeout: float = 5.0) -> bool:
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
        if not is_http_ready(url):
            return
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for HTTP unready: {url}")


def collect_logs(src_dir: Path, output_tar: Path) -> None:
    if not src_dir.exists():
        return
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_tar, "w:gz") as tar:
        tar.add(src_dir, arcname=src_dir.name)


def _common_command_envs(commands: list["ServerCommand"]) -> dict[str, str]:
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


def _extract_dtype(config: ExternalDPConfig, commands: list["ServerCommand"]) -> str:
    has_w8a8 = "w8a8" in config.model.lower()
    has_quant_ascend = any("--quantization ascend" in command.display_cmd for command in commands)
    return "w8a8" if has_w8a8 and has_quant_ascend else "bf16"


def _extract_features(commands: list["ServerCommand"]) -> list[str]:
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
    commands: list["ServerCommand"],
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
    commands: list["ServerCommand"],
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
    commands: list["ServerCommand"],
    results: list[Any],
    output_dir: Path | None = None,
) -> Path:
    output = build_benchmark_results(config=config, ranks=ranks, commands=commands, results=results)
    job_name = os.environ.get("BENCHMARK_JOB_NAME", "") or config.test_name.replace(" ", "-")
    return write_results_json(output, job_name=job_name, output_dir=output_dir)
