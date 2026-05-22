import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import (
    ROUTING_GENERIC_DP,
    ROUTING_PD,
    ExternalDPConfig,
    ExternalDPEndpoint,
    NodeTemplate,
    replace_cluster_placeholders,
)

try:
    import vllm

    VLLM_VERSION = vllm.__version__
except Exception:
    VLLM_VERSION = ""

logger = logging.getLogger(__name__)

BRACED_VARIABLE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
UNBRACED_VARIABLE_RE = re.compile(r"(?<!\$)\$([A-Za-z_][A-Za-z0-9_]*)")
PROXY_HEALTHCHECK_PATH = "/healthcheck"
SENSITIVE_ENV_TOKENS = ("TOKEN", "SECRET", "PASSWORD", "ACCESS_KEY")
_PORT_ENV_KEYS = {"SERVER_PORT", "ENCODE_PORT", "PD_PORT", "PROXY_PORT"}
_INFRA_ENV_KEYS = {
    "HCCL_IF_IP",
    "HCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "LOCAL_IP",
    "NIC_NAME",
    "MASTER_IP",
    "DISAGGREGATED_PREFILL_PROXY_SCRIPT",
}
_PERF_METRIC_RENAME: dict[str, str] = {
    "Benchmark Duration": "Benchmark_Duration(BD)",
    "Prefill Token Throughput": "Prefill_Token_Throughput(PTT)",
    "Input Token Throughput": "Input_Token_Throughput(ITT)",
    "Output Token Throughput": "Output_Token_Throughput(OTT)",
    "Total Token Throughput": "Total_Token_Throughput(TTT)",
}


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
            "HOST": endpoint.bind_host,
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


def _extract_hardware(runner: str) -> str:
    runner_lower = runner.lower()
    for label in ("a3", "a2"):
        if label in runner_lower:
            return label.upper()
    return runner


def _task_passed(case_config: dict[str, Any], result: Any) -> bool:
    if result == "":
        return False
    case_type = case_config.get("case_type")
    baseline = case_config.get("baseline")
    threshold = case_config.get("threshold")
    if baseline is None or threshold is None:
        return True
    if case_type == "accuracy" and isinstance(result, (int, float)):
        return abs(float(result) - float(baseline)) <= float(threshold)
    if case_type == "performance" and isinstance(result, list) and len(result) == 2:
        _, result_json = result
        throughput_str = result_json.get("Output Token Throughput", {}).get("total", "")
        try:
            throughput_val = float(throughput_str.replace("token/s", "").strip())
            return throughput_val >= float(threshold) * float(baseline)
        except (ValueError, AttributeError):
            return False
    return True


def _build_task_entry(case_key: str, case_config: dict[str, Any], result: Any) -> dict[str, Any]:
    dataset_path = case_config.get("dataset_path", "")
    dataset_conf = case_config.get("dataset_conf", "")
    if dataset_path:
        task_name = dataset_path.split("/", 1)[-1]
    elif dataset_conf:
        task_name = dataset_conf.split("/")[0]
    else:
        task_name = case_key

    case_type = case_config.get("case_type", "unknown")
    metrics: dict[str, float] = {}
    if case_type == "accuracy" and isinstance(result, (int, float)):
        metrics["accuracy"] = round(float(result), 4)
    elif case_type == "performance" and isinstance(result, list) and len(result) == 2:
        _, result_json = result
        for metric_name, metric_data in result_json.items():
            if not isinstance(metric_data, dict):
                continue
            total_str = metric_data.get("total", "")
            try:
                value = float(total_str.replace("token/s", "").replace("ms", "").replace("s", "").strip())
                metrics[_PERF_METRIC_RENAME.get(metric_name, metric_name)] = round(value, 4)
            except (ValueError, AttributeError):
                pass

    test_input_keys = ("num_prompts", "max_out_len", "batch_size", "request_rate")
    test_input = {key: case_config[key] for key in test_input_keys if key in case_config}
    target: dict[str, Any] = {}
    if case_config.get("baseline") is not None:
        target["baseline"] = case_config["baseline"]
    if case_config.get("threshold") is not None:
        target["threshold"] = case_config["threshold"]

    entry: dict[str, Any] = {"name": task_name, "metrics": metrics, "test_input": test_input}
    if target:
        entry["target"] = target
    entry["pass_fail"] = "pass" if _task_passed(case_config, result) else "fail"
    return entry


def _filter_environment(envs: dict[str, Any]) -> dict[str, Any]:
    exclude = _PORT_ENV_KEYS | _INFRA_ENV_KEYS
    return {key: value for key, value in envs.items() if key not in exclude}


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


def _build_topology(config: ExternalDPConfig, endpoints: list[ExternalDPEndpoint]) -> dict[str, Any]:
    endpoint_items = []
    for endpoint in endpoints:
        item = asdict(endpoint)
        item.pop("bind_host", None)
        item.pop("port_start", None)
        item.pop("dp_rpc_port", None)
        item.pop("pp_size", None)
        item.pop("cp_size", None)
        item.pop("sp_size", None)
        item.pop("dp_address", None)
        item.pop("dp_size_local", None)
        item.pop("dp_size", None)
        endpoint_items.append(item)

    return {
        "summary": {
            "routing_type": config.routing.type,
            "num_nodes": config.num_nodes,
            "npu_per_node": config.npu_per_node,
            "proxy": {
                "node_index": config.routing.proxy_node_index,
                "host": config.routing.proxy_host,
                "port": config.routing.proxy_port,
            },
            "groups": config.routing.groups,
        },
        "endpoints": endpoint_items,
    }


def build_benchmark_results(
    *,
    config: ExternalDPConfig,
    endpoints: list[ExternalDPEndpoint],
    commands: list[BuiltCommand],
    results: list[Any],
) -> dict[str, Any]:
    valid_items = [(case["case_name"], case) for case in config.benchmark_cases]
    tasks = [_build_task_entry(key, case, result) for (key, case), result in zip(valid_items, results)]
    runner = os.environ.get("VLLM_CI_RUNNER", "")
    common_envs = _common_command_envs(commands)

    return {
        "model_name": config.model,
        "hardware": _extract_hardware(runner),
        "dtype": _extract_dtype(config, commands),
        "feature": _extract_features(commands),
        "vllm_version": VLLM_VERSION,
        "vllm_ascend_version": os.environ.get("VLLM_ASCEND_REF", ""),
        "tasks": tasks,
        "serve_cmd": _build_serve_cmd(config, endpoints, commands),
        "environment": _filter_environment(common_envs),
        "external_dp_topology": _build_topology(config, endpoints),
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
    if output_dir is None:
        output_dir = Path("/root/.cache/benchmark_results") / job_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{job_name}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Benchmark results saved to PVC at {output_path}")
    return output_path
