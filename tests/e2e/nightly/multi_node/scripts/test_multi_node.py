import json
import logging
import os
import shlex
from typing import Any

import pytest

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly.multi_node.scripts.multi_node_config import (
    MultiNodeConfig, MultiNodeConfigLoader, ProxyLauncher)
from tools.aisbench import run_aisbench_cases

logger = logging.getLogger(__name__)

_PORT_ENV_KEYS = {"SERVER_PORT", "ENCODE_PORT", "PD_PORT", "PROXY_PORT"}
_INFRA_ENV_KEYS = {
    "HCCL_IF_IP", "HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME", "LOCAL_IP", "NIC_NAME", "MASTER_IP",
    "DISAGGREGATED_PREFILL_PROXY_SCRIPT",
}

_FEATURE_ENVS: dict[str, str] = {
    "VLLM_ASCEND_ENABLE_FLASHCOMM": "flashcomm",
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": "flashcomm1",
    "VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE": "topk_optimize",
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "matmul_allreduce",
    "VLLM_ASCEND_ENABLE_MLAPO": "mlapo",
    "VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL": "context_parallel",
    "VLLM_ASCEND_ENABLE_FUSED_MC2": "fused_mc2",
}

_PERF_METRIC_RENAME: dict[str, str] = {
    "Benchmark Duration": "Benchmark_Duration(BD)",
    "Prefill Token Throughput": "Prefill_Token_Throughput(PTT)",
    "Input Token Throughput": "Input_Token_Throughput(ITT)",
    "Output Token Throughput": "Output_Token_Throughput(OTT)",
    "Total Token Throughput": "Total_Token_Throughput(TTT)",
}


def _extract_hardware(runner: str) -> str:
    runner_lower = runner.lower()
    for label in ("a3", "a2"):
        if label in runner_lower:
            return label.upper()
    return runner


def _extract_dtype(config: MultiNodeConfig) -> str:
    """Determine weight dtype: w8a8 if model name contains 'w8a8' and any node uses --quantization ascend."""
    has_w8a8 = "w8a8" in config.model.lower()
    has_quant_ascend = any(
        "--quantization ascend" in node.server_cmd for node in config.nodes
    )
    return "w8a8" if (has_w8a8 and has_quant_ascend) else "bf16"


def _cmd_to_list(server_cmd: list[str] | str) -> list[str]:
    """Normalize server_cmd to a list of argument strings."""
    if isinstance(server_cmd, str):
        try:
            return shlex.split(server_cmd)
        except ValueError:
            return server_cmd.split()
    return list(server_cmd)


def _extract_server_cmd_value(cmd_list: list[str], flag: str) -> str | None:
    """Return the value following `flag` in a command list, or None."""
    try:
        idx = cmd_list.index(flag)
        return cmd_list[idx + 1]
    except (ValueError, IndexError):
        return None


def _parse_json_flag(cmd_list: list[str], flag: str) -> dict[str, Any]:
    """Extract and JSON-parse the value following `flag` in a command list."""
    val = _extract_server_cmd_value(cmd_list, flag)
    if not val:
        return {}
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        return {}


def _extract_features(server_cmd: list[str] | str, envs: dict[str, Any]) -> list[str]:
    """Extract enabled feature names from server_cmd and environment variables."""
    cmd_list = _cmd_to_list(server_cmd)
    features: list[str] = []

    # Features from --additional-config JSON
    additional = _parse_json_flag(cmd_list, "--additional-config")
    if additional.get("enable_weight_nz_layout"):
        features.append("weight_nz_layout")
    wp = additional.get("weight_prefetch_config") or {}
    if isinstance(wp, dict) and wp.get("enabled"):
        features.append("weight_prefetch")
    tc = additional.get("torchair_graph_config") or {}
    if isinstance(tc, dict) and tc.get("enabled"):
        features.append("torchair_graph")
    asc = additional.get("ascend_scheduler_config") or {}
    if isinstance(asc, dict) and asc.get("enabled"):
        features.append("ascend_scheduler")

    # Features from --compilation-config JSON
    compilation = _parse_json_flag(cmd_list, "--compilation-config")
    if compilation.get("cudagraph_mode"):
        features.append("aclgraph")

    # Features from --speculative-config JSON
    speculative = _parse_json_flag(cmd_list, "--speculative-config")
    if speculative:
        features.append(speculative.get("method", "speculative"))

    # Features from direct flags
    if "--async-scheduling" in cmd_list:
        features.append("async_scheduling")
    if "--enable-expert-parallel" in cmd_list:
        features.append("expert_parallel")

    # Features from environment variables
    for env_key, feature_name in _FEATURE_ENVS.items():
        val = str(envs.get(env_key, "0"))
        if val not in ("0", "", "false", "False"):
            features.append(feature_name)
    if int(envs.get("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", 0)) > 0:
        features.append("flashcomm2")

    return features


def _build_serve_cmd(config: MultiNodeConfig) -> dict[str, Any]:
    """Build serve_cmd dict: pd format for disaggregated, dp format for multi-node."""
    if config.disagg_cfg:
        pd: dict[str, str] = {}
        for node in config.nodes:
            idx = node.index
            if config.disagg_cfg.is_prefiller(idx):
                n = config.disagg_cfg.prefiller_indices.index(idx)
                pd[f"prefill-{n}"] = node.server_cmd
            elif config.disagg_cfg.is_decoder(idx):
                n = config.disagg_cfg.decoder_indices.index(idx)
                pd[f"decode-{n}"] = node.server_cmd
        return {"pd": pd}
    return {"dp": {f"node{node.index}": node.server_cmd for node in config.nodes}}


def _filter_environment(envs: dict[str, Any]) -> dict[str, Any]:
    """Return env vars with internal port and infrastructure keys removed."""
    exclude = _PORT_ENV_KEYS | _INFRA_ENV_KEYS
    return {k: v for k, v in envs.items() if k not in exclude}


def _task_passed(case_config: dict[str, Any], result: Any) -> bool:
    """Return True if a single benchmark result meets its baseline/threshold."""
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
    """Build a single task dict in the required format."""
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

    if result == "":
        pass
    elif case_type == "accuracy" and isinstance(result, (int, float)):
        metrics["accuracy"] = round(float(result), 4)
    elif case_type == "performance" and isinstance(result, list) and len(result) == 2:
        _, result_json = result
        for metric_name, metric_data in result_json.items():
            if not isinstance(metric_data, dict):
                continue
            total_str = metric_data.get("total", "")
            try:
                value = float(
                    total_str.replace("token/s", "").replace("ms", "").replace("s", "").strip()
                )
                metrics[_PERF_METRIC_RENAME.get(metric_name, metric_name)] = round(value, 4)
            except (ValueError, AttributeError):
                pass

    test_input_keys = ("num_prompts", "max_out_len", "batch_size", "request_rate")
    test_input = {k: case_config[k] for k in test_input_keys if k in case_config}

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


def _save_benchmark_results_json(config: MultiNodeConfig, results: list[Any]) -> None:
    """Serialize acc & perf benchmark results to a JSON file under benchmark_results/."""
    runner = os.environ.get("VLLM_CI_RUNNER", "")

    # Filter out None benchmark cases; results align with the non-None ones in order
    valid_items = [(case["case_name"], case) for case in config.benchmark_cases]

    tasks = [
        _build_task_entry(key, case_cfg, result)
        for (key, case_cfg), result in zip(valid_items, results)
    ]

    output: dict[str, Any] = {
        "model_name": config.model,
        "hardware": _extract_hardware(runner),
        "dtype": _extract_dtype(config),
        "feature": _extract_features(config.nodes[0].server_cmd, config.envs),
        "vllm_version": os.environ.get("VLLM_VERSION", ""),
        "vllm_ascend_version": os.environ.get("VLLM_ASCEND_VERSION", ""),
        "tasks": tasks,
        "serve_cmd": _build_serve_cmd(config),
        "environment": _filter_environment(config.envs),
    }

    job_name = os.environ.get("BENCHMARK_JOB_NAME", "")
    pvc_benchmark_dir = os.path.join("/root/.cache/benchmark_results", job_name)
    os.makedirs(pvc_benchmark_dir, exist_ok=True)
    pvc_output_path = os.path.join(pvc_benchmark_dir, f"{job_name}.json")
    with open(pvc_output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Benchmark results saved to PVC at %s", pvc_output_path)
    print(f"Benchmark results saved to PVC at {pvc_output_path}")


@pytest.mark.asyncio
async def test_multi_node() -> None:
    config = MultiNodeConfigLoader.from_yaml()

    with ProxyLauncher(
            nodes=config.nodes,
            disagg_cfg=config.disagg_cfg,
            envs=config.envs,
            proxy_port=config.proxy_port,
            cur_index=config.cur_index,
    ) as proxy:

        with RemoteOpenAIServer(
                model=config.model,
                vllm_serve_args=config.server_cmd,
                server_port=config.server_port,
                server_host=config.master_ip,
                env_dict=config.envs,
                auto_port=False,
                proxy_port=proxy.proxy_port,
                disaggregated_prefill=config.disagg_cfg,
                nodes_info=config.nodes,
                max_wait_seconds=2800,
        ) as server:

            host, port = config.benchmark_endpoint

            if config.is_master:
                results = run_aisbench_cases(
                    model=config.model,
                    port=port,
                    aisbench_cases=config.benchmark_cases,
                    host_ip=host,
                )
                _save_benchmark_results_json(config, results)
            else:
                # We should keep listening on the master node's server url determining when to exit.
                server.hang_until_terminated(
                    f"http://{host}:{config.server_port}/health")
