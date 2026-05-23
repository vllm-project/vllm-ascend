import asyncio
import json
import logging
import os
import shlex
import subprocess
import sys
from typing import Any

import openai
import psutil
import pytest
import vllm

from tests.e2e.conftest import DisaggEpdProxy, RemoteEPDServer, RemoteOpenAIServer
from tests.e2e.nightly.single_node.models.scripts.single_node_config import (
    SingleNodeConfig,
    SingleNodeConfigLoader,
)
from tools.aisbench import run_aisbench_cases

logger = logging.getLogger(__name__)

configs = SingleNodeConfigLoader.from_yaml_cases()


async def run_completion_test(config: SingleNodeConfig, server: "RemoteOpenAIServer | DisaggEpdProxy") -> None:
    client = server.get_async_client()
    batch = await client.completions.create(
        model=config.model,
        prompt=config.prompts,
        **config.api_keyword_args,
    )
    choices: list[openai.types.CompletionChoice] = batch.choices
    assert choices[0].text, "empty response"
    print(choices)


async def run_image_test(config: SingleNodeConfig, server: "RemoteOpenAIServer | DisaggEpdProxy") -> None:
    from tools.send_mm_request import send_image_request

    send_image_request(config.model, server)


async def run_chat_completion_test(config: SingleNodeConfig, server: "RemoteOpenAIServer | DisaggEpdProxy") -> None:
    from tools.send_request import send_v1_chat_completions

    send_v1_chat_completions(
        config.prompts[0],
        model=config.model,
        server=server,
        request_args=config.api_keyword_args,
    )


def run_benchmark_comparisons(config: SingleNodeConfig, results: Any) -> None:
    """General assertion engine for aisbench outcomes mapped directly from YAML."""

    comparisons = config.extra_config.get("benchmark_comparisons_args", [])

    if not comparisons:
        return

    # Valid task keys defined in benchmarks mapping
    valid_keys = [k for k, v in config.benchmarks.items() if v]

    metrics_cache = {}

    for comp in comparisons:
        metric = comp.get("metric", "TTFT")
        baseline_key = comp.get("baseline")
        target_key = comp.get("target")
        ratio = comp.get("ratio", 1.0)
        op = comp.get("operator", "<")

        if not baseline_key or not target_key:
            logger.warning("Invalid comparison config: missing baseline or target. %s", comp)
            continue

        if metric not in metrics_cache:
            if metric == "TTFT":
                from tools.aisbench import get_TTFT

                # map TTFT outputs directly to their corresponding benchmark test case names
                metrics_cache[metric] = dict(zip(valid_keys, get_TTFT(results)))
            else:
                logger.warning("Unsupported metric for comparison: %s", metric)
                continue

        metric_dict = metrics_cache[metric]
        baseline_val = metric_dict.get(baseline_key)
        target_val = metric_dict.get(target_key)

        if baseline_val is None or target_val is None:
            logger.warning("Missing data to compare %s and %s in metrics: %s", baseline_key, target_key, metric_dict)
            continue

        expected_threshold = baseline_val * ratio

        eval_str = f"metric {metric}: {target_key}({target_val}) {op} {baseline_key}({baseline_val}) * {ratio}"

        if op == "<":
            assert target_val < expected_threshold, f"Assertion Failed: {eval_str} [threshold: {expected_threshold}]"
        elif op == ">":
            assert target_val > expected_threshold, f"Assertion Failed: {eval_str} [threshold: {expected_threshold}]"
        elif op == "<=":
            assert target_val <= expected_threshold, f"Assertion Failed: {eval_str} [threshold: {expected_threshold}]"
        elif op == ">=":
            assert target_val >= expected_threshold, f"Assertion Failed: {eval_str} [threshold: {expected_threshold}]"
        else:
            logger.warning("Unsupported comparison operator: %s", op)
            continue

        print(f"✅ Comparison passed: {eval_str} [threshold: {expected_threshold}]")


async def run_check_rank0_process_count(
    config: SingleNodeConfig, server: "RemoteOpenAIServer | DisaggEpdProxy"
) -> None:
    proc = await asyncio.create_subprocess_exec(
        "npu-smi",
        "info",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await proc.communicate()
    if proc.returncode == 0:
        logger.info("npu-smi info:\n%s", stdout_bytes.decode(errors="ignore"))
    else:
        logger.warning("npu-smi info failed: %s", stderr_bytes.decode(errors="ignore"))

    vllm_serve_procs = [
        p
        for p in psutil.process_iter(attrs=["pid", "cmdline"], ad_value=None)
        if p.info["cmdline"]
        and any("vllm" in arg for arg in p.info["cmdline"])
        and any("serve" in arg for arg in p.info["cmdline"])
    ]
    count = len(vllm_serve_procs)
    assert count == 1, (
        f"rank0 process count check failed: expected exactly 1 vllm serve process on rank0, found {count}"
    )


# Extend this dictionary to add new test capabilities
TEST_HANDLERS = {
    "completion": run_completion_test,
    "image": run_image_test,
    "chat_completion": run_chat_completion_test,
    "check_rank0_process_count": run_check_rank0_process_count,
}


async def _dispatch_tests(config: SingleNodeConfig, server: "RemoteOpenAIServer | DisaggEpdProxy") -> None:
    """Dispatches requested tests defined in yaml."""
    for test_name in config.test_content:
        if test_name == "benchmark_comparisons":
            continue

        handler = TEST_HANDLERS.get(test_name)
        if handler:
            await handler(config, server)
        else:
            logger.warning("No handler registered for test content type: %s", test_name)


def _extract_server_cmd_value(server_cmd: list[str], flag: str) -> str | None:
    """Return the value following `flag` in a server_cmd list, or None."""
    try:
        idx = server_cmd.index(flag)
        return server_cmd[idx + 1]
    except (ValueError, IndexError):
        return None


def _extract_hardware(runner: str) -> str:
    """Derive hardware label (e.g. 'A2', 'A3') from runner name."""
    runner_lower = runner.lower()
    for label in ("a3", "a2"):
        if label in runner_lower:
            return label.upper()
    return runner


_PORT_ENV_KEYS = {"SERVER_PORT", "ENCODE_PORT", "PD_PORT", "PROXY_PORT"}

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


def _extract_dtype(config: SingleNodeConfig) -> str:
    """Determine weight dtype: w8a8 if model name contains 'w8a8' and --quantization ascend is set, else bf16."""
    has_w8a8 = "w8a8" in config.model.lower()
    has_quant_ascend = _extract_server_cmd_value(config.server_cmd, "--quantization") == "ascend"
    return "w8a8" if (has_w8a8 and has_quant_ascend) else "bf16"


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
    if isinstance(server_cmd, str):
        try:
            cmd_list = shlex.split(server_cmd)
        except ValueError:
            cmd_list = server_cmd.split()
    else:
        cmd_list = list(server_cmd)

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


def _build_serve_cmd(config: SingleNodeConfig) -> dict[str, str]:
    """Build serve_cmd dict with mix key for single-node deployments."""
    args = " ".join(config.server_cmd)
    return {"mix": f"vllm serve {config.model} {args}".strip()}


def _filter_environment(envs: dict[str, Any]) -> dict[str, Any]:
    """Return env vars with internal port keys removed."""
    return {k: v for k, v in envs.items() if k not in _PORT_ENV_KEYS}


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
        # benchmark run failed — no metrics available
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
                value = float(total_str.replace("token/s", "").replace("ms", "").replace("s", "").strip())
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


def _all_passed(case_configs: list[dict[str, Any]], results: list[Any]) -> bool:
    """Return True only when every benchmark result meets its baseline/threshold."""
    return all(_task_passed(cfg, res) for cfg, res in zip(case_configs, results))


def _save_benchmark_results_json(config: SingleNodeConfig, benchmark_keys: list[str], results: list[Any]) -> None:
    """Serialize acc & perf benchmark results to a JSON file under benchmark_results/."""
    runner = os.environ.get("VLLM_CI_RUNNER", "")
    case_configs = [config.benchmarks[k] for k in benchmark_keys]

    tasks = [
        _build_task_entry(key, case_cfg, result) for key, case_cfg, result in zip(benchmark_keys, case_configs, results)
    ]

    passed = _all_passed(case_configs, results)

    output: dict[str, Any] = {
        "model_name": config.model,
        "hardware": _extract_hardware(runner),
        "dtype": _extract_dtype(config),
        "feature": _extract_features(config.server_cmd, config.envs),
        "vllm_version": vllm.__version__,
        "vllm_ascend_version": os.environ.get("VLLM_ASCEND_VERSION", ""),
        "tasks": tasks,
        "serve_cmd": _build_serve_cmd(config),
        "environment": _filter_environment(config.envs),
        "pass_fail": "pass" if passed else "fail",
    }

    os.makedirs("benchmark_results", exist_ok=True)
    job_name = os.environ.get("BENCHMARK_JOB_NAME") or config.name
    safe_name = job_name.replace("/", "_").replace(" ", "_")
    output_path = os.path.join("benchmark_results", f"{safe_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Benchmark results saved to %s", output_path)
    print(f"Benchmark results saved to {output_path}")


def _run_benchmarks(config: SingleNodeConfig, port: int) -> None:
    """Run Aisbench benchmarks and process benchmark-dependent custom assertions."""
    benchmark_keys = [k for k, v in config.benchmarks.items() if v]
    aisbench_cases = [config.benchmarks[k] for k in benchmark_keys]
    if not aisbench_cases:
        return

    result = run_aisbench_cases(
        model=config.model,
        port=port,
        aisbench_cases=aisbench_cases,
    )

    _save_benchmark_results_json(config, benchmark_keys, result)

    if "benchmark_comparisons" in config.test_content:
        run_benchmark_comparisons(config, result)


@pytest.mark.asyncio
@pytest.mark.parametrize("config", configs, ids=[config.name for config in configs])
async def test_single_node(config: SingleNodeConfig) -> None:
    # TODO: remove this part after the transformers version upgraded
    if config.special_dependencies:
        for k, v in config.special_dependencies.items():
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"{k}=={v}",
            ]
            subprocess.call(command)
    if config.service_mode == "epd":
        with (
            RemoteEPDServer(vllm_serve_args=config.epd_server_cmds, env_dict=config.envs) as _,
            DisaggEpdProxy(proxy_args=config.epd_proxy_args, env_dict=config.envs) as proxy,
        ):
            await _dispatch_tests(config, proxy)
            _run_benchmarks(config, proxy.port)
        return

    # Standard OpenAI service mode
    with RemoteOpenAIServer(
        model=config.model,
        vllm_serve_args=config.server_cmd,
        server_port=config.server_port,
        env_dict=config.envs,
        auto_port=False,
    ) as server:
        await _dispatch_tests(config, server)
        _run_benchmarks(config, config.server_port)
