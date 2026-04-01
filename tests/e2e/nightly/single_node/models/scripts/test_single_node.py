import json
import logging
import os
from typing import Any

import openai
import pytest
import subprocess
import sys

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


# Extend this dictionary to add new test capabilities
TEST_HANDLERS = {
    "completion": run_completion_test,
    "image": run_image_test,
    "chat_completion": run_chat_completion_test,
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


def _build_task_entry(case_key: str, case_config: dict[str, Any], result: Any) -> dict[str, Any]:
    """Build a single task dict in the required format."""
    dataset_conf = case_config.get("dataset_conf", "") or case_config.get("dataset_path", "")
    task_name = dataset_conf.split("/")[0] if dataset_conf else case_key
    case_type = case_config.get("case_type", "unknown")
    metrics: list[dict[str, Any]] = []

    if result == "":
        # benchmark run failed — no metrics available
        pass
    elif case_type == "accuracy" and isinstance(result, (int, float)):
        metrics.append({"name": "accuracy", "value": round(float(result), 4)})
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
                metrics.append({"name": metric_name, "value": round(value, 4)})
            except (ValueError, AttributeError):
                pass

    return {"name": task_name, "metrics": metrics}


def _build_test_config(config: SingleNodeConfig) -> dict[str, Any]:
    """Extract test configuration from server_cmd and benchmark case configs."""
    perf_case: dict[str, Any] = next(
        (v for v in config.benchmarks.values() if v and v.get("case_type") == "performance"),
        {},
    )
    tp_str = _extract_server_cmd_value(config.server_cmd, "--tensor-parallel-size")
    gmu_str = _extract_server_cmd_value(config.server_cmd, "--gpu-memory-utilization")

    test_cfg: dict[str, Any] = {
        "output_len": perf_case.get("max_out_len"),
        "batch_size": perf_case.get("batch_size"),
        "num_prompts": perf_case.get("num_prompts"),
        "tensor_parallel_size": int(tp_str) if tp_str else None,
        "gpu_memory_utilization": float(gmu_str) if gmu_str else None,
    }
    return {k: v for k, v in test_cfg.items() if v is not None}


def _all_passed(case_configs: list[dict[str, Any]], results: list[Any]) -> bool:
    """Return True only when every benchmark result meets its baseline/threshold."""
    for case_config, result in zip(case_configs, results):
        if result == "":
            return False
        case_type = case_config.get("case_type")
        baseline = case_config.get("baseline")
        threshold = case_config.get("threshold")
        if baseline is None or threshold is None:
            continue
        if case_type == "accuracy" and isinstance(result, (int, float)):
            if abs(float(result) - float(baseline)) > float(threshold):
                return False
        elif case_type == "performance" and isinstance(result, list) and len(result) == 2:
            _, result_json = result
            throughput_str = result_json.get("Output Token Throughput", {}).get("total", "")
            try:
                throughput_val = float(throughput_str.replace("token/s", "").strip())
                if throughput_val < float(threshold) * float(baseline):
                    return False
            except (ValueError, AttributeError):
                return False
    return True


def _save_benchmark_results_json(config: SingleNodeConfig, benchmark_keys: list[str], results: list[Any]) -> None:
    """Serialize acc & perf benchmark results to a JSON file under benchmark_results/."""
    runner = os.environ.get("VLLM_CI_RUNNER", "")
    case_configs = [config.benchmarks[k] for k in benchmark_keys]

    tasks = [
        _build_task_entry(key, case_cfg, result)
        for key, case_cfg, result in zip(benchmark_keys, case_configs, results)
    ]

    passed = _all_passed(case_configs, results)

    output: dict[str, Any] = {
        "model_name": config.model,
        "hardware": _extract_hardware(runner),
        "vllm_version": os.environ.get("VLLM_VERSION", ""),
        "vllm_ascend_version": os.environ.get("VLLM_ASCEND_VERSION", ""),
        "pass_fail": "pass" if passed else "fail",
        "tasks": tasks,
        "test_config": _build_test_config(config),
        "known_issues": config.extra_config.get("known_issues", ""),
        "notes": config.extra_config.get("notes", ""),
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
                "-m", "pip", "install",
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
