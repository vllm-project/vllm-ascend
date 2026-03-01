import logging
from typing import Any

import openai
import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import DisaggEpdProxy, RemoteEPDServer, RemoteOpenAIServer
from tests.e2e.nightly.single_node.models.scripts.single_node_config import (
    SingleNodeConfig,
    SingleNodeConfigLoader,
    expand_values,
)
from tools.aisbench import run_aisbench_cases

logger = logging.getLogger(__name__)

# Load configurations at import time for parametrizations
configs = SingleNodeConfigLoader.from_yaml_cases()

# -------------------------------------------------------------
# Modular Test Handlers Registry
# -------------------------------------------------------------


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


def _run_benchmarks(config: SingleNodeConfig, port: int) -> None:
    """Run Aisbench benchmarks and process benchmark-dependent custom assertions."""
    aisbench_cases = [v for v in config.benchmarks.values() if v]
    if not aisbench_cases:
        return

    result = run_aisbench_cases(
        model=config.model,
        port=port,
        aisbench_cases=aisbench_cases,
    )

    # Run generic metric comparisons if explicitly defined in tests or backward compatible TTFT key is present
    if "benchmark_comparisons" in config.test_content:
        run_benchmark_comparisons(config, result)


@pytest.mark.asyncio
@pytest.mark.parametrize("config", configs)
async def test_single_node(config: SingleNodeConfig) -> None:
    if config.service_mode == "epd":
        encode_port = str(get_open_port())
        decode_port = str(get_open_port())
        proxy_port = str(get_open_port())
        expand_envs = {
            **config.envs,
            "ENCODE_PORT": encode_port,
            "DECODE_PORT": decode_port,
            "PD_PORT": decode_port,
            "PROXY_PORT": proxy_port,
        }

        epd_server_cmds = [expand_values(cmd, expand_envs) for cmd in config.epd_server_cmds]
        epd_proxy_args = expand_values(config.epd_proxy_args, expand_envs)

        with (
            RemoteEPDServer(vllm_serve_args=epd_server_cmds, env_dict=config.envs) as _,
            DisaggEpdProxy(proxy_args=epd_proxy_args, env_dict=config.envs) as proxy,
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
