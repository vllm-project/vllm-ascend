import json
import logging
import os
import shlex
import subprocess
import sys
from typing import Any

import requests
import pytest
import vllm

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly.multi_node.internal_dp.scripts.multi_node_config import (
    MultiNodeConfig,
    MultiNodeConfigLoader,
    ProxyLauncher,
)
from tests.e2e.nightly.multi_node.scripts.benchmark_results import (
    build_task_entry,
    extract_hardware,
    filter_environment,
    write_results_json,
)
from tests.e2e.nightly.multi_node.scripts.utils import ProxyServer
from tools.aisbench import run_aisbench_cases

logger = logging.getLogger(__name__)

_FEATURE_ENVS: dict[str, str] = {
    "VLLM_ASCEND_ENABLE_FLASHCOMM": "flashcomm",
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": "flashcomm1",
    "VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE": "topk_optimize",
    "VLLM_ASCEND_ENABLE_MLAPO": "mlapo",
    "VLLM_ASCEND_ENABLE_FUSED_MC2": "fused_mc2",
}


def _extract_dtype(config: MultiNodeConfig) -> str:
    """Determine weight dtype: w8a8 if model name contains 'w8a8' and any node uses --quantization ascend."""
    has_w8a8 = "w8a8" in config.model.lower()
    has_quant_ascend = any("--quantization ascend" in node.server_cmd for node in config.nodes)
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
    if "--enable-expert-parallel" in cmd_list:
        features.append("expert_parallel")

    # Features from environment variables
    for env_key, feature_name in _FEATURE_ENVS.items():
        val = str(envs.get(env_key, "0"))
        if val not in ("0", "", "false", "False"):
            features.append(feature_name)

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


def _build_server_adapters(config: MultiNodeConfig) -> tuple[ProxyServer, ProxyServer, ProxyServer]:
    host, port = config.benchmark_endpoint
    completion_server = ProxyServer(host, port)

    if config.disagg_cfg:
        prefill_idx = config.disagg_cfg.prefiller_indices[0]
        decode_idx = config.disagg_cfg.decoder_indices[0]
        sp = config.server_port
        tokenize_server = ProxyServer(config.nodes[prefill_idx].ip, sp)
        metrics_server = ProxyServer(config.nodes[decode_idx].ip, sp)
    else:
        tokenize_server = completion_server
        metrics_server = completion_server

    return completion_server, tokenize_server, metrics_server


def _run_chat_completion(
    config: MultiNodeConfig,
    completion_server: ProxyServer,
    tokenize_server: ProxyServer,
) -> None:
    from tools.send_request import resolve_prompt, send_v1_chat_completions

    prompts = config.chat_prompts or ["Hello!"]
    expected = config.expected_response or {}

    max_model_len_str = _extract_server_cmd_value(config.server_cmd_list, "--max-model-len")
    max_model_len = int(max_model_len_str) if max_model_len_str else None

    if isinstance(config.api_keyword_args, list):
        api_args_list = config.api_keyword_args
        if len(api_args_list) != len(prompts):
            raise ValueError(f"""
api_keyword_args list length ({len(api_args_list)}) must match prompts length ({len(prompts)})""")
    else:
        api_args_list = [config.api_keyword_args] * len(prompts)

    if isinstance(expected.get("per_prompt"), list):
        expected_list = expected["per_prompt"]
    else:
        expected_list = [expected] * len(prompts)

    for prompt_raw, api_args, exp in zip(prompts, api_args_list, expected_list):
        prompt, actual_prompt_tokens = resolve_prompt(tokenize_server, prompt_raw, use_chat=True)
        if actual_prompt_tokens is not None:
            exp = dict(exp) if exp else {}
            exp.setdefault("prompt_tokens", actual_prompt_tokens)
        send_v1_chat_completions(
            prompt,
            model=config.model,
            server=completion_server,
            request_args=api_args,
            expected=exp,
            max_model_len=max_model_len,
        )


def _run_spec_decode_acceptance(
    config: MultiNodeConfig,
    metrics_server: ProxyServer,
    baseline: tuple[int, list[int]] | None = None,
) -> None:
    from tools.spec_decode_metrics import measure_acceptance_rate, validate_acceptance_rate

    spec_config = _parse_json_flag(config.server_cmd_list, "--speculative-config")
    num_speculative_tokens = int(spec_config.get("num_speculative_tokens", 1))

    acceptance_cfg = config.acceptance_rate or {}
    baseline_val = acceptance_cfg.get("baseline")
    tolerance = acceptance_cfg.get("tolerance", 0.05)

    if baseline_val is None:
        logger.warning("acceptance_rate.baseline not set in config, skipping validation")
        baseline_val = 0.0

    if baseline is None:
        baseline = (0, [0] * num_speculative_tokens)

    _, all_rates = measure_acceptance_rate(metrics_server, num_speculative_tokens, baseline)
    validate_acceptance_rate(all_rates[0], float(baseline_val), float(tolerance))


def _save_benchmark_results_json(config: MultiNodeConfig, results: list[Any]) -> None:
    """Serialize acc & perf benchmark results to a JSON file under benchmark_results/."""
    runner = os.environ.get("VLLM_CI_RUNNER", "")

    # Filter out None benchmark cases; results align with the non-None ones in order
    valid_items = [(case["case_name"], case) for case in config.benchmark_cases]

    tasks = [build_task_entry(key, case_cfg, result) for (key, case_cfg), result in zip(valid_items, results)]

    output: dict[str, Any] = {
        "model_name": config.model,
        "hardware": extract_hardware(runner),
        "dtype": _extract_dtype(config),
        "feature": _extract_features(config.nodes[0].server_cmd, config.envs),
        "vllm_version": vllm.__version__,
        "vllm_ascend_version": os.environ.get("VLLM_ASCEND_REF", ""),
        "tasks": tasks,
        "serve_cmd": _build_serve_cmd(config),
        "environment": filter_environment(config.envs),
    }

    job_name = os.environ.get("BENCHMARK_JOB_NAME", "")
    write_results_json(output, job_name=job_name)


@pytest.mark.asyncio
async def test_multi_node() -> None:
    config = MultiNodeConfigLoader.from_yaml()
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

    with (
        ProxyLauncher(
            nodes=config.nodes,
            disagg_cfg=config.disagg_cfg,
            envs=config.envs,
            proxy_port=config.proxy_port,
            cur_index=config.cur_index,
        ) as proxy,
        RemoteOpenAIServer(
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
        ) as server,
    ):
        host, port = config.benchmark_endpoint

        if config.is_master:
            completion_server, tokenize_server, metrics_server = _build_server_adapters(config)

            if "chat_completion" in config.test_content:
                _run_chat_completion(config, completion_server, tokenize_server)

            spec_baseline = None
            if "spec_decode_acceptance" in config.test_content:
                from tools.spec_decode_metrics import capture_baseline

                spec_config = _parse_json_flag(config.server_cmd_list, "--speculative-config")
                num_spec_tokens = int(spec_config.get("num_speculative_tokens", 1))

                def warmup_fn():
                    requests.post(
                        completion_server.url_for("v1", "chat", "completions"),
                        json={
                            "model": config.model,
                            "messages": [{"role": "user", "content": "Hello!"}],
                            "max_tokens": 16,
                        },
                        timeout=120,
                    )

                spec_baseline = capture_baseline(metrics_server, num_spec_tokens, warmup_fn)

            results = run_aisbench_cases(
                model=config.model,
                port=port,
                aisbench_cases=config.benchmark_cases,
                host_ip=host,
            )
            _save_benchmark_results_json(config, results)

            if "spec_decode_acceptance" in config.test_content:
                _run_spec_decode_acceptance(config, metrics_server, spec_baseline)
        else:
            server.hang_until_terminated(f"http://{host}:{config.server_port}/health")
