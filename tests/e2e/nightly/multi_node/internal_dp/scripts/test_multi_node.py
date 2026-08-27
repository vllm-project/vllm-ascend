import json
import logging
import os
import shlex
import subprocess
import sys
from typing import Any

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
from tools.aisbench import run_aisbench_cases

logger = logging.getLogger(__name__)

_FEATURE_ENVS: dict[str, str] = {
    "VLLM_ASCEND_ENABLE_FLASHCOMM": "flashcomm",
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": "flashcomm1",
    "VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE": "topk_optimize",
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "matmul_allreduce",
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
            results = run_aisbench_cases(
                model=config.model,
                port=port,
                aisbench_cases=config.benchmark_cases,
                host_ip=host,
            )
            _save_benchmark_results_json(config, results)
        else:
            # We should keep listening on the master node's server url determining when to exit.
            server.hang_until_terminated(f"http://{host}:{config.server_port}/health")
