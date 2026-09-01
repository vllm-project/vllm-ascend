import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import requests
import vllm

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly.multi_node.internal_dp.scripts.multi_node_config import (
    MultiNodeConfig,
    MultiNodeConfigLoader,
    ProxyLauncher,
)
from tests.e2e.nightly.multi_node.scripts.benchmark_results import (
    build_task_entry,
    compare_version_results,
    extract_hardware,
    filter_environment,
    write_results_json,
)
from tests.e2e.nightly.scripts.result_postprocess import postprocess_benchmark_results
from tools.aisbench import run_aisbench_cases

logger = logging.getLogger(__name__)

_FEATURE_ENVS: dict[str, str] = {
    "VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE": "topk_optimize",
}

_FEATURE_CONFIGS: dict[str, str] = {
    "enable_fused_mc2": "fused_mc2",
    "enable_mlapo": "mlapo",
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
    additional = _parse_json_flag(cmd_list, "--additional-config") or _parse_json_flag(cmd_list, "--additional_config")
    for config_key, feature_name in _FEATURE_CONFIGS.items():
        if additional.get(config_key):
            features.append(feature_name)
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


def _save_benchmark_results_json(
    config: MultiNodeConfig,
    cases: list[dict],
    results: list[Any],
    version: str | None = None,
) -> None:
    """Serialize acc & perf benchmark results to a JSON file under benchmark_results/."""
    runner = os.environ.get("VLLM_CI_RUNNER", "")

    # Filter out None benchmark cases; results align with the non-None ones in order
    valid_items = [(case["case_name"], case) for case in cases]

    tasks = [build_task_entry(key, case_cfg, result) for (key, case_cfg), result in zip(valid_items, results)]

    output: dict[str, Any] = {
        "model_name": config.model,
        "version": version or "default",
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
    if version:
        write_results_json(
            output,
            job_name=f"{job_name}_{version}",
            output_dir=Path("/root/.cache/benchmark_results") / job_name,
        )
    else:
        write_results_json(output, job_name=job_name)


def _abort_marker_path() -> str:
    """Return the shared-PVC path used by the leader to abort worker waiting."""
    log_prefix = os.environ.get("LOG_PREFIX", "/tmp")
    return os.path.join(log_prefix, "abort")


def _version_done_marker_path(version_index: int) -> str:
    """Return the shared-PVC path marking the end of a version run on the leader."""
    log_prefix = os.environ.get("LOG_PREFIX", "/tmp")
    return os.path.join(log_prefix, f"version_done_{version_index}")


def _raise_if_aborted(marker_path: str) -> None:
    if os.path.exists(marker_path):
        raise RuntimeError(f"Leader aborted the multi-version run (abort marker: {marker_path})")


def _hang_until_version_done(
    health_url: str,
    *,
    done_marker: str,
    abort_marker: str,
    timeout_seconds: int = 2800,
    max_consecutive_failures: int = 6,
) -> None:
    """Wait until the leader finishes the current version, failing fast on crashes.

    The leader writes the done marker before shutting down its server, so a
    healthy version transition always exits through the marker. If the leader
    disappears without the marker, raise after several consecutive failed
    health checks instead of returning normally (which would leave the worker
    waiting for a leader that will never start the next version).
    """
    start = time.time()
    consecutive_failures = 0
    while time.time() - start < timeout_seconds:
        _raise_if_aborted(abort_marker)
        if os.path.exists(done_marker):
            return
        try:
            resp = requests.get(health_url, timeout=5)
            healthy = resp.status_code == 200
        except requests.RequestException:
            healthy = False
        if healthy:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    f"Leader at {health_url} is unreachable for {consecutive_failures} consecutive "
                    f"health checks without a done marker ({done_marker})"
                )
        time.sleep(5)
    raise TimeoutError(f"Timed out after {timeout_seconds}s waiting for leader at {health_url}")


def _run_single_version(
    config: MultiNodeConfig,
    *,
    envs: dict[str, str] | None = None,
    version: str | None = None,
    version_index: int | None = None,
    results_by_version: dict[str, dict[str, Any]] | None = None,
    abort_marker: str | None = None,
    benchmark_cases: list[dict] | None = None,
) -> None:
    node_envs = envs if envs is not None else config.envs
    cases = benchmark_cases if benchmark_cases is not None else config.benchmark_cases
    with (
        ProxyLauncher(
            nodes=config.nodes,
            disagg_cfg=config.disagg_cfg,
            envs=node_envs,
            proxy_port=config.proxy_port,
            cur_index=config.cur_index,
        ) as proxy,
        RemoteOpenAIServer(
            model=config.model,
            vllm_serve_args=config.server_cmd,
            server_port=config.server_port,
            server_host=config.master_ip,
            env_dict=node_envs,
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
                aisbench_cases=cases,
                host_ip=host,
            )
            _save_benchmark_results_json(config, cases, results, version=version)
            if results_by_version is not None and version is not None:
                results_by_version[version] = {case["case_name"]: result for case, result in zip(cases, results)}
            if version_index is not None:
                Path(_version_done_marker_path(version_index)).touch()
        else:
            # We should keep listening on the master node's server url determining when to exit.
            if abort_marker is None:
                server.hang_until_terminated(f"http://{host}:{config.server_port}/health")
            else:
                assert version_index is not None, "version_index is required in multi-version mode"
                _hang_until_version_done(
                    f"http://{host}:{config.server_port}/health",
                    done_marker=_version_done_marker_path(version_index),
                    abort_marker=abort_marker,
                )

    postprocess_benchmark_results(
        [(key, case_cfg, result) for (key, case_cfg), result in zip(valid_items, results)],
        job_name=job_name or config.test_name,
    )


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

    if not config.versions:
        _run_single_version(config)
        return

    abort_marker = _abort_marker_path()
    results_by_version: dict[str, dict[str, Any]] = {}
    try:
        for version_index, version in enumerate(config.versions):
            version_name = version["name"]
            version_envs = {**config.envs, **version["env"]}
            selected = version.get("benchmarks")
            version_cases = (
                config.benchmark_cases
                if selected is None
                else [case for case in config.benchmark_cases if case["case_name"] in set(selected)]
            )
            logger.info(
                "Starting version %s with env overrides: %s, cases: %s",
                version_name,
                version["env"],
                [case["case_name"] for case in version_cases],
            )
            _run_single_version(
                config,
                envs=version_envs,
                version=version_name,
                version_index=version_index,
                results_by_version=results_by_version,
                abort_marker=abort_marker,
                benchmark_cases=version_cases,
            )
    except Exception:
        if config.is_master:
            try:
                Path(abort_marker).touch()
            except OSError:
                logger.exception("Failed to write abort marker %s", abort_marker)
        raise

    if config.is_master:
        baseline_versions = [version["name"] for version in config.versions if version["is_baseline"]]
        report, passed = compare_version_results(
            benchmark_cases=config.benchmark_cases,
            results_by_version=results_by_version,
            baseline_version_name=baseline_versions[0],
            default_threshold=config.version_threshold,
        )
        for entry in report:
            logger.info("Version comparison: %s", entry)
            print(f"VERSION COMPARISON: {entry}")
        assert passed, "Version performance comparison failed (see report above)"
