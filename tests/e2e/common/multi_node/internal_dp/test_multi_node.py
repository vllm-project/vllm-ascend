import logging
import os
import subprocess
import sys
from typing import Any

import pytest
import vllm

from tests.e2e.common.multi_node.benchmark_results import (
    build_task_entry,
    extract_hardware,
    filter_environment,
    write_results_json,
)
from tests.e2e.common.multi_node.internal_dp.multi_node_config import (
    DEFAULT_SERVER_PORT,
    MultiNodeConfig,
    MultiNodeConfigLoader,
    ProxyLauncher,
)
from tests.e2e.common.spec_decode import (
    get_speculative_config,
    get_speculative_num_tokens,
    make_chat_warmup_fn,
    parse_json_flag,
    select_acceptance_case,
    split_server_cmd,
)
from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly.scripts.result_postprocess import postprocess_benchmark_results
from tools.aisbench import run_aisbench_cases
from tools.profiling.capture import ProfileSpec, ServeInstance, ServeManifest, configure_profiler_command, profile_root

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


def _decode_server_cmd(config: MultiNodeConfig) -> str:
    """Return the decode-side command, which owns the draft position count."""
    if config.disagg_cfg:
        for node in config.nodes:
            if config.disagg_cfg.is_decoder(node.index):
                return node.server_cmd
    return config.nodes[0].server_cmd


def _speculative_num_tokens(config: MultiNodeConfig) -> int:
    """Return the draft position count derived from the decode-side configuration."""
    return get_speculative_num_tokens(_decode_server_cmd(config))


def _spec_metrics_servers(config: MultiNodeConfig) -> list:
    """Return the metrics endpoints that report draft counters for routed requests."""
    from tools.spec_decode_metrics import MetricsEndpoint

    if config.disagg_cfg:
        return [
            MetricsEndpoint(node.ip, int((node.envs or {}).get("SERVER_PORT", DEFAULT_SERVER_PORT)))
            for node in config.nodes
            if config.disagg_cfg.is_decoder(node.index) and not node.headless
        ]
    host, port = config.benchmark_endpoint
    return [MetricsEndpoint(host, port)]


def _extract_features(server_cmd: list[str] | str, envs: dict[str, Any]) -> list[str]:
    """Extract enabled feature names from server_cmd and environment variables."""
    cmd_list = split_server_cmd(server_cmd)
    features: list[str] = []

    # Features from --additional-config JSON
    additional = parse_json_flag(cmd_list, "--additional-config") or parse_json_flag(cmd_list, "--additional_config")
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
    compilation = parse_json_flag(cmd_list, "--compilation-config")
    if compilation.get("cudagraph_mode"):
        features.append("aclgraph")

    # Features from --speculative-config JSON
    speculative = get_speculative_config(cmd_list)
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


def _save_benchmark_results_json(config: MultiNodeConfig, results: list[Any]) -> None:
    """Serialize acc & perf benchmark results to a JSON file under benchmark_results/."""
    runner = os.environ.get("VLLM_CI_RUNNER", "")

    # Filter out None benchmark cases; results align with the non-None ones in order
    valid_items = [(case["benchmark_name"], case) for case in config.benchmark_cases]

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

    postprocess_benchmark_results(
        [(key, case_cfg, result) for (key, case_cfg), result in zip(valid_items, results)],
        job_name=job_name or config.test_name,
    )


def _acceptance_case_index(config: MultiNodeConfig) -> int | None:
    """Return the accuracy benchmark index that owns the acceptance sampling window."""
    cases = [(str(case["benchmark_name"]), case) for case in config.benchmark_cases]
    return select_acceptance_case(cases, config.acceptance_rate.get("benchmark"))


def _run_benchmark_group(config: MultiNodeConfig, host: str, port: int, cases: list[dict[str, Any]]) -> list[Any]:
    """Run a subset of benchmark cases and return their results."""
    if not cases:
        return []
    return run_aisbench_cases(
        model=config.model,
        port=port,
        aisbench_cases=cases,
        host_ip=host,
    )


def _run_benchmarks_with_acceptance_window(config: MultiNodeConfig, host: str, port: int) -> None:
    """Sample draft counters around one accuracy benchmark and validate every position."""
    from tools.spec_decode_metrics import capture_baseline, measure_acceptance_rate, validate_acceptance_rates

    cases = config.benchmark_cases
    bound_index = _acceptance_case_index(config)
    baseline_val = config.acceptance_rate.get("baseline")

    tolerance = float(config.acceptance_rate.get("tolerance", 0.05))
    num_speculative_tokens = _speculative_num_tokens(config)
    metrics_servers = _spec_metrics_servers(config)

    warmup_fn = make_chat_warmup_fn(f"http://{host}:{port}/v1/chat/completions", config.model)

    # Unbound configs keep the legacy whole-session sampling window.
    if bound_index is None:
        errors: list[Exception] = []
        spec_baseline = None
        try:
            spec_baseline = capture_baseline(metrics_servers, num_speculative_tokens, warmup_fn)
        except Exception as e:
            errors.append(e)
            logger.error("Failed to capture spec decode baseline: %s", e)

        try:
            results = _run_benchmark_group(config, host, port, cases)
        except Exception as e:
            errors.append(e)
            logger.error("AISBench run failed: %s", e)
            results = ["" for _ in cases]
        _save_benchmark_results_json(config, results)

        if spec_baseline is not None and baseline_val is not None:
            try:
                _, all_rates = measure_acceptance_rate(metrics_servers, num_speculative_tokens, spec_baseline)
                validate_acceptance_rates(all_rates, baseline_val, tolerance)
            except Exception as e:
                errors.append(e)
                logger.error("Spec decode acceptance rate validation failed: %s", e)

        if errors:
            raise errors[0]
        return

    if baseline_val is None:
        logger.warning("acceptance_rate.baseline not set in config, skipping validation")
        _save_benchmark_results_json(config, _run_benchmark_group(config, host, port, cases))
        return

    results: list[Any] = [""] * len(cases)
    errors: list[Exception] = []

    def run_group(start_index: int, group: list[dict[str, Any]]) -> None:
        if not group:
            return
        try:
            group_results = _run_benchmark_group(config, host, port, group)
        except Exception as e:
            errors.append(e)
            logger.error("Benchmark group starting at %d failed: %s", start_index, e)
            group_results = ["" for _ in group]
        for offset, result in enumerate(group_results):
            results[start_index + offset] = result

    run_group(0, cases[:bound_index])

    spec_baseline = None
    try:
        spec_baseline = capture_baseline(
            metrics_servers,
            num_speculative_tokens,
            warmup_fn,
        )
    except Exception as e:
        errors.append(e)
        logger.error("Failed to capture spec decode baseline: %s", e)

    # The accuracy workload always runs so accuracy evidence is never dropped by a counter failure.
    run_group(bound_index, cases[bound_index : bound_index + 1])

    if spec_baseline is not None:
        try:
            _, all_rates = measure_acceptance_rate(metrics_servers, num_speculative_tokens, spec_baseline)
            validate_acceptance_rates(all_rates, baseline_val, tolerance)
        except Exception as e:
            errors.append(e)
            logger.error("Spec decode acceptance rate validation failed: %s", e)

    # Remaining benchmarks still run after an acceptance failure so nightly evidence is preserved.
    run_group(bound_index + 1, cases[bound_index + 1 :])

    _save_benchmark_results_json(config, results)

    if errors:
        raise errors[0]


def _configure_profiling(config: MultiNodeConfig, spec: ProfileSpec) -> None:
    """Configure every node and associate headless traces with their controlling API."""
    if not spec.enabled:
        return
    api_nodes = {node.ip: node for node in config.nodes if not node.headless}
    instances = {}
    for node in config.nodes:
        if config.disagg_cfg and config.disagg_cfg.is_prefiller(node.index):
            name = f"prefill-{config.disagg_cfg.prefiller_indices.index(node.index)}"
        elif config.disagg_cfg and config.disagg_cfg.is_decoder(node.index):
            name = f"decode-{config.disagg_cfg.decoder_indices.index(node.index)}"
        else:
            name = f"dp-{node.index}"
        api_node = node
        if node.headless:
            leader_ip = (
                config.disagg_cfg.master_ip_for_node(node.index, config.nodes)
                if config.disagg_cfg
                else config.master_ip
            )
            api_node = api_nodes[leader_ip]
        port = (api_node.envs or {}).get("SERVER_PORT", DEFAULT_SERVER_PORT)
        instances[node.index] = ServeInstance.from_endpoint(name, f"http://{api_node.ip}:{port}", node_index=node.index)
    if config.is_master:
        ServeManifest(tuple(instances.values())).write(profile_root() / "serve_manifest.json")
    config.server_cmd = configure_profiler_command(config.server_cmd, instances[config.cur_node.index], spec)


@pytest.mark.asyncio
async def test_multi_node() -> None:
    config = MultiNodeConfigLoader.from_yaml()
    _configure_profiling(config, ProfileSpec.from_env())
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
            if "spec_decode_acceptance" in config.test_content:
                _run_benchmarks_with_acceptance_window(config, host, port)
            else:
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
