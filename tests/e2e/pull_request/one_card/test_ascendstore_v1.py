# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare production/v1 serve latency for cold and fully external prefixes."""

import json
import os
import shlex
import shutil
import time
from dataclasses import replace
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from statistics import median
from typing import Any

import pytest
import requests
from vllm.benchmarks.datasets import SampleRequest
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_ports_list

from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.ascendstore_v1_benchmark import (
    BENCHMARK_TIMEOUT_SECONDS,
    BLOCK_SIZE,
    CONTROL_TIMEOUT_SECONDS,
    MODEL,
    PROMPT_TOKEN_COUNT,
    REQUEST_COUNT,
    REUSABLE_TOKEN_COUNT,
    make_requests,
    read_prompt_token_sources,
    reset_local_prefix_cache,
    run_benchmark,
    wait_for_prompt_token_sources,
    write_report,
)

OUTPUT_TOKEN_COUNTS = (1, 8, 32)
ROUNDS_PER_RUN = 2
CONCURRENCIES = (1, 8)
# Eight independent 8K prefixes occupy about 9 GiB; clear the dedicated pool between rounds.
POOL_MEMORY_BYTES = 32 << 30
LOCAL_BUFFER_BYTES = 1 << 30
SUMMARY_METRICS = ("median_ttft_ms", "median_tpot_ms", "median_e2el_ms", "request_throughput")
WORKER_EXTENSION = "tests.e2e.pull_request.one_card.ascendstore_v1_performance.AscendStoreBenchmarkWorkerExtension"
TIMING_SCHEDULER = "tests.e2e.pull_request.one_card.ascendstore_v1_scheduler.AscendStoreTimingScheduler"
DIAGNOSTIC_OUTPUT_TOKENS = 8
DIAGNOSTIC_CONCURRENCY = 1
CONNECTORS = (
    (
        "production",
        "AscendStoreConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
    ),
    (
        "v1",
        "AscendStoreV1Connector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector",
    ),
)
# Separate server lifetimes balance early/late execution without colocating two models on one NPU.
CROSSOVER_RUNS = CONNECTORS + CONNECTORS[::-1]

pytestmark = pytest.mark.e2e_model(MODEL)


def test_production_and_v1_serve_cold_and_external_hit_performance(tmp_path: Path) -> None:
    """Report upstream benchmark timings; gate transfer provenance and outputs, not an uncalibrated speedup."""
    assert shutil.which("mooncake_master") is not None, "The serve benchmark requires mooncake_master on PATH"
    model_path = maybe_model_redirect(MODEL)
    tokenizer = get_tokenizer(model_path)
    inputs = make_requests(tokenizer, "benchmark")
    warmup = make_requests(tokenizer, "warmup", count=1)[0]
    # Distinct first blocks prevent local or external sharing between measured requests and warmup.
    first_blocks = [tuple(sample.prompt[:BLOCK_SIZE]) for sample in inputs]
    assert len(set(first_blocks + [tuple(warmup.prompt[:BLOCK_SIZE])])) == len(first_blocks) + 1

    exercises, execution_runs = _run_crossover(tmp_path, model_path, tokenizer, inputs, warmup)
    observations = {implementation: result["measurements"] for implementation, result in exercises.items()}
    diagnostics = {implementation: result["diagnostics"] for implementation, result in exercises.items()}
    repeat_count = len(diagnostics["production"])
    diagnostic_comparisons = []
    for round_index in range(repeat_count):
        production = diagnostics["production"][round_index]["benchmark"]
        candidate = diagnostics["v1"][round_index]["benchmark"]
        instrumentation_effect = {}
        for implementation in diagnostics:
            profiled = diagnostics[implementation][round_index]["benchmark"]
            baseline = observations[implementation][str(DIAGNOSTIC_OUTPUT_TOKENS)][str(DIAGNOSTIC_CONCURRENCY)]
            reference = baseline[round_index]["cold"]
            instrumentation_effect[implementation] = {
                "profiled_over_uninstrumented_median_ttft": profiled["median_ttft_ms"] / reference["median_ttft_ms"],
                "profiled_over_uninstrumented_median_e2el": profiled["median_e2el_ms"] / reference["median_e2el_ms"],
                "profiled_over_uninstrumented_request_throughput": (
                    profiled["request_throughput"] / reference["request_throughput"]
                ),
                "generated_texts_equal": profiled["generated_texts"] == reference["generated_texts"],
            }
        diagnostic_comparisons.append(
            {
                "production_run": diagnostics["production"][round_index]["run"],
                "v1_run": diagnostics["v1"][round_index]["run"],
                "generated_texts_equal": candidate["generated_texts"] == production["generated_texts"],
                "instrumentation_effect": instrumentation_effect,
            }
        )
    comparisons: dict[str, Any] = {}
    summaries: dict[str, Any] = {implementation: {} for implementation, _, _ in CONNECTORS}
    for output_tokens in OUTPUT_TOKEN_COUNTS:
        output_key = str(output_tokens)
        comparisons[output_key] = {}
        for implementation in summaries:
            summaries[implementation][output_key] = {}
        for concurrency in CONCURRENCIES:
            concurrency_key = str(concurrency)
            production_rounds = observations["production"][output_key][concurrency_key]
            candidate_rounds = observations["v1"][output_key][concurrency_key]
            paired_rounds = []
            for production, candidate in zip(production_rounds, candidate_rounds, strict=True):
                comparison = {}
                for phase in ("cold", "external_hit"):
                    reference, actual = production[phase], candidate[phase]
                    comparison[phase] = {
                        "production_run": production["run"],
                        "v1_run": candidate["run"],
                        "generated_texts_equal": actual["generated_texts"] == reference["generated_texts"],
                        "different_request_indices": [
                            index
                            for index, (reference_text, actual_text) in enumerate(
                                zip(reference["generated_texts"], actual["generated_texts"], strict=True)
                            )
                            if actual_text != reference_text
                        ],
                        "v1_over_production_median_ttft": actual["median_ttft_ms"] / reference["median_ttft_ms"],
                        "v1_over_production_request_throughput": (
                            actual["request_throughput"] / reference["request_throughput"]
                        ),
                    }
                paired_rounds.append(comparison)
            comparisons[output_key][concurrency_key] = paired_rounds
            for implementation in summaries:
                rounds = observations[implementation][output_key][concurrency_key]
                summaries[implementation][output_key][concurrency_key] = _summarize_rounds(rounds)
    report = {
        "model": MODEL,
        "prompt_tokens": PROMPT_TOKEN_COUNT,
        "external_hit_tokens_per_request": REUSABLE_TOKEN_COUNT,
        "output_tokens": OUTPUT_TOKEN_COUNTS,
        "rounds_per_scene": repeat_count,
        "rounds_per_run": ROUNDS_PER_RUN,
        "concurrencies": CONCURRENCIES,
        "requests_per_phase": REQUEST_COUNT,
        "reset_in_benchmark_timing": False,
        "same_prompts_across_scenes": True,
        "prompt_token_id_sha256": [sha256(json.dumps(sample.prompt).encode()).hexdigest() for sample in inputs],
        "implementation_order": [implementation for implementation, _, _ in CROSSOVER_RUNS],
        "execution_runs": execution_runs,
        "post_response_store_drain_in_benchmark_timing": False,
        "tpot_for_one_output_token": "not applicable",
        "measurements": observations,
        "round_summaries": summaries,
        "comparisons": comparisons,
        "stage_timing_diagnostics": {
            "scene": {
                "concurrency": DIAGNOSTIC_CONCURRENCY,
                "output_tokens": DIAGNOSTIC_OUTPUT_TOKENS,
                "phase": "cold",
            },
            "separate_instrumented_runs": True,
            "control_stage_recording_enabled": False,
            "intervals_overlap_and_must_not_be_added": True,
            "step_ids_are_process_local": True,
            "cleanup_request_ids_are_context_not_execution": True,
            "npu_events_read_after_benchmark": True,
            "measurements": diagnostics,
            "comparisons": diagnostic_comparisons,
        },
    }
    # The existing selected-test log artifact preserves the complete report without CI changes.
    write_report(tmp_path / "ascendstore-serve-performance.json", "ASCENDSTORE_SERVE_PERFORMANCE", report)
    # A wording difference must not discard other scenes or their reset/provenance evidence.
    for output_key, concurrency_comparisons in comparisons.items():
        for concurrency_key, rounds in concurrency_comparisons.items():
            for round_index, comparison in enumerate(rounds, start=1):
                for phase, result in comparison.items():
                    assert result["generated_texts_equal"], (
                        output_key,
                        concurrency_key,
                        round_index,
                        phase,
                        result["different_request_indices"],
                    )
    for round_index, comparison in enumerate(diagnostic_comparisons, start=1):
        assert comparison["generated_texts_equal"], ("diagnostic", round_index, "production/v1 outputs differ")
        for implementation, effect in comparison["instrumentation_effect"].items():
            assert effect["generated_texts_equal"], ("diagnostic", round_index, implementation, "hooks changed output")


def _run_crossover(
    tmp_path: Path, model_path: str, tokenizer: TokenizerLike, inputs: list[SampleRequest], warmup: SampleRequest
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exercises: dict[str, Any] = {
        implementation: {
            "measurements": {
                str(tokens): {str(concurrency): [] for concurrency in CONCURRENCIES} for tokens in OUTPUT_TOKEN_COUNTS
            },
            "diagnostics": [],
        }
        for implementation, _, _ in CONNECTORS
    }
    execution_runs = []
    for run_index, (implementation, connector_name, module) in enumerate(CROSSOVER_RUNS, start=1):
        run_path = tmp_path / f"run-{run_index}-{implementation}"
        run_path.mkdir()
        # Each call closes its server and private pool before the next implementation starts.
        start = time.monotonic()
        result = _exercise_connector(run_path, model_path, tokenizer, inputs, warmup, connector_name, module)
        execution_runs.append(
            {"run": run_index, "implementation": implementation, "duration_seconds": time.monotonic() - start}
        )
        accumulated = exercises[implementation]
        for output_key, concurrency_measurements in result["measurements"].items():
            for concurrency_key, rounds in concurrency_measurements.items():
                accumulated["measurements"][output_key][concurrency_key].extend(
                    {"run": run_index, "round_in_run": round_index, **observation}
                    for round_index, observation in enumerate(rounds, start=1)
                )
        accumulated["diagnostics"].extend({"run": run_index, **observation} for observation in result["diagnostics"])
    return exercises, execution_runs


def _summarize_rounds(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep round dispersion per scene, rather than pooling separately reset runs into one throughput."""
    summary: dict[str, dict[str, Any]] = {}
    for phase in ("cold", "external_hit"):
        summary[phase] = {}
        for metric in SUMMARY_METRICS:
            values = [observation[phase][metric] for observation in rounds]
            summary[phase][metric] = (
                {"median": median(values), "min": min(values), "max": max(values)}
                if all(value is not None for value in values)
                else None
            )
    return summary


def _exercise_connector(
    tmp_path: Path,
    model_path: str,
    tokenizer: TokenizerLike,
    inputs: list[SampleRequest],
    warmup: SampleRequest,
    connector_name: str,
    connector_module: str,
) -> dict[str, Any]:
    scheduler_timing_path = tmp_path / f"{connector_name}-scheduler-stage-timing.json"
    master_port, metrics_port, lookup_port, server_port = get_open_ports_list(4)
    pool_config = MooncakeKVPoolConfig(
        config={
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "ascend",
            "device_name": "",
            "global_segment_size": POOL_MEMORY_BYTES,
            "local_buffer_size": LOCAL_BUFFER_BYTES,
        },
        master_port=master_port,
        metrics_port=metrics_port,
    )
    transfer_config = {
        "kv_connector": connector_name,
        "kv_connector_module_path": connector_module,
        "kv_role": "kv_both",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "backend": "mooncake",
            "lookup_rpc_port": lookup_port,
            "use_layerwise": False,
            "load_async": False,
            "save_decode_cache": False,
            "discard_partial_chunks": True,
            "benchmark_stage_timing_path": str(scheduler_timing_path),
        },
    }
    server_args = shlex.split(f"""
        --host 127.0.0.1 --port {server_port} --served-model-name {MODEL}
        --dtype bfloat16 --max-model-len 16384 --max-num-batched-tokens 16384
        --max-num-seqs {max(CONCURRENCIES)} --block-size {BLOCK_SIZE}
        --tensor-parallel-size 1 --pipeline-parallel-size 1 --gpu-memory-utilization 0.8
        --enable-prefix-caching --prefix-caching-hash-algo sha256 --no-enable-chunked-prefill
        --enforce-eager --no-async-scheduling --seed 42
    """) + [
        "--kv-transfer-config",
        json.dumps(transfer_config),
        "--worker-extension-cls",
        WORKER_EXTENSION,
        "--scheduler-cls",
        TIMING_SCHEDULER,
    ]
    # This optional NPU service dependency is needed only when executing our E2E.
    pool_runtime = import_module("tests.e2e.common.single_node.kv_pool_runtime")
    observations: dict[str, Any] = {}
    with pool_runtime.SingleNodeMooncakeManager(pool_config, f"{tmp_path.name}-{connector_name}") as pool:
        # The CLI/Worker subprocess does not inherit pytest's sys.path, and installed packages exclude tests.
        repository_root = str(Path(__file__).resolve().parents[4])
        worker_pythonpath = os.pathsep.join(filter(None, (repository_root, os.environ.get("PYTHONPATH", ""))))
        server_envs = {
            **pool.server_envs,
            "PYTHONPATH": worker_pythonpath,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_KV_CACHE_LAYOUT": "NHD",
            "VLLM_SERVER_DEV_MODE": "1",
        }
        with RemoteOpenAIServer(
            model_path,
            server_args,
            server_host="127.0.0.1",
            server_port=server_port,
            auto_port=False,
            env_dict=server_envs,
        ) as server:
            # Warm kernels with an unrelated prefix, then wait for its metric accounting before measurement.
            baseline = read_prompt_token_sources(server)
            response = requests.post(
                server.url_for("v1", "completions"),
                timeout=BENCHMARK_TIMEOUT_SECONDS,
                json={
                    "model": MODEL,
                    "prompt": warmup.prompt,
                    "max_tokens": max(OUTPUT_TOKEN_COUNTS),
                    "temperature": 0,
                    "seed": 42,
                    "ignore_eos": True,
                },
            )
            response.raise_for_status()
            assert response.json()["usage"]["completion_tokens"] == max(OUTPUT_TOKEN_COUNTS)
            wait_for_prompt_token_sources(server, baseline, PROMPT_TOKEN_COUNT)

            for output_tokens in OUTPUT_TOKEN_COUNTS:
                observations[str(output_tokens)] = {}
                samples = [replace(sample, expected_output_len=output_tokens) for sample in inputs]
                for concurrency in CONCURRENCIES:
                    rounds = []
                    for _ in range(ROUNDS_PER_RUN):
                        # Store must drain before force-removing objects from this test's private Mooncake master.
                        pool_reset = _call_worker_rpc(server, "reset_ascendstore_benchmark_pool", [pool.master_address])
                        assert pool_reset["removed_objects"] > 0, "The preceding requests did not publish external KV"
                        cold_reset = reset_local_prefix_cache(server)
                        cold = run_benchmark(server, tokenizer, samples, concurrency)
                        assert cold["prompt_tokens_by_source"] == {
                            "local_compute": REQUEST_COUNT * PROMPT_TOKEN_COUNT,
                            "local_cache_hit": 0,
                            "external_kv_transfer": 0,
                        }, "Cold requests must not reuse local or external KV"

                        # Local reset may wait for retained Store blocks; the benchmark timer excludes that wait.
                        external_reset = reset_local_prefix_cache(server)
                        external = run_benchmark(server, tokenizer, samples, concurrency)
                        assert external["prompt_tokens_by_source"] == {
                            "local_compute": REQUEST_COUNT,
                            "local_cache_hit": 0,
                            "external_kv_transfer": REQUEST_COUNT * REUSABLE_TOKEN_COUNT,
                        }, "Every reusable prefix must come from External, leaving one token per request to compute"
                        rounds.append(
                            {
                                "cold": {"pool_reset_before": pool_reset, "reset_before": cold_reset, **cold},
                                "external_hit": {"reset_before": external_reset, **external},
                            }
                        )
                    observations[str(output_tokens)][str(concurrency)] = rounds
            # Keep completed controls even if the new diagnostic cannot install or collect its hooks.
            write_report(
                tmp_path / f"{connector_name}-uninstrumented.json",
                "ASCENDSTORE_UNINSTRUMENTED_CHECKPOINT",
                {"connector": connector_name, "stage_recording_enabled": False, "measurements": observations},
            )
            diagnostics = _run_stage_timing_diagnostics(
                server, tokenizer, inputs, pool.master_address, scheduler_timing_path
            )
    return {"measurements": observations, "diagnostics": diagnostics}


def _run_stage_timing_diagnostics(
    server: RemoteOpenAIServer,
    tokenizer: TokenizerLike,
    inputs: list[SampleRequest],
    master_address: str,
    scheduler_timing_path: Path,
) -> list[dict[str, Any]]:
    """Replay only C1/8/cold with temporary hooks; never mix these timings into the uninstrumented matrix."""
    observations: list[dict[str, Any]] = []
    checkpoint_path = scheduler_timing_path.with_name(f"{scheduler_timing_path.stem}-diagnostic-progress.json")
    for round_index in range(ROUNDS_PER_RUN):
        samples = [
            replace(
                sample,
                expected_output_len=DIAGNOSTIC_OUTPUT_TOKENS,
                request_id=f"ascendstore-stage-{round_index + 1}-{index}",
            )
            for index, sample in enumerate(inputs)
        ]
        pool_reset = _call_worker_rpc(server, "reset_ascendstore_benchmark_pool", [master_address])
        assert pool_reset["removed_objects"] > 0
        local_reset = reset_local_prefix_cache(server)
        observation = {"pool_reset_before": pool_reset, "reset_before": local_reset, "round": round_index + 1}
        observations.append(observation)
        try:
            _call_worker_rpc(server, "start_ascendstore_stage_timing")
            try:
                measurements = run_benchmark(server, tokenizer, samples, DIAGNOSTIC_CONCURRENCY)
                observation["benchmark"] = measurements
                assert measurements["prompt_tokens_by_source"] == {
                    "local_compute": REQUEST_COUNT * PROMPT_TOKEN_COUNT,
                    "local_cache_hit": 0,
                    "external_kv_transfer": 0,
                }, "The stage-timing replay must remain a genuinely cold C1 request batch"
            finally:
                # Resolve deferred events and restore hooks outside benchmark timing, even after failed requests.
                worker_timings = _call_worker_rpc(server, "stop_ascendstore_stage_timing")
                observation["worker"] = worker_timings
                reset_local_prefix_cache(server)
            scheduler_timings = json.loads(scheduler_timing_path.read_text(encoding="utf-8"))
            observation["scheduler"] = scheduler_timings
            for timings in (worker_timings, scheduler_timings):
                assert timings["records"], "Diagnostic hooks produced no timing evidence"
                assert any(
                    f"ascendstore-stage-{round_index + 1}-" in request_id
                    for record in timings["records"]
                    for request_id in record["request_ids"]
                ), "Timing evidence belongs to a different diagnostic round"
            observation["stage_summary"] = _summarize_stage_timings(
                worker_timings["records"] + scheduler_timings["records"]
            )
        except BaseException as error:
            # Emit partial traces only on failure; successful traces appear once in the complete final report.
            observation["error"] = f"{type(error).__name__}: {error}"
            write_report(
                checkpoint_path, "ASCENDSTORE_DIAGNOSTIC_CHECKPOINT", {"status": "failed", "rounds": observations}
            )
            raise
    return observations


def _summarize_stage_timings(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Group overlapping intervals by stage and token shape, without fabricating an additive latency breakdown."""
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        tokens = record["scheduled_tokens"]
        phase = "unattributed"
        if tokens and sum(tokens.values()) > 0:
            phase = "prefill" if max(tokens.values()) > 1 else "decode"
        elif record["step_id"] is not None:
            phase = "cleanup"
        groups.setdefault(phase, {}).setdefault(record["stage"], []).append(record)
    summary: dict[str, dict[str, Any]] = {}
    for phase, stages in groups.items():
        summary[phase] = {}
        for stage, intervals in stages.items():
            host_ms = [interval["host_ms"] for interval in intervals]
            device_ms = [interval["device_ms"] for interval in intervals if "device_ms" in interval]
            summary[phase][stage] = {
                "calls": len(intervals),
                "total_host_ms": sum(host_ms),
                "median_host_ms": median(host_ms),
                "max_host_ms": max(host_ms),
                "total_device_stream_ms": sum(device_ms) if device_ms else None,
                "median_device_stream_ms": median(device_ms) if device_ms else None,
            }
    return summary


def _call_worker_rpc(server: RemoteOpenAIServer, method: str, args: list[str] | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    response = requests.post(
        server.url_for("collective_rpc"),
        json={"method": method, "args": args or [], "timeout": CONTROL_TIMEOUT_SECONDS},
        timeout=CONTROL_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    results = response.json()["results"]
    assert len(results) == 1 and isinstance(results[0], dict), "The ordinary benchmark requires one Worker result"
    return {**results[0], "duration_ms": (time.perf_counter() - start) * 1000}
