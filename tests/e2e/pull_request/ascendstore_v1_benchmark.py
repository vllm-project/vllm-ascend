# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request generation and upstream benchmark measurements for our AscendStore E2Es."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

import requests
from prometheus_client.parser import text_string_to_metric_families
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.serve import TaskType, benchmark
from vllm.tokenizers import TokenizerLike
from vllm.v1.metrics.stats import PromptTokenStats

from tests.e2e.common.kv_pool.ascendstore_byte_accuracy import compare_saved_bytes

if TYPE_CHECKING:
    from tests.e2e.conftest import RemoteOpenAIServer

MODEL = "Qwen/Qwen3-8B"
BLOCK_SIZE = 128
REUSABLE_TOKEN_COUNT = 8192
PROMPT_TOKEN_COUNT = REUSABLE_TOKEN_COUNT + 1
REQUEST_COUNT = 16
OUTPUT_TOKEN_COUNT = 256
CONCURRENCY = 8
ROUNDS_PER_RUN = 2
WORKER_EXTENSION = "tests.e2e.pull_request.one_card.ascendstore_v1_performance.AscendStoreBenchmarkWorkerExtension"
CONNECTORS = (
    ("production", "AscendStoreConnector"),
    ("v1", "AscendStoreV1Connector"),
)
# Separate server lifetimes balance early/late execution without colocating competing implementations.
CROSSOVER_RUNS = CONNECTORS + CONNECTORS[::-1]
SUMMARY_METRICS = (
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
)
ACCURACY_PROMPT_TOKENS = 2 * BLOCK_SIZE + 1
HTTP_TIMEOUT_SECONDS = 30
CONTROL_TIMEOUT_SECONDS = 60
BENCHMARK_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 0.1


def write_report(path: Path, label: str, report: dict[str, Any]) -> None:
    """Persist complete or partial evidence locally and in the existing CI log artifact."""
    serialized = json.dumps(report, indent=2)
    path.write_text(serialized, encoding="utf-8")
    print(f"\n{label}\n{serialized}")


def make_requests(
    tokenizer: TokenizerLike,
    scenario: str,
    count: int = REQUEST_COUNT,
    *,
    prompt_tokens: int = PROMPT_TOKEN_COUNT,
    output_tokens: int = OUTPUT_TOKEN_COUNT,
) -> list[SampleRequest]:
    document = tokenizer.encode(
        "External KV caches retain attention keys and values for previously computed token prefixes. "
        "A request can restore those values instead of recomputing the same document. ",
        add_special_tokens=False,
    )
    question = tokenizer.encode(
        "\nExplain this document in several paragraphs with examples.\n", add_special_tokens=False
    )
    samples = []
    for index in range(count):
        header = tokenizer.encode(f"Document {scenario}-{index}.\n", add_special_tokens=False)
        remaining = prompt_tokens - len(header) - len(question)
        assert document and remaining > 0
        body = (document * ((remaining + len(document) - 1) // len(document)))[:remaining]
        samples.append(
            SampleRequest(
                prompt=header + body + question,
                prompt_len=prompt_tokens,
                expected_output_len=output_tokens,
            )
        )
    return samples


def run_benchmark(server: RemoteOpenAIServer, tokenizer: TokenizerLike, samples: list[SampleRequest]) -> dict[str, Any]:
    baseline, preemptions_before = _read_serving_counters(server)
    # Reuse upstream timing, streaming and concurrency. No readiness or warmup generation may repopulate local KV.
    operation = benchmark(
        task_type=TaskType.GENERATION,
        endpoint_type="openai",
        api_url=server.url_for("v1", "completions"),
        base_url=server.url_root,
        model_id=MODEL,
        model_name=MODEL,
        tokenizer=tokenizer,
        input_requests=samples,
        logprobs=None,
        request_rate=float("inf"),
        burstiness=1.0,
        disable_tqdm=True,
        num_warmups=0,
        profile=False,
        selected_percentile_metrics=["ttft"],
        selected_percentiles=[50.0, 95.0],
        ignore_eos=True,
        goodput_config_dict={},
        max_concurrency=CONCURRENCY,
        lora_modules=None,
        extra_headers=None,
        extra_body={"temperature": 0, "seed": 42},
        ready_check_timeout_sec=0,
        probe_request_rate=0,
    )
    result = asyncio.run(asyncio.wait_for(operation, timeout=BENCHMARK_TIMEOUT_SECONDS))
    # Keep scalars, per-request latency and outputs; omit bulky per-token timing arrays from the CI report.
    measurements = {
        **{key: result[key] for key in (*SUMMARY_METRICS, "completed", "failed", "duration")},
        **{
            key: result[key] for key in ("ttfts", "latencies", "generated_texts", "input_lens", "output_lens", "errors")
        },
    }
    try:
        assert result["completed"] == len(samples) and result["failed"] == 0, result["errors"]
        assert result["input_lens"] == [sample.prompt_len for sample in samples]
        assert result["output_lens"] == [sample.expected_output_len for sample in samples]
        assert result["median_ttft_ms"] > 0 and result["request_throughput"] > 0
        measurements["prompt_tokens_by_source"], preemptions_after = _wait_for_prompt_tokens(
            server, baseline, sum(sample.prompt_len for sample in samples)
        )
        measurements["preemptions"] = preemptions_after - preemptions_before
        assert measurements["preemptions"] == 0, "KV capacity caused preemption; this is not the intended workload"
    except BaseException:
        print("\nASCENDSTORE_FAILED_MEASUREMENT\n" + json.dumps(measurements, indent=2))
        raise
    return measurements


def reset_local_prefix_cache(server: RemoteOpenAIServer) -> dict[str, float | int]:
    """Record the complete local-only reset, including bounded retries, outside serving benchmark timing."""
    start = time.perf_counter()
    attempts = 0
    while time.perf_counter() - start < CONTROL_TIMEOUT_SECONDS:
        response = requests.post(
            server.url_for("reset_prefix_cache"), params={"reset_external": "false"}, timeout=HTTP_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        attempts += 1
        if response.json().get("success") is True:
            return {"duration_ms": (time.perf_counter() - start) * 1000, "attempts": attempts}
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError("Local prefix reset did not succeed after waiting for requests and retained Store blocks")


def _read_serving_counters(server: RemoteOpenAIServer) -> tuple[dict[str, float], float]:
    response = requests.get(server.url_for("metrics"), timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    sources: dict[str, float] = {}
    preemptions: list[float] = []
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name == "vllm:prompt_tokens_by_source_total":
                source = sample.labels["source"]
                sources[source] = sources.get(source, 0.0) + sample.value
            elif sample.name == "vllm:num_preemptions_total":
                preemptions.append(sample.value)
    assert set(sources) == set(PromptTokenStats.ALL_SOURCES), "Upstream prompt-token provenance metrics are required"
    assert preemptions, "Upstream preemption counters are required"
    return sources, sum(preemptions)


def _wait_for_prompt_tokens(
    server: RemoteOpenAIServer, baseline: dict[str, float], expected_tokens: int
) -> tuple[dict[str, float], float]:
    deadline = time.monotonic() + CONTROL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        sources, preemptions = _read_serving_counters(server)
        delta = {source: sources[source] - baseline[source] for source in baseline}
        assert all(value >= 0 for value in delta.values()), "Prompt-token counters unexpectedly reset"
        assert sum(delta.values()) <= expected_tokens, "Unmeasured requests contaminated benchmark provenance"
        if sum(delta.values()) == expected_tokens:
            return delta, preemptions
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"Prompt-token metrics did not account for all {expected_tokens} measured input tokens")


def assert_prompt_sources(
    measurement: dict[str, Any],
    *,
    external: bool,
    request_count: int = REQUEST_COUNT,
    prompt_tokens: int = PROMPT_TOKEN_COUNT,
) -> None:
    assert measurement["prompt_tokens_by_source"] == {
        "local_compute": request_count if external else request_count * prompt_tokens,
        "local_cache_hit": 0,
        "external_kv_transfer": request_count * (prompt_tokens - 1) if external else 0,
    }, "Cold requests must compute locally; warm requests must load every reusable block from External"


def call_worker_rpc(
    server: RemoteOpenAIServer, method: str, worker_count: int = 1, args: list[str] | None = None
) -> dict[str, Any]:
    start = time.perf_counter()
    response = requests.post(
        server.url_for("collective_rpc"),
        json={"method": method, "args": args or [], "timeout": CONTROL_TIMEOUT_SECONDS},
        timeout=CONTROL_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    workers = response.json()["results"]
    assert len(workers) == worker_count and all(isinstance(worker, dict) for worker in workers), workers
    return {"workers": workers, "duration_ms": (time.perf_counter() - start) * 1000}


def make_report(inputs: list[SampleRequest], warmup: list[SampleRequest]) -> dict[str, Any]:
    first_blocks = [tuple(sample.prompt[:BLOCK_SIZE]) for sample in inputs + warmup]
    assert len(set(first_blocks)) == len(first_blocks), "Warmup and measured requests must not share prefixes"
    return {
        "model": MODEL,
        "prompt_tokens": PROMPT_TOKEN_COUNT,
        "external_hit_tokens_per_request": REUSABLE_TOKEN_COUNT,
        "output_tokens": OUTPUT_TOKEN_COUNT,
        "concurrency": CONCURRENCY,
        "requests_per_phase": REQUEST_COUNT,
        "rounds_per_run": ROUNDS_PER_RUN,
        "implementation_order": [item[0] for item in CROSSOVER_RUNS],
        "prompt_token_id_sha256": [sha256(json.dumps(sample.prompt).encode()).hexdigest() for sample in inputs],
        "timing_scope": "upstream serving benchmark; reset, pool clear and post-response Store drain excluded",
        "load_scope": "fully External-cached prompt; one final prompt token and all output tokens still compute",
        "concurrency_scope": "closed-loop client limit; not a guarantee of eight requests in every model step",
        "total_token_throughput_note": "includes cached input tokens; not NPU compute throughput",
        "throughput_scope": "complete 256-output-token requests; includes Decode, not isolated Backend throughput",
        "accuracy_policy": "two full blocks, one request, one output token; exact KV bytes gate all performance runs",
        "text_parity_policy": "report only; request failures, wrong token counts or hit sources fail the test",
        "summary_policy": "median and range of separate rounds; percentiles are per-round, not pooled",
        "runs": [],
        "accuracy": [],
    }


def run_comparison(tmp_path: Path, report: dict[str, Any], exercise: Callable[..., None]) -> None:
    """Fail the entire test before any performance server starts if accuracy fails."""
    reference_directory = None
    reference_objects = None
    for connector in CONNECTORS:
        path = tmp_path / f"accuracy-{connector[0]}"
        path.mkdir()
        directory = path / "saved-kv"
        observations: list[dict[str, Any]] = []
        report["accuracy"].append({"implementation": connector[0], "observations": observations})
        exercise(path, connector=connector, observations=observations, accuracy_directory=directory)
        objects = observations[0]["stored_objects"]
        if reference_directory is None:
            reference_directory, reference_objects = directory, objects
        else:
            compare_saved_bytes(reference_directory, directory, reference_objects, objects)
    report["accuracy_status"] = "passed"
    for run_index, connector in enumerate(CROSSOVER_RUNS, start=1):
        path = tmp_path / f"performance-{run_index}-{connector[0]}"
        path.mkdir()
        run: dict[str, Any] = {"run": run_index, "implementation": connector[0], "measurements": []}
        report["runs"].append(run)
        start = time.monotonic()
        try:
            exercise(path, connector=connector, observations=run["measurements"])
        finally:
            run["duration_seconds"] = time.monotonic() - start
    report["comparisons"] = compare_runs(report["runs"])
    report["status"] = "measured"


def run_byte_accuracy(producer, consumer, tokenizer, directory: Path, observations: list, consumer_tp: int = 1) -> None:
    samples = make_requests(tokenizer, "byte-accuracy", count=1, prompt_tokens=ACCURACY_PROMPT_TOKENS, output_tokens=1)
    row: dict[str, Any] = {}
    observations.append(row)
    call_worker_rpc(producer, "start_ascendstore_byte_accuracy", args=[str(directory)])
    if consumer is not producer:
        call_worker_rpc(consumer, "start_ascendstore_byte_accuracy", consumer_tp, args=[str(directory)])
    # No warmup request: these dedicated servers have no cached accuracy prefix.
    cold = run_benchmark(producer, tokenizer, samples)
    assert_prompt_sources(cold, external=False, request_count=1, prompt_tokens=ACCURACY_PROMPT_TOKENS)
    stored = call_worker_rpc(producer, "finish_ascendstore_byte_accuracy", args=["store"])
    row["stored_objects"] = stored["workers"][0]["objects"]
    reset_local_prefix_cache(consumer)
    call_worker_rpc(consumer, "clear_ascendstore_accuracy_kv", consumer_tp)
    loaded = run_benchmark(consumer, tokenizer, samples)
    assert_prompt_sources(loaded, external=True, request_count=1, prompt_tokens=ACCURACY_PROMPT_TOKENS)
    evidence = call_worker_rpc(consumer, "finish_ascendstore_byte_accuracy", consumer_tp, args=["load"])
    row["loaded_objects"] = {}
    for worker in evidence["workers"]:
        assert not row["loaded_objects"].keys() & worker["objects"].keys(), "Duplicate Load object ownership"
        row["loaded_objects"].update(worker["objects"])
    assert row["stored_objects"] == row["loaded_objects"], "Load did not cover every saved byte"
    row["status"] = "passed"
    call_worker_rpc(producer, "stop_ascendstore_byte_accuracy")
    if consumer is not producer:
        call_worker_rpc(consumer, "stop_ascendstore_byte_accuracy", consumer_tp)


def compare_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = {}
    for phase in ("cold_store", "external_load"):
        observations = {
            implementation: [
                (run["run"], row["round"], row[phase])
                for run in runs
                if run["implementation"] == implementation
                for row in run["measurements"]
            ]
            for implementation, _ in CONNECTORS
        }
        production, candidate = observations["production"], observations["v1"]
        comparison: dict[str, Any] = {"rounds_per_implementation": len(production), "text_comparisons": []}
        for expected, actual in zip(production, candidate, strict=True):
            differences = [
                index
                for index, (left, right) in enumerate(
                    zip(expected[2]["generated_texts"], actual[2]["generated_texts"], strict=True)
                )
                if left != right
            ]
            comparison["text_comparisons"].append(
                {
                    "production_run": expected[0],
                    "v1_run": actual[0],
                    "round_in_run": expected[1],
                    "generated_texts_equal": not differences,
                    "different_request_indices": differences,
                }
            )
        for metric in SUMMARY_METRICS:
            summaries = {}
            for implementation, rounds in observations.items():
                values = [row[2][metric] for row in rounds]
                summaries[implementation] = {"median": median(values), "min": min(values), "max": max(values)}
            reference, actual = summaries["production"]["median"], summaries["v1"]["median"]
            ratio = actual / reference if reference > 0 else None
            improvement = None if ratio is None else 100 * (ratio - 1 if metric.endswith("throughput") else 1 - ratio)
            comparison[metric] = {**summaries, "v1_over_production": ratio, "improvement_percent": improvement}
        comparisons[phase] = comparison
    return comparisons
