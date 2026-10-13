# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request generation and upstream benchmark measurements for our AscendStore E2Es."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from prometheus_client.parser import text_string_to_metric_families
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.serve import TaskType, benchmark
from vllm.tokenizers import TokenizerLike
from vllm.v1.metrics.stats import PromptTokenStats

if TYPE_CHECKING:
    from tests.e2e.conftest import RemoteOpenAIServer

MODEL = "Qwen/Qwen3-8B"
BLOCK_SIZE = 128
REUSABLE_TOKEN_COUNT = 8192
PROMPT_TOKEN_COUNT = REUSABLE_TOKEN_COUNT + 1
REQUEST_COUNT = 8
HTTP_TIMEOUT_SECONDS = 30
CONTROL_TIMEOUT_SECONDS = 60
BENCHMARK_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 0.1


def write_report(path: Path, label: str, report: dict[str, Any]) -> None:
    """Persist evidence locally and in the existing CI log artifact, including partial diagnostic failures."""
    serialized = json.dumps(report, indent=2)
    path.write_text(serialized, encoding="utf-8")
    print(f"\n{label}\n{serialized}")


def make_requests(tokenizer: TokenizerLike, scenario: str, count: int = REQUEST_COUNT) -> list[SampleRequest]:
    document = tokenizer.encode(
        "External KV caches retain attention keys and values for previously computed token prefixes. "
        "A request can restore those values instead of recomputing the same document. ",
        add_special_tokens=False,
    )
    question = tokenizer.encode("\nSummarize the document in one sentence.\n", add_special_tokens=False)
    samples = []
    for index in range(count):
        header = tokenizer.encode(f"Document {scenario}-{index}.\n", add_special_tokens=False)
        remaining = PROMPT_TOKEN_COUNT - len(header) - len(question)
        assert document and remaining > 0
        body = (document * ((remaining + len(document) - 1) // len(document)))[:remaining]
        samples.append(
            SampleRequest(
                prompt=header + body + question,
                prompt_len=PROMPT_TOKEN_COUNT,
                expected_output_len=1,
            )
        )
    return samples


def run_benchmark(
    server: RemoteOpenAIServer, tokenizer: TokenizerLike, samples: list[SampleRequest], concurrency: int
) -> dict[str, Any]:
    baseline = read_prompt_token_sources(server)
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
        selected_percentile_metrics=["ttft", "tpot", "e2el"],
        selected_percentiles=[50.0],
        ignore_eos=True,
        goodput_config_dict={},
        max_concurrency=concurrency,
        lora_modules=None,
        extra_headers=None,
        extra_body={"temperature": 0, "seed": 42},
        ready_check_timeout_sec=0,
        probe_request_rate=0,
    )
    result = asyncio.run(asyncio.wait_for(operation, timeout=BENCHMARK_TIMEOUT_SECONDS))
    assert result["completed"] == len(samples) and result["failed"] == 0, result["errors"]
    assert result["input_lens"] == [PROMPT_TOKEN_COUNT] * len(samples)
    output_tokens = samples[0].expected_output_len
    assert all(sample.expected_output_len == output_tokens for sample in samples)
    assert result["output_lens"] == [output_tokens] * len(samples)
    assert result["median_ttft_ms"] > 0 and result["request_throughput"] > 0
    sources = wait_for_prompt_token_sources(server, baseline, len(samples) * PROMPT_TOKEN_COUNT)
    # Keep scalars, per-request latency and outputs; omit bulky per-token timing arrays from the CI report.
    measurements = {
        **{key: value for key, value in result.items() if isinstance(value, (int, float))},
        "ttfts": result["ttfts"],
        "latencies": result["latencies"],
        "generated_texts": result["generated_texts"],
        "prompt_tokens_by_source": sources,
    }
    if output_tokens == 1:
        # No interval after the first token exists; upstream's zero TPOT is not a measured decode speed.
        for metric in measurements:
            if "tpot" in metric:
                measurements[metric] = None
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


def read_prompt_token_sources(server: RemoteOpenAIServer) -> dict[str, float]:
    response = requests.get(server.url_for("metrics"), timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    sources: dict[str, float] = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name == "vllm:prompt_tokens_by_source_total":
                source = sample.labels["source"]
                sources[source] = sources.get(source, 0.0) + sample.value
    assert set(sources) == set(PromptTokenStats.ALL_SOURCES), "Upstream prompt-token provenance metrics are required"
    return sources


def wait_for_prompt_token_sources(
    server: RemoteOpenAIServer, baseline: dict[str, float], expected_tokens: int
) -> dict[str, float]:
    deadline = time.monotonic() + CONTROL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        sources = read_prompt_token_sources(server)
        delta = {source: sources[source] - baseline[source] for source in baseline}
        assert all(value >= 0 for value in delta.values()), "Prompt-token counters unexpectedly reset"
        assert sum(delta.values()) <= expected_tokens, "Unmeasured requests contaminated benchmark provenance"
        if sum(delta.values()) == expected_tokens:
            return delta
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"Prompt-token metrics did not account for all {expected_tokens} measured input tokens")
