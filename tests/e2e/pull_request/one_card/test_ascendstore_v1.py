# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare production/v1 serve latency for cold and fully external prefixes."""

import asyncio
import json
import shlex
import shutil
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.serve import TaskType, benchmark
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_ports_list
from vllm.v1.metrics.stats import PromptTokenStats

from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.conftest import RemoteOpenAIServer

MODEL = "Qwen/Qwen3-8B"
BLOCK_SIZE = 128
REUSABLE_TOKEN_COUNT = 8192
PROMPT_TOKEN_COUNT = REUSABLE_TOKEN_COUNT + 1
OUTPUT_TOKEN_COUNT = 32
REQUEST_COUNT = 8
CONCURRENCIES = (1, 8)
# Sixteen independent 8K prefixes plus warmup occupy about 20 GiB of BF16 KV.
POOL_MEMORY_BYTES = 32 << 30
LOCAL_BUFFER_BYTES = 1 << 30
HTTP_TIMEOUT_SECONDS = 30
CONTROL_TIMEOUT_SECONDS = 60
BENCHMARK_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 0.1
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

pytestmark = pytest.mark.e2e_model(MODEL)


def test_production_and_v1_serve_cold_and_external_hit_performance(tmp_path: Path) -> None:
    """Report upstream benchmark timings; gate transfer provenance and outputs, not an uncalibrated speedup."""
    assert shutil.which("mooncake_master") is not None, "The serve benchmark requires mooncake_master on PATH"
    model_path = maybe_model_redirect(MODEL)
    tokenizer = get_tokenizer(model_path)
    inputs = {concurrency: _make_requests(tokenizer, str(concurrency)) for concurrency in CONCURRENCIES}
    warmup = _make_requests(tokenizer, "warmup", count=1)[0]
    # Distinct first blocks prevent local or external sharing between measured requests and warmup.
    first_blocks = [tuple(sample.prompt[:BLOCK_SIZE]) for batch in inputs.values() for sample in batch]
    assert len(set(first_blocks + [tuple(warmup.prompt[:BLOCK_SIZE])])) == len(first_blocks) + 1

    observations = {
        implementation: _exercise_connector(tmp_path, model_path, tokenizer, inputs, warmup, connector_name, module)
        for implementation, connector_name, module in CONNECTORS
    }
    comparisons: dict[str, dict[str, dict[str, float]]] = {}
    for concurrency in CONCURRENCIES:
        comparisons[str(concurrency)] = {}
        for phase in ("cold", "external_hit"):
            production = observations["production"][str(concurrency)][phase]
            candidate = observations["v1"][str(concurrency)][phase]
            assert candidate["generated_texts"] == production["generated_texts"], (concurrency, phase)
            comparisons[str(concurrency)][phase] = {
                "v1_over_production_median_ttft": candidate["median_ttft_ms"] / production["median_ttft_ms"],
                "v1_over_production_request_throughput": (
                    candidate["request_throughput"] / production["request_throughput"]
                ),
            }
    report = {
        "model": MODEL,
        "prompt_tokens": PROMPT_TOKEN_COUNT,
        "external_hit_tokens_per_request": REUSABLE_TOKEN_COUNT,
        "output_tokens": OUTPUT_TOKEN_COUNT,
        "requests_per_phase": REQUEST_COUNT,
        "reset_in_benchmark_timing": False,
        "measurements": observations,
        "comparisons": comparisons,
    }
    serialized = json.dumps(report, indent=2)
    (tmp_path / "ascendstore-serve-performance.json").write_text(serialized, encoding="utf-8")
    # The existing selected-test log artifact preserves the complete report without CI changes.
    print(f"\nASCENDSTORE_SERVE_PERFORMANCE\n{serialized}")


def _exercise_connector(
    tmp_path: Path,
    model_path: str,
    tokenizer: TokenizerLike,
    inputs: dict[int, list[SampleRequest]],
    warmup: SampleRequest,
    connector_name: str,
    connector_module: str,
) -> dict[str, Any]:
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
        },
    }
    server_args = shlex.split(f"""
        --host 127.0.0.1 --port {server_port} --served-model-name {MODEL}
        --dtype bfloat16 --max-model-len 16384 --max-num-batched-tokens 16384
        --max-num-seqs {max(CONCURRENCIES)} --block-size {BLOCK_SIZE}
        --tensor-parallel-size 1 --pipeline-parallel-size 1 --gpu-memory-utilization 0.8
        --enable-prefix-caching --prefix-caching-hash-algo sha256 --no-enable-chunked-prefill
        --enforce-eager --no-async-scheduling --seed 42
    """) + ["--kv-transfer-config", json.dumps(transfer_config)]
    # This optional NPU service dependency is needed only when executing our E2E.
    pool_runtime = import_module("tests.e2e.common.single_node.kv_pool_runtime")
    observations = {}
    with pool_runtime.SingleNodeMooncakeManager(pool_config, f"{tmp_path.name}-{connector_name}") as pool:
        server_envs = {
            **pool.server_envs,
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
            baseline = _read_prompt_token_sources(server)
            response = requests.post(
                server.url_for("v1", "completions"),
                timeout=BENCHMARK_TIMEOUT_SECONDS,
                json={
                    "model": MODEL,
                    "prompt": warmup.prompt,
                    "max_tokens": OUTPUT_TOKEN_COUNT,
                    "temperature": 0,
                    "seed": 42,
                    "ignore_eos": True,
                },
            )
            response.raise_for_status()
            assert response.json()["usage"]["completion_tokens"] == OUTPUT_TOKEN_COUNT
            _wait_for_prompt_token_sources(server, baseline, PROMPT_TOKEN_COUNT)

            for concurrency, samples in inputs.items():
                cold_reset = _reset_local_prefix_cache(server)
                cold = _run_benchmark(server, tokenizer, samples, concurrency)
                assert cold["prompt_tokens_by_source"] == {
                    "local_compute": REQUEST_COUNT * PROMPT_TOKEN_COUNT,
                    "local_cache_hit": 0,
                    "external_kv_transfer": 0,
                }, "Cold requests must not reuse local or external KV"

                # Reset may wait for retained Store source blocks; neither this wait nor reset is timed by benchmark().
                external_reset = _reset_local_prefix_cache(server)
                external = _run_benchmark(server, tokenizer, samples, concurrency)
                assert external["prompt_tokens_by_source"] == {
                    "local_compute": REQUEST_COUNT,
                    "local_cache_hit": 0,
                    "external_kv_transfer": REQUEST_COUNT * REUSABLE_TOKEN_COUNT,
                }, "Every reusable prefix must come from External, leaving one token per request to compute"
                observations[str(concurrency)] = {
                    "cold": {"reset_before": cold_reset, **cold},
                    "external_hit": {"reset_before": external_reset, **external},
                }
    return observations


def _make_requests(tokenizer: TokenizerLike, scenario: str, count: int = REQUEST_COUNT) -> list[SampleRequest]:
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
                prompt=header + body + question, prompt_len=PROMPT_TOKEN_COUNT, expected_output_len=OUTPUT_TOKEN_COUNT
            )
        )
    return samples


def _run_benchmark(
    server: RemoteOpenAIServer, tokenizer: TokenizerLike, samples: list[SampleRequest], concurrency: int
) -> dict[str, Any]:
    baseline = _read_prompt_token_sources(server)
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
    )
    result = asyncio.run(asyncio.wait_for(operation, timeout=BENCHMARK_TIMEOUT_SECONDS))
    assert result["completed"] == len(samples) and result["failed"] == 0, result["errors"]
    assert result["input_lens"] == [PROMPT_TOKEN_COUNT] * len(samples)
    assert result["output_lens"] == [OUTPUT_TOKEN_COUNT] * len(samples)
    assert result["median_ttft_ms"] > 0 and result["request_throughput"] > 0
    sources = _wait_for_prompt_token_sources(server, baseline, len(samples) * PROMPT_TOKEN_COUNT)
    # Keep scalars, per-request latency and outputs; omit bulky per-token timing arrays from the CI report.
    return {
        **{key: value for key, value in result.items() if isinstance(value, (int, float))},
        "ttfts": result["ttfts"],
        "latencies": result["latencies"],
        "generated_texts": result["generated_texts"],
        "prompt_tokens_by_source": sources,
    }


def _reset_local_prefix_cache(server: RemoteOpenAIServer) -> dict[str, float | int]:
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


def _read_prompt_token_sources(server: RemoteOpenAIServer) -> dict[str, float]:
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


def _wait_for_prompt_token_sources(
    server: RemoteOpenAIServer, baseline: dict[str, float], expected_tokens: int
) -> dict[str, float]:
    deadline = time.monotonic() + CONTROL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        sources = _read_prompt_token_sources(server)
        delta = {source: sources[source] - baseline[source] for source in baseline}
        assert all(value >= 0 for value in delta.values()), "Prompt-token counters unexpectedly reset"
        assert sum(delta.values()) <= expected_tokens, "Unmeasured requests contaminated benchmark provenance"
        if sum(delta.values()) == expected_tokens:
            return delta
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"Prompt-token metrics did not account for all {expected_tokens} measured input tokens")
