# SPDX-License-Identifier: Apache-2.0
"""Temporary A/B benchmarks for the AscendStore lookup payload RFC."""

import asyncio
import hashlib
import json
import math
import statistics
import time
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import requests
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port
from vllm.v1.serial_utils import MsgpackEncoder

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.nightly.single_node.models.scripts.kv_pool_runtime import SingleNodeMemcacheManager
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LookupHashMode

MODEL = "Qwen/Qwen3-8B"
MODEL_MAX_LEN = 40960
PROMPT_SEED = " ".join(f"lookup payload marker {index}" for index in range(256))

pytestmark = pytest.mark.e2e_model(MODEL)

RPC_FULL_HASHES = 4096
RPC_SUFFIX_HASHES = 32
RPC_WARMUPS = 50
RPC_SAMPLES = 1000

BLOCK_SIZE = 128
COMMON_PREFIX_BLOCKS = 312
SERVING_REQUESTS = 64
SERVING_OUTPUT_TOKENS = 8
GPU_BLOCKS = 640
SERVING_INPUT_TOKENS = (COMMON_PREFIX_BLOCKS + 1) * BLOCK_SIZE
assert SERVING_INPUT_TOKENS + SERVING_OUTPUT_TOKENS <= MODEL_MAX_LEN
assert COMMON_PREFIX_BLOCKS + 2 * SERVING_REQUESTS + 1 <= GPU_BLOCKS
SERVING_SAMPLES_PER_INSTANCE = 5
SERVING_MODE_ORDERS = (
    (LookupHashMode.FULL, LookupHashMode.SUFFIX),
    (LookupHashMode.SUFFIX, LookupHashMode.FULL),
    (LookupHashMode.SUFFIX, LookupHashMode.FULL),
    (LookupHashMode.FULL, LookupHashMode.SUFFIX),
)
METRIC_STABLE_READS = 2
METRIC_POLL_INTERVAL_SECONDS = 0.2
METRIC_SETTLE_TIMEOUT_SECONDS = 10

LOOKUP_METRICS = (
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:ascend_store_load_get_keys_total",
    "vllm:ascend_store_lookup_hashes_sent_total",
    "vllm:ascend_store_lookup_hashes_omitted_total",
)
QUIESCENCE_METRICS = (
    "vllm:ascend_store_delayed_release_requests",
    "vllm:ascend_store_delayed_release_blocks",
)
OBSERVED_METRICS = (*LOOKUP_METRICS, *QUIESCENCE_METRICS)
WORKLOAD_METRICS = LOOKUP_METRICS[:3]
SERVING_PERFORMANCE_FIELDS = (
    "request_throughput_per_second",
    "output_throughput_tokens_per_second",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "mean_e2el_ms",
    "p50_e2el_ms",
    "p90_e2el_ms",
    "p99_e2el_ms",
    "mean_itl_ms",
    "p99_itl_ms",
)


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(samples: list[float]) -> dict[str, float]:
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples)
    return {
        "mean_us": mean * 1e6,
        "stdev_us": stdev * 1e6,
        "coefficient_of_variation_percent": stdev / mean * 100,
        "p50_us": percentile(samples, 50) * 1e6,
        "p90_us": percentile(samples, 90) * 1e6,
        "p99_us": percentile(samples, 99) * 1e6,
    }


def value_distribution(values: list[float]) -> dict[str, float | int]:
    assert values
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = stdev / math.sqrt(len(values))
    return {
        "count": len(values),
        "mean": mean,
        "stdev": stdev,
        "standard_error": standard_error,
        "mean_ci95_normal_low": mean - 1.96 * standard_error,
        "mean_ci95_normal_high": mean + 1.96 * standard_error,
        "coefficient_of_variation_percent": stdev / mean * 100 if mean else 0.0,
        "min": min(values),
        "p10": percentile(values, 10),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "max": max(values),
    }


class LookupStub:
    def __init__(self, expected_calls: int):
        self.expected_calls = expected_calls
        self.calls = 0
        self.hash_counts: dict[LookupHashMode, list[int]] = {
            LookupHashMode.FULL: [],
            LookupHashMode.SUFFIX: [],
        }
        self.server: Any = None

    def lookup_scheduler(
        self,
        token_len,
        block_hashes,
        kv_cache_group_ids,
        use_layerwise,
        hbm_hit_tokens,
        lookup_hash_mode,
    ):
        self.calls += 1
        self.hash_counts[lookup_hash_mode].append(len(block_hashes))
        if self.calls == self.expected_calls:
            assert self.server is not None
            self.server.running = False
        return token_len


def lookup_config() -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank=0),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"lookup_rpc_port": get_open_port()},
        ),
    )


def encoded_hash_bytes(block_hashes: list[bytes]) -> int:
    frames = MsgpackEncoder().encode([block_hash.hex() for block_hash in block_hashes])
    return sum(len(frame) for frame in frames)


def test_lookup_rpc_payload_benchmark():
    """Measure the Msgpack/ZMQ lookup round trip without model execution."""
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import LookupKeyServer
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import LookupKeyClient

    full_hashes = [index.to_bytes(32, "big") for index in range(RPC_FULL_HASHES)]
    suffix_hashes = full_hashes[-RPC_SUFFIX_HASHES:]
    calls_per_mode = RPC_WARMUPS + RPC_SAMPLES
    stub = LookupStub(expected_calls=2 * calls_per_mode)
    config = lookup_config()
    server = LookupKeyServer(stub, config)
    stub.server = server
    client = LookupKeyClient(config)
    samples: dict[LookupHashMode, list[float]] = {
        LookupHashMode.FULL: [],
        LookupHashMode.SUFFIX: [],
    }

    try:
        for _ in range(RPC_WARMUPS):
            client.lookup(
                RPC_FULL_HASHES * 16,
                full_hashes,
                hbm_hit_tokens=(RPC_FULL_HASHES - RPC_SUFFIX_HASHES) * 16,
                lookup_hash_mode=LookupHashMode.FULL,
            )
            client.lookup(
                RPC_FULL_HASHES * 16,
                suffix_hashes,
                hbm_hit_tokens=(RPC_FULL_HASHES - RPC_SUFFIX_HASHES) * 16,
                lookup_hash_mode=LookupHashMode.SUFFIX,
            )

        for sample_index in range(RPC_SAMPLES):
            order: tuple[tuple[LookupHashMode, list[bytes]], ...] = (
                (LookupHashMode.FULL, full_hashes),
                (LookupHashMode.SUFFIX, suffix_hashes),
            )
            if sample_index % 2:
                order = tuple(reversed(order))
            for mode, hashes in order:
                start = time.perf_counter()
                assert (
                    client.lookup(
                        RPC_FULL_HASHES * 16,
                        hashes,
                        hbm_hit_tokens=(RPC_FULL_HASHES - RPC_SUFFIX_HASHES) * 16,
                        lookup_hash_mode=mode,
                    )
                    == RPC_FULL_HASHES * 16
                )
                samples[mode].append(time.perf_counter() - start)
    finally:
        client.close()
        server.thread.join(timeout=5)
        server.close()

    full_bytes = encoded_hash_bytes(full_hashes)
    suffix_bytes = encoded_hash_bytes(suffix_hashes)
    result = {
        "config": {"warmups": RPC_WARMUPS, "samples_per_mode": RPC_SAMPLES},
        "hashes": {"full": len(full_hashes), "suffix": len(suffix_hashes)},
        "encoded_hash_bytes": {"full": full_bytes, "suffix": suffix_bytes},
        "payload_reduction_percent": (1 - suffix_bytes / full_bytes) * 100,
        "latency": {
            "full": latency_summary(samples[LookupHashMode.FULL]),
            "suffix": latency_summary(samples[LookupHashMode.SUFFIX]),
        },
        "mean_latency_improvement_percent": (
            1 - statistics.fmean(samples[LookupHashMode.SUFFIX]) / statistics.fmean(samples[LookupHashMode.FULL])
        )
        * 100,
    }
    print("\nKVPP_LOOKUP_RPC_BENCHMARK=" + json.dumps(result, sort_keys=True))
    assert stub.hash_counts[LookupHashMode.FULL] == [RPC_FULL_HASHES] * calls_per_mode
    assert stub.hash_counts[LookupHashMode.SUFFIX] == [RPC_SUFFIX_HASHES] * calls_per_mode
    assert suffix_bytes < full_bytes * 0.01
    assert statistics.fmean(samples[LookupHashMode.SUFFIX]) < statistics.fmean(samples[LookupHashMode.FULL])


def metric_total(metrics_text: str, name: str) -> float:
    return sum(
        float(line.split()[-1]) for line in metrics_text.splitlines() if line.startswith((f"{name}{{", f"{name} "))
    )


def metric_snapshot(url_root: str) -> dict[str, float]:
    response = requests.get(url_root + "/metrics", timeout=30)
    response.raise_for_status()
    return {name: metric_total(response.text, name) for name in OBSERVED_METRICS}


def stable_metric_snapshot(url_root: str) -> dict[str, float]:
    """Wait until prior requests stop changing the counters we measure."""
    deadline = time.monotonic() + METRIC_SETTLE_TIMEOUT_SECONDS
    previous = metric_snapshot(url_root)
    stable_reads = 0
    while time.monotonic() < deadline:
        time.sleep(METRIC_POLL_INTERVAL_SECONDS)
        current = metric_snapshot(url_root)
        saves_quiescent = all(current[name] == 0 for name in QUIESCENCE_METRICS)
        if current == previous and saves_quiescent:
            stable_reads += 1
            if stable_reads >= METRIC_STABLE_READS:
                return current
        else:
            stable_reads = 0
            previous = current
    raise TimeoutError(f"Lookup metrics did not settle: {previous}")


def wait_for_batch_metrics(
    url_root: str,
    before: dict[str, float],
    expected_sent_hashes: int,
) -> dict[str, float]:
    deadline = time.monotonic() + METRIC_SETTLE_TIMEOUT_SECONDS
    current = metric_snapshot(url_root)
    sent_metric = "vllm:ascend_store_lookup_hashes_sent_total"
    while current[sent_metric] - before[sent_metric] < expected_sent_hashes:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Lookup metrics did not publish the completed batch: "
                f"expected_sent={expected_sent_hashes}, before={before}, current={current}"
            )
        time.sleep(METRIC_POLL_INTERVAL_SECONDS)
        current = metric_snapshot(url_root)
    return stable_metric_snapshot(url_root)


def metric_deltas(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {name: after[name] - before[name] for name in LOOKUP_METRICS}


def output_digest(outputs: list[str]) -> str:
    encoded = json.dumps(outputs, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def mismatch_indices(actual: list[str], expected: list[str]) -> list[int]:
    assert len(actual) == len(expected)
    return [index for index, (left, right) in enumerate(zip(actual, expected, strict=True)) if left != right]


def complete(url_root: str, prompt: list[int]) -> str:
    response = requests.post(
        url_root + "/v1/completions",
        json={
            "model": "kvpp-test",
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": SERVING_OUTPUT_TOKENS,
            "ignore_eos": True,
        },
        timeout=900,
    )
    response.raise_for_status()
    result = response.json()
    assert result["choices"][0]["finish_reason"] == "length"
    return result["choices"][0]["text"]


@dataclass
class RequestTiming:
    text: str
    ttft: float
    e2el: float
    output_tokens: int
    itls: list[float]


async def stream_completion(client: httpx.AsyncClient, url_root: str, prompt: list[int]) -> RequestTiming:
    start = time.perf_counter()
    first_token_at: float | None = None
    previous_token_at: float | None = None
    itls: list[float] = []
    text_parts: list[str] = []
    output_tokens = 0
    async with client.stream(
        "POST",
        url_root + "/v1/completions",
        json={
            "model": "kvpp-test",
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": SERVING_OUTPUT_TOKENS,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        timeout=900,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage")
            if usage is not None:
                output_tokens = int(usage.get("completion_tokens", output_tokens))
            choices = chunk.get("choices") or []
            if not choices or not choices[0].get("text"):
                continue
            now = time.perf_counter()
            text_parts.append(choices[0]["text"])
            if first_token_at is None:
                first_token_at = now
            if previous_token_at is not None:
                itls.append(now - previous_token_at)
            previous_token_at = now
    end = time.perf_counter()
    assert first_token_at is not None
    if output_tokens == 0:
        output_tokens = len(itls) + 1
    return RequestTiming(
        text="".join(text_parts),
        ttft=first_token_at - start,
        e2el=end - start,
        output_tokens=output_tokens,
        itls=itls,
    )


async def run_concurrent_requests(url_root: str, prompts: list[list[int]]) -> tuple[list[RequestTiming], float]:
    limits = httpx.Limits(max_connections=len(prompts), max_keepalive_connections=len(prompts))
    async with httpx.AsyncClient(limits=limits) as client:
        start = time.perf_counter()
        timings = await asyncio.gather(*(stream_completion(client, url_root, prompt) for prompt in prompts))
        return timings, time.perf_counter() - start


def serving_summary(timings: list[RequestTiming], wall_time: float) -> dict[str, float]:
    ttfts = [timing.ttft for timing in timings]
    e2els = [timing.e2el for timing in timings]
    itls = [itl for timing in timings for itl in timing.itls]
    output_tokens = sum(timing.output_tokens for timing in timings)
    return {
        "request_throughput_per_second": len(timings) / wall_time,
        "output_throughput_tokens_per_second": output_tokens / wall_time,
        "mean_ttft_ms": statistics.fmean(ttfts) * 1e3,
        "p50_ttft_ms": percentile(ttfts, 50) * 1e3,
        "p90_ttft_ms": percentile(ttfts, 90) * 1e3,
        "p99_ttft_ms": percentile(ttfts, 99) * 1e3,
        "mean_e2el_ms": statistics.fmean(e2els) * 1e3,
        "p50_e2el_ms": percentile(e2els, 50) * 1e3,
        "p90_e2el_ms": percentile(e2els, 90) * 1e3,
        "p99_e2el_ms": percentile(e2els, 99) * 1e3,
        "mean_itl_ms": statistics.fmean(itls) * 1e3 if itls else 0,
        "p99_itl_ms": percentile(itls, 99) * 1e3 if itls else 0,
    }


def expected_lookup_hash_metrics(mode: LookupHashMode) -> tuple[int, int]:
    if mode is LookupHashMode.FULL:
        return SERVING_REQUESTS * (COMMON_PREFIX_BLOCKS + 1), 0
    return SERVING_REQUESTS, SERVING_REQUESTS * COMMON_PREFIX_BLOCKS


def validate_batch_metrics(
    deltas: dict[str, float],
    mode: LookupHashMode,
    *,
    expect_external_hit: bool,
) -> list[str]:
    errors = []
    expected_sent, expected_omitted = expected_lookup_hash_metrics(mode)
    sent = deltas["vllm:ascend_store_lookup_hashes_sent_total"]
    omitted = deltas["vllm:ascend_store_lookup_hashes_omitted_total"]
    if sent != expected_sent:
        errors.append(f"sent_hashes={sent}, expected={expected_sent}")
    if omitted != expected_omitted:
        errors.append(f"omitted_hashes={omitted}, expected={expected_omitted}")

    prefix_hits = deltas["vllm:prefix_cache_hits_total"]
    external_hits = deltas["vllm:external_prefix_cache_hits_total"]
    load_keys = deltas["vllm:ascend_store_load_get_keys_total"]
    if prefix_hits <= 0:
        errors.append(f"prefix_cache_hits={prefix_hits}, expected>0")
    if expect_external_hit:
        if external_hits <= 0:
            errors.append(f"external_prefix_cache_hits={external_hits}, expected>0")
        if load_keys <= 0:
            errors.append(f"load_get_keys={load_keys}, expected>0")
    else:
        if external_hits != 0:
            errors.append(f"external_prefix_cache_hits={external_hits}, expected=0")
        if load_keys != 0:
            errors.append(f"load_get_keys={load_keys}, expected=0")
    return errors


def run_serving_batch(
    url_root: str,
    prompts: list[list[int]],
    mode: LookupHashMode,
    expected_outputs: list[str] | None,
    *,
    expect_external_hit: bool,
) -> tuple[dict[str, Any], list[str]]:
    before = stable_metric_snapshot(url_root)
    timings, wall_time = asyncio.run(run_concurrent_requests(url_root, prompts))
    expected_sent, _ = expected_lookup_hash_metrics(mode)
    after = wait_for_batch_metrics(url_root, before, expected_sent)
    outputs = [timing.text for timing in timings]
    mismatches = mismatch_indices(outputs, expected_outputs) if expected_outputs is not None else []
    deltas = metric_deltas(after, before)
    errors = validate_batch_metrics(deltas, mode, expect_external_hit=expect_external_hit)
    summary: dict[str, Any] = serving_summary(timings, wall_time)
    summary.update(deltas)
    summary.update(
        {
            "output_digest": output_digest(outputs),
            "output_mismatch_count": len(mismatches),
            "output_mismatch_indices": mismatches,
            "validation_errors": errors,
            "valid": not mismatches and not errors,
        }
    )
    return summary, outputs


def reset_and_warm_hbm(url_root: str, common_prefix: list[int]) -> None:
    response = requests.post(url_root + "/reset_prefix_cache", timeout=30)
    response.raise_for_status()
    complete(url_root, common_prefix)
    stable_metric_snapshot(url_root)


def memcache_config() -> MemcacheKVPoolConfig:
    return MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config={
            "meta": {
                "ock.mmc.log_level": "info",
                "ock.mmc.meta_service.metrics_url": f"http://127.0.0.1:{get_open_port()}",
            },
            "local": {
                "ock.mmc.log_level": "info",
                "ock.mmc.local_service.world_size": 1,
                "ock.mmc.local_service.protocol": "device_sdma",
                "ock.mmc.local_service.dram.size": "16GB",
            },
        },
    )


def run_serving_mode(
    mode: LookupHashMode,
    tmp_path,
    round_index: int,
    order_position: int,
) -> dict[str, Any]:
    from tests.e2e.conftest import RemoteOpenAIServer

    config = memcache_config()
    isolation_id = uuid.uuid4().hex
    case_name = f"{tmp_path.name}-r{round_index}-p{order_position}-{mode.value}-{isolation_id}"
    with SingleNodeMemcacheManager(config, case_name) as pool:
        port = get_open_port()
        # This is intentionally a low-noise near-context-limit scenario for
        # lookup payload savings. A dense model and short output minimize
        # unrelated model, collective, and decode work, while the long local
        # prefix maximizes the hashes that FULL sends and SUFFIX omits without
        # exceeding the model's declared context length.
        args = [
            "--served-model-name",
            "kvpp-test",
            "--trust-remote-code",
            "--tensor-parallel-size",
            "1",
            "--enforce-eager",
            "--max-model-len",
            str(MODEL_MAX_LEN),
            "--max-num-batched-tokens",
            "4096",
            "--max-num-seqs",
            str(SERVING_REQUESTS),
            "--block-size",
            str(BLOCK_SIZE),
            "--num-gpu-blocks-override",
            str(GPU_BLOCKS),
            "--gpu-memory-utilization",
            "0.8",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--seed",
            "42",
            "--generation-config",
            "vllm",
            "--port",
            str(port),
            "--additional-config",
            '{"enable_kvpp":true}',
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_producer",
                    "kv_connector_extra_config": {
                        "lookup_rpc_port": "0",
                        "backend": "memcache",
                        "use_layerwise": False,
                        "load_async": False,
                        "lookup_hash_mode": mode.value,
                    },
                }
            ),
        ]
        with RemoteOpenAIServer(
            maybe_model_redirect(MODEL),
            args,
            server_port=port,
            auto_port=False,
            env_dict={**pool.server_envs, "VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_SERVER_DEV_MODE": "1"},
        ) as server:
            tokenized = requests.post(
                server.url_for("tokenize"),
                json={"model": "kvpp-test", "prompt": PROMPT_SEED},
                timeout=30,
            )
            tokenized.raise_for_status()
            seed_tokens = tokenized.json()["tokens"]
            common_size = COMMON_PREFIX_BLOCKS * BLOCK_SIZE
            common_prefix = (seed_tokens * ((common_size + len(seed_tokens) - 1) // len(seed_tokens)))[:common_size]
            suffix_seed = (seed_tokens * ((BLOCK_SIZE + len(seed_tokens) - 1) // len(seed_tokens)))[:BLOCK_SIZE]
            prompts = [common_prefix + suffix_seed[index:] + suffix_seed[:index] for index in range(SERVING_REQUESTS)]
            assert all(len(prompt) == SERVING_INPUT_TOKENS for prompt in prompts)

            # This first concurrent batch only populates the fresh pool. It is
            # not a correctness oracle: the behavior under test is whether
            # FULL and SUFFIX produce equivalent loaded results.
            complete(server.url_root, common_prefix)
            baseline, _ = run_serving_batch(
                server.url_root,
                prompts,
                mode,
                expected_outputs=None,
                expect_external_hit=False,
            )

            # Exercise the exact load path once without timing it. Besides
            # warming runtime state, this distinguishes a cache/load failure
            # from a noisy performance sample before measurements begin.
            reset_and_warm_hbm(server.url_root, common_prefix)
            load_warmup, _ = run_serving_batch(
                server.url_root,
                prompts,
                mode,
                expected_outputs=None,
                expect_external_hit=True,
            )

            run_report: dict[str, Any] = {
                "round": round_index,
                "order_position": order_position,
                "mode": mode.value,
                "isolation_id": isolation_id,
                "baseline": baseline,
                "load_warmup": load_warmup,
                "samples": [],
            }
            if not baseline["valid"] or not load_warmup["valid"]:
                run_report["valid"] = False
                return run_report

            # Every sample starts from the same explicit state: a fresh HBM
            # cache containing only the common prefix and a pre-populated,
            # instance-local external pool containing every suffix.
            for sample_index in range(SERVING_SAMPLES_PER_INSTANCE):
                reset_and_warm_hbm(server.url_root, common_prefix)
                sample, _ = run_serving_batch(
                    server.url_root,
                    prompts,
                    mode,
                    expected_outputs=None,
                    expect_external_hit=True,
                )
                sample["sample_index"] = sample_index
                run_report["samples"].append(sample)

            run_report["valid"] = all(sample["valid"] for sample in run_report["samples"])
            return run_report


def aggregate_mode_runs(runs: list[dict[str, Any]], mode: LookupHashMode) -> dict[str, Any]:
    mode_runs = [run for run in runs if run["mode"] == mode.value]
    samples = [sample for run in mode_runs for sample in run["samples"]]
    valid_samples = [sample for sample in samples if sample["valid"]]

    def distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            field: value_distribution([sample[field] for sample in rows])
            for field in (*SERVING_PERFORMANCE_FIELDS, *LOOKUP_METRICS)
            if rows
        }

    return {
        "instances": len(mode_runs),
        "valid_instances": sum(run["valid"] for run in mode_runs),
        "samples": len(samples),
        "valid_samples": len(valid_samples),
        "requests_in_valid_samples": len(valid_samples) * SERVING_REQUESTS,
        "all_sample_distributions": distributions(samples),
        "valid_sample_distributions": distributions(valid_samples),
    }


def paired_serving_comparison(runs: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = []
    validation_errors = []
    digests_by_mode = {
        mode.value: {
            sample["output_digest"]
            for run in runs
            if run["mode"] == mode.value
            for sample in run["samples"]
            if sample["valid"]
        }
        for mode in (LookupHashMode.FULL, LookupHashMode.SUFFIX)
    }
    for mode, digests in digests_by_mode.items():
        if len(digests) != 1:
            validation_errors.append(f"mode={mode}: expected one stable output digest, got={sorted(digests)}")
    if all(len(digests) == 1 for digests in digests_by_mode.values()) and (
        digests_by_mode[LookupHashMode.FULL.value] != digests_by_mode[LookupHashMode.SUFFIX.value]
    ):
        validation_errors.append(
            "stable output digests differ between modes: "
            f"full={sorted(digests_by_mode[LookupHashMode.FULL.value])}, "
            f"suffix={sorted(digests_by_mode[LookupHashMode.SUFFIX.value])}"
        )
    for round_index in range(len(SERVING_MODE_ORDERS)):
        by_mode = {run["mode"]: run for run in runs if run["round"] == round_index}
        full_run = by_mode[LookupHashMode.FULL.value]
        suffix_run = by_mode[LookupHashMode.SUFFIX.value]
        if len(full_run["samples"]) != len(suffix_run["samples"]):
            validation_errors.append(
                f"round={round_index}: sample counts differ, "
                f"full={len(full_run['samples'])}, suffix={len(suffix_run['samples'])}"
            )
            continue
        for sample_index, (full, suffix) in enumerate(zip(full_run["samples"], suffix_run["samples"], strict=True)):
            errors = []
            if not full["valid"]:
                errors.append("full sample is invalid")
            if not suffix["valid"]:
                errors.append("suffix sample is invalid")
            if full["output_digest"] != suffix["output_digest"]:
                errors.append(f"output digests differ: full={full['output_digest']}, suffix={suffix['output_digest']}")
            for name in WORKLOAD_METRICS:
                if full[name] != suffix[name]:
                    errors.append(f"{name}: full={full[name]}, suffix={suffix[name]}")
            if errors:
                validation_errors.append(f"round={round_index}, sample={sample_index}: " + "; ".join(errors))
                continue
            comparisons.append(
                {
                    "round": round_index,
                    "sample": sample_index,
                    "request_throughput_gain_percent": (
                        suffix["request_throughput_per_second"] / full["request_throughput_per_second"] - 1
                    )
                    * 100,
                    "output_throughput_gain_percent": (
                        suffix["output_throughput_tokens_per_second"] / full["output_throughput_tokens_per_second"] - 1
                    )
                    * 100,
                    "mean_ttft_reduction_percent": (1 - suffix["mean_ttft_ms"] / full["mean_ttft_ms"]) * 100,
                    "p99_ttft_reduction_percent": (1 - suffix["p99_ttft_ms"] / full["p99_ttft_ms"]) * 100,
                    "mean_e2el_reduction_percent": (1 - suffix["mean_e2el_ms"] / full["mean_e2el_ms"]) * 100,
                    "p99_e2el_reduction_percent": (1 - suffix["p99_e2el_ms"] / full["p99_e2el_ms"]) * 100,
                    "mean_itl_reduction_percent": (1 - suffix["mean_itl_ms"] / full["mean_itl_ms"]) * 100,
                    "p99_itl_reduction_percent": (1 - suffix["p99_itl_ms"] / full["p99_itl_ms"]) * 100,
                }
            )

    comparison_fields = (
        "request_throughput_gain_percent",
        "output_throughput_gain_percent",
        "mean_ttft_reduction_percent",
        "p99_ttft_reduction_percent",
        "mean_e2el_reduction_percent",
        "p99_e2el_reduction_percent",
        "mean_itl_reduction_percent",
        "p99_itl_reduction_percent",
    )
    distributions = {
        field: {
            **value_distribution([comparison[field] for comparison in comparisons]),
            "suffix_win_count": sum(comparison[field] > 0 for comparison in comparisons),
        }
        for field in comparison_fields
        if comparisons
    }
    instance_pair_means = []
    for round_index in range(len(SERVING_MODE_ORDERS)):
        samples = [comparison for comparison in comparisons if comparison["round"] == round_index]
        if len(samples) != SERVING_SAMPLES_PER_INSTANCE:
            continue
        instance_pair_means.append(
            {
                "round": round_index,
                **{field: statistics.fmean(sample[field] for sample in samples) for field in comparison_fields},
            }
        )
    instance_pair_distributions = {
        field: {
            **value_distribution([comparison[field] for comparison in instance_pair_means]),
            "suffix_win_count": sum(comparison[field] > 0 for comparison in instance_pair_means),
        }
        for field in comparison_fields
        if instance_pair_means
    }
    return {
        "pairs": len(comparisons),
        "expected_pairs": len(SERVING_MODE_ORDERS) * SERVING_SAMPLES_PER_INSTANCE,
        "validation_errors": validation_errors,
        "sample_distributions": distributions,
        "instance_pair_mean_distributions": instance_pair_distributions,
        "instance_pair_means": instance_pair_means,
        "raw_pairs": comparisons,
    }


@wait_until_npu_memory_free()
def test_vllm_serve_lookup_payload_benchmark(tmp_path):
    """Compare isolated, counterbalanced serving runs after a long HBM hit."""
    pytest.importorskip("memcache_hybrid")
    runs = []
    for round_index, mode_order in enumerate(SERVING_MODE_ORDERS):
        for order_position, mode in enumerate(mode_order):
            runs.append(run_serving_mode(mode, tmp_path, round_index, order_position))

    paired = paired_serving_comparison(runs)
    report = {
        "config": {
            "mode_orders": [[mode.value for mode in order] for order in SERVING_MODE_ORDERS],
            "model": MODEL,
            "scenario": "lookup_payload_stress",
            "tensor_parallel_size": 1,
            "expert_parallel": False,
            "async_scheduling": False,
            "load_async": False,
            "instances_per_mode": len(SERVING_MODE_ORDERS),
            "samples_per_instance": SERVING_SAMPLES_PER_INSTANCE,
            "requests_per_sample": SERVING_REQUESTS,
            "measured_requests_per_mode": (len(SERVING_MODE_ORDERS) * SERVING_SAMPLES_PER_INSTANCE * SERVING_REQUESTS),
            "max_model_len": MODEL_MAX_LEN,
            "input_tokens_per_request": SERVING_INPUT_TOKENS,
            "common_prefix_blocks": COMMON_PREFIX_BLOCKS,
            "suffix_blocks": 1,
            "output_tokens": SERVING_OUTPUT_TOKENS,
        },
        "aggregate": {
            mode.value: aggregate_mode_runs(runs, mode) for mode in (LookupHashMode.FULL, LookupHashMode.SUFFIX)
        },
        "paired": paired,
        "runs": runs,
    }
    report["valid"] = (
        all(run["valid"] for run in runs)
        and not paired["validation_errors"]
        and paired["pairs"] == paired["expected_pairs"]
    )
    print("\nKVPP_LOOKUP_PAYLOAD_SERVING_BENCHMARK=" + json.dumps(report, sort_keys=True))
    assert report["valid"], json.dumps(
        {
            "invalid_runs": [{"round": run["round"], "mode": run["mode"]} for run in runs if not run["valid"]],
            "pair_validation_errors": paired["validation_errors"],
        },
        sort_keys=True,
    )
