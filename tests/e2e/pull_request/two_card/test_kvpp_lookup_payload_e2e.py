# SPDX-License-Identifier: Apache-2.0
"""Clean end-to-end A/B benchmark for the AscendStore lookup payload RFC.

This file deliberately does not enable lookup timing.  Its timed window only
contains paired serving requests; the companion cost-breakdown test enables
the opt-in scheduler timer in separate server instances.
"""

import asyncio
import hashlib
import json
import math
import random
import statistics
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import requests
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.nightly.single_node.models.scripts.kv_pool_runtime import SingleNodeMemcacheManager
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LookupHashMode

MODEL = "Eco-Tech/Qwen3.5-27B-w8a8-mtp"
TENSOR_PARALLEL_SIZE = 2
MODEL_MAX_LEN = 67584
BLOCK_SIZE = 128
TOTAL_BLOCKS = 512
INPUT_TOKENS = TOTAL_BLOCKS * BLOCK_SIZE
OUTPUT_TOKENS = 8
REQUESTS_PER_SAMPLE = 4
MANIFEST_SEED = 20260915
PROMPT_SEED = " ".join(f"lookup payload evidence marker {index}" for index in range(512))

# A symmetric schedule prevents warmup or thermal drift from being correlated
# with HBM hit length.  The 50%-94% range is intentionally useful but not the
# near-100% best case used by the earlier stress benchmark.
HBM_BLOCK_SCHEDULE = (256, 448, 480, 384, 384, 480, 448, 256)
EXTERNAL_HIT_STRATA = ((0.15, 0.30), (0.30, 0.45), (0.55, 0.70), (0.70, 0.85))
SERVING_MODE_ORDERS = (
    (LookupHashMode.FULL, LookupHashMode.SUFFIX),
    (LookupHashMode.SUFFIX, LookupHashMode.FULL),
)

METRIC_STABLE_READS = 2
METRIC_POLL_INTERVAL_SECONDS = 0.2
METRIC_SETTLE_TIMEOUT_SECONDS = 30

CORE_LOOKUP_METRICS = (
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:ascend_store_load_get_keys_total",
    "vllm:ascend_store_lookup_hashes_sent_total",
    "vllm:ascend_store_lookup_hashes_omitted_total",
)
PROFILE_LOOKUP_METRICS = (
    "vllm:ascend_store_lookup_duration_seconds_total",
    "vllm:ascend_store_lookup_requests_total",
)
QUIESCENCE_METRICS = (
    "vllm:ascend_store_delayed_release_requests",
    "vllm:ascend_store_delayed_release_blocks",
)
WORKLOAD_METRICS = CORE_LOOKUP_METRICS[:3]
SERVING_PERFORMANCE_FIELDS = (
    "request_throughput_per_second",
    "output_throughput_tokens_per_second",
    "batch_wall_seconds",
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

assert INPUT_TOKENS + OUTPUT_TOKENS <= MODEL_MAX_LEN
pytestmark = pytest.mark.e2e_model(MODEL)


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    sample_index: int
    hbm_blocks: int
    external_blocks: tuple[int, ...]

    @property
    def remaining_blocks(self) -> int:
        return TOTAL_BLOCKS - self.hbm_blocks


@dataclass(frozen=True)
class Workload:
    spec: ScenarioSpec
    common_prefix: list[int]
    prompts: list[list[int]]
    preload_prompts: list[list[int]]
    manifest_digest: str


@dataclass
class RequestTiming:
    text: str
    ttft: float
    e2el: float
    output_tokens: int
    itls: list[float]


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


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


def scenario_specs() -> tuple[ScenarioSpec, ...]:
    specs = []
    for sample_index, hbm_blocks in enumerate(HBM_BLOCK_SCHEDULE):
        remaining = TOTAL_BLOCKS - hbm_blocks
        rng = random.Random(MANIFEST_SEED + sample_index)
        external_blocks = [
            max(1, min(remaining - 1, round(remaining * rng.uniform(low, high)))) for low, high in EXTERNAL_HIT_STRATA
        ]
        rng.shuffle(external_blocks)
        specs.append(
            ScenarioSpec(
                scenario_id=f"sample-{sample_index:02d}",
                sample_index=sample_index,
                hbm_blocks=hbm_blocks,
                external_blocks=tuple(external_blocks),
            )
        )
    return tuple(specs)


def warmup_spec() -> ScenarioSpec:
    remaining = TOTAL_BLOCKS - 400
    return ScenarioSpec(
        scenario_id="warmup",
        sample_index=-1,
        hbm_blocks=400,
        external_blocks=tuple(round(remaining * fraction) for fraction in (0.2, 0.4, 0.6, 0.8)),
    )


def cyclic_tokens(seed_tokens: list[int], length: int, offset: int) -> list[int]:
    assert seed_tokens
    start = offset % len(seed_tokens)
    rotated = seed_tokens[start:] + seed_tokens[:start]
    return (rotated * ((length + len(rotated) - 1) // len(rotated)))[:length]


def build_workload(seed_tokens: list[int], spec: ScenarioSpec) -> Workload:
    common_tokens = spec.hbm_blocks * BLOCK_SIZE
    suffix_tokens = spec.remaining_blocks * BLOCK_SIZE
    scenario_offset = (spec.sample_index + 2) * 7919
    common_prefix = cyclic_tokens(seed_tokens, common_tokens, scenario_offset)
    prompts = []
    preload_prompts = []
    for request_index, external_blocks in enumerate(spec.external_blocks):
        suffix = cyclic_tokens(
            seed_tokens,
            suffix_tokens,
            scenario_offset + (request_index + 1) * 104729,
        )
        prompt = common_prefix + suffix
        prompts.append(prompt)
        preload_prompts.append(prompt[: (spec.hbm_blocks + external_blocks) * BLOCK_SIZE])

    manifest = {
        "seed": MANIFEST_SEED,
        "seed_tokens_sha256": hashlib.sha256(json.dumps(seed_tokens, separators=(",", ":")).encode()).hexdigest(),
        "scenario_id": spec.scenario_id,
        "hbm_blocks": spec.hbm_blocks,
        "external_blocks": spec.external_blocks,
        "prompt_offsets": [scenario_offset + (index + 1) * 104729 for index in range(REQUESTS_PER_SAMPLE)],
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    assert len(prompts) == REQUESTS_PER_SAMPLE
    assert all(len(prompt) == INPUT_TOKENS for prompt in prompts)
    return Workload(spec, common_prefix, prompts, preload_prompts, digest)


def metric_total(metrics_text: str, name: str) -> float:
    return sum(
        float(line.split()[-1]) for line in metrics_text.splitlines() if line.startswith((f"{name}{{", f"{name} "))
    )


def observed_metrics(profile_lookup: bool) -> tuple[str, ...]:
    profile_metrics = PROFILE_LOOKUP_METRICS if profile_lookup else ()
    return (*CORE_LOOKUP_METRICS, *profile_metrics, *QUIESCENCE_METRICS)


def metric_snapshot(url_root: str, profile_lookup: bool) -> dict[str, float]:
    response = requests.get(url_root + "/metrics", timeout=30)
    response.raise_for_status()
    return {name: metric_total(response.text, name) for name in observed_metrics(profile_lookup)}


def stable_metric_snapshot(url_root: str, profile_lookup: bool) -> dict[str, float]:
    """Wait until asynchronous saves and connector metric publication settle."""
    deadline = time.monotonic() + METRIC_SETTLE_TIMEOUT_SECONDS
    previous = metric_snapshot(url_root, profile_lookup)
    stable_reads = 0
    while time.monotonic() < deadline:
        time.sleep(METRIC_POLL_INTERVAL_SECONDS)
        current = metric_snapshot(url_root, profile_lookup)
        saves_quiescent = all(current[name] == 0 for name in QUIESCENCE_METRICS)
        if current == previous and saves_quiescent:
            stable_reads += 1
            if stable_reads >= METRIC_STABLE_READS:
                return current
        else:
            stable_reads = 0
            previous = current
    raise TimeoutError(f"AscendStore metrics did not settle: {previous}")


def wait_for_batch_metrics(
    url_root: str,
    before: dict[str, float],
    expected_sent_hashes: int,
    profile_lookup: bool,
) -> dict[str, float]:
    deadline = time.monotonic() + METRIC_SETTLE_TIMEOUT_SECONDS
    sent_metric = "vllm:ascend_store_lookup_hashes_sent_total"
    current = metric_snapshot(url_root, profile_lookup)
    while current[sent_metric] - before[sent_metric] < expected_sent_hashes:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Lookup metrics did not publish the completed batch: "
                f"expected_sent={expected_sent_hashes}, before={before}, current={current}"
            )
        time.sleep(METRIC_POLL_INTERVAL_SECONDS)
        current = metric_snapshot(url_root, profile_lookup)
    return stable_metric_snapshot(url_root, profile_lookup)


def metric_deltas(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {name: after[name] - before[name] for name in before if name not in QUIESCENCE_METRICS}


def output_digest(outputs: list[str]) -> str:
    encoded = json.dumps(outputs, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def complete(url_root: str, prompt: list[int], max_tokens: int = OUTPUT_TOKENS) -> str:
    response = requests.post(
        url_root + "/v1/completions",
        json={
            "model": "kvpp-test",
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
        },
        timeout=1800,
    )
    response.raise_for_status()
    result = response.json()
    assert result["choices"][0]["finish_reason"] == "length"
    return result["choices"][0]["text"]


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
            "max_tokens": OUTPUT_TOKENS,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        timeout=1800,
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
        "batch_wall_seconds": wall_time,
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


def expected_hash_metrics(mode: LookupHashMode, spec: ScenarioSpec) -> tuple[int, int]:
    if mode is LookupHashMode.FULL:
        return REQUESTS_PER_SAMPLE * TOTAL_BLOCKS, 0
    return REQUESTS_PER_SAMPLE * spec.remaining_blocks, REQUESTS_PER_SAMPLE * spec.hbm_blocks


def validate_batch_metrics(
    deltas: dict[str, float],
    mode: LookupHashMode,
    spec: ScenarioSpec,
    profile_lookup: bool,
) -> list[str]:
    errors = []
    expected_sent, expected_omitted = expected_hash_metrics(mode, spec)
    expected_hbm_tokens = REQUESTS_PER_SAMPLE * spec.hbm_blocks * BLOCK_SIZE
    expected_external_tokens = sum(spec.external_blocks) * BLOCK_SIZE
    expected = {
        "vllm:ascend_store_lookup_hashes_sent_total": expected_sent,
        "vllm:ascend_store_lookup_hashes_omitted_total": expected_omitted,
        "vllm:prefix_cache_hits_total": expected_hbm_tokens,
        "vllm:external_prefix_cache_hits_total": expected_external_tokens,
    }
    for name, value in expected.items():
        if deltas[name] != value:
            errors.append(f"{name}={deltas[name]}, expected={value}")
    if deltas["vllm:ascend_store_load_get_keys_total"] <= 0:
        errors.append("vllm:ascend_store_load_get_keys_total must be positive")
    if profile_lookup:
        lookup_requests = deltas["vllm:ascend_store_lookup_requests_total"]
        lookup_duration = deltas["vllm:ascend_store_lookup_duration_seconds_total"]
        if lookup_requests != REQUESTS_PER_SAMPLE:
            errors.append(f"profiled_lookup_requests={lookup_requests}, expected={REQUESTS_PER_SAMPLE}")
        if lookup_duration <= 0:
            errors.append(f"profiled_lookup_duration={lookup_duration}, expected>0")
    return errors


def run_serving_batch(
    url_root: str,
    workload: Workload,
    mode: LookupHashMode,
    profile_lookup: bool,
) -> dict[str, Any]:
    before = stable_metric_snapshot(url_root, profile_lookup)
    timings, wall_time = asyncio.run(run_concurrent_requests(url_root, workload.prompts))
    expected_sent, _ = expected_hash_metrics(mode, workload.spec)
    after = wait_for_batch_metrics(url_root, before, expected_sent, profile_lookup)
    outputs = [timing.text for timing in timings]
    deltas = metric_deltas(after, before)
    errors = validate_batch_metrics(deltas, mode, workload.spec, profile_lookup)
    summary: dict[str, Any] = serving_summary(timings, wall_time)
    summary.update(deltas)
    if profile_lookup:
        duration = deltas["vllm:ascend_store_lookup_duration_seconds_total"]
        requests_count = deltas["vllm:ascend_store_lookup_requests_total"]
        summary["lookup_mean_ms"] = duration / requests_count * 1e3
        summary["lookup_share_of_batch_wall_percent"] = duration / wall_time * 100
        if summary["lookup_share_of_batch_wall_percent"] > 100.5:
            errors.append(
                "lookup blocking time exceeds single-scheduler batch wall time: "
                f"share={summary['lookup_share_of_batch_wall_percent']}"
            )
    summary.update(
        {
            "scenario_id": workload.spec.scenario_id,
            "sample_index": workload.spec.sample_index,
            "manifest_digest": workload.manifest_digest,
            "hbm_blocks": workload.spec.hbm_blocks,
            "hbm_hit_percent": workload.spec.hbm_blocks / TOTAL_BLOCKS * 100,
            "external_blocks": workload.spec.external_blocks,
            "external_hit_percent_of_suffix": (
                sum(workload.spec.external_blocks) / (REQUESTS_PER_SAMPLE * workload.spec.remaining_blocks) * 100
            ),
            "output_digest": output_digest(outputs),
            "validation_errors": errors,
            "valid": not errors,
        }
    )
    return summary


def reset_prefix_cache(url_root: str, profile_lookup: bool) -> None:
    response = requests.post(url_root + "/reset_prefix_cache", timeout=30)
    response.raise_for_status()
    stable_metric_snapshot(url_root, profile_lookup)


def prepare_workload(url_root: str, workload: Workload, profile_lookup: bool) -> None:
    """Create exact external prefixes, then leave only the common prefix in HBM."""
    reset_prefix_cache(url_root, profile_lookup)
    for preload_prompt in workload.preload_prompts:
        complete(url_root, preload_prompt, max_tokens=1)
    stable_metric_snapshot(url_root, profile_lookup)
    reset_prefix_cache(url_root, profile_lookup)
    complete(url_root, workload.common_prefix, max_tokens=1)
    stable_metric_snapshot(url_root, profile_lookup)


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
                "ock.mmc.local_service.world_size": TENSOR_PARALLEL_SIZE,
                "ock.mmc.local_service.protocol": "device_sdma",
                "ock.mmc.local_service.dram.size": "16GB",
            },
        },
    )


def server_args(mode: LookupHashMode, profile_lookup: bool, port: int) -> list[str]:
    return [
        "--served-model-name",
        "kvpp-test",
        "--trust-remote-code",
        "--quantization",
        "ascend",
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--data-parallel-size",
        "1",
        "--enforce-eager",
        "--max-model-len",
        str(MODEL_MAX_LEN),
        "--max-num-batched-tokens",
        "16384",
        "--max-num-seqs",
        str(REQUESTS_PER_SAMPLE),
        "--block-size",
        str(BLOCK_SIZE),
        "--gpu-memory-utilization",
        "0.9",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--mm-processor-cache-gb",
        "0",
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
                    "profile_lookup": profile_lookup,
                },
            }
        ),
    ]


def run_serving_mode(
    mode: LookupHashMode,
    tmp_path,
    round_index: int,
    order_position: int,
    *,
    profile_lookup: bool,
) -> dict[str, Any]:
    from tests.e2e.conftest import RemoteOpenAIServer

    config = memcache_config()
    isolation_id = uuid.uuid4().hex
    case_name = f"{tmp_path.name}-r{round_index}-p{order_position}-{mode.value}-{isolation_id}"
    with SingleNodeMemcacheManager(config, case_name) as pool:
        port = get_open_port()
        with RemoteOpenAIServer(
            maybe_model_redirect(MODEL),
            server_args(mode, profile_lookup, port),
            server_port=port,
            auto_port=False,
            env_dict={
                **pool.server_envs,
                "VLLM_USE_MODELSCOPE": "true",
                "VLLM_USE_V2_MODEL_RUNNER": "0",
                "VLLM_SERVER_DEV_MODE": "1",
            },
        ) as server:
            tokenized = requests.post(
                server.url_for("tokenize"),
                json={"model": "kvpp-test", "prompt": PROMPT_SEED},
                timeout=60,
            )
            tokenized.raise_for_status()
            seed_tokens = tokenized.json()["tokens"]

            # Warm runtime and the real external-load path with a unique prompt;
            # none of its keys can satisfy a measured scenario.
            warmup = build_workload(seed_tokens, warmup_spec())
            prepare_workload(server.url_root, warmup, profile_lookup)
            warmup_result = run_serving_batch(server.url_root, warmup, mode, profile_lookup)
            run_report: dict[str, Any] = {
                "round": round_index,
                "order_position": order_position,
                "mode": mode.value,
                "profile_lookup": profile_lookup,
                "isolation_id": isolation_id,
                "warmup": warmup_result,
                "samples": [],
            }
            if not warmup_result["valid"]:
                run_report["valid"] = False
                return run_report

            for spec in scenario_specs():
                workload = build_workload(seed_tokens, spec)
                prepare_workload(server.url_root, workload, profile_lookup)
                run_report["samples"].append(run_serving_batch(server.url_root, workload, mode, profile_lookup))

            run_report["valid"] = all(sample["valid"] for sample in run_report["samples"])
            return run_report


def aggregate_mode_runs(runs: list[dict[str, Any]], mode: LookupHashMode) -> dict[str, Any]:
    mode_runs = [run for run in runs if run["mode"] == mode.value]
    samples = [sample for run in mode_runs for sample in run["samples"]]
    valid_samples = [sample for sample in samples if sample["valid"]]
    fields = (*SERVING_PERFORMANCE_FIELDS, *CORE_LOOKUP_METRICS)
    return {
        "instances": len(mode_runs),
        "valid_instances": sum(run["valid"] for run in mode_runs),
        "samples": len(samples),
        "valid_samples": len(valid_samples),
        "requests_in_valid_samples": len(valid_samples) * REQUESTS_PER_SAMPLE,
        "valid_sample_distributions": {
            field: value_distribution([sample[field] for sample in valid_samples]) for field in fields if valid_samples
        },
    }


def paired_serving_comparison(runs: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = []
    validation_errors = []
    comparison_fields = (
        "request_throughput_gain_percent",
        "output_throughput_gain_percent",
        "batch_wall_reduction_percent",
        "mean_ttft_reduction_percent",
        "p99_ttft_reduction_percent",
        "mean_e2el_reduction_percent",
        "p99_e2el_reduction_percent",
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
        for full, suffix in zip(full_run["samples"], suffix_run["samples"], strict=True):
            errors = []
            for field in ("scenario_id", "manifest_digest", "hbm_blocks", "external_blocks"):
                if full[field] != suffix[field]:
                    errors.append(f"{field}: full={full[field]}, suffix={suffix[field]}")
            if not full["valid"] or not suffix["valid"]:
                errors.append("one or both samples are invalid")
            if full["output_digest"] != suffix["output_digest"]:
                errors.append(f"output digests differ: full={full['output_digest']}, suffix={suffix['output_digest']}")
            for name in WORKLOAD_METRICS:
                if full[name] != suffix[name]:
                    errors.append(f"{name}: full={full[name]}, suffix={suffix[name]}")
            if errors:
                validation_errors.append(f"round={round_index}, scenario={full['scenario_id']}: " + "; ".join(errors))
                continue
            comparisons.append(
                {
                    "round": round_index,
                    "scenario_id": full["scenario_id"],
                    "hbm_hit_percent": full["hbm_hit_percent"],
                    "external_hit_percent_of_suffix": full["external_hit_percent_of_suffix"],
                    "request_throughput_gain_percent": (
                        suffix["request_throughput_per_second"] / full["request_throughput_per_second"] - 1
                    )
                    * 100,
                    "output_throughput_gain_percent": (
                        suffix["output_throughput_tokens_per_second"] / full["output_throughput_tokens_per_second"] - 1
                    )
                    * 100,
                    "batch_wall_reduction_percent": (1 - suffix["batch_wall_seconds"] / full["batch_wall_seconds"])
                    * 100,
                    "mean_ttft_reduction_percent": (1 - suffix["mean_ttft_ms"] / full["mean_ttft_ms"]) * 100,
                    "p99_ttft_reduction_percent": (1 - suffix["p99_ttft_ms"] / full["p99_ttft_ms"]) * 100,
                    "mean_e2el_reduction_percent": (1 - suffix["mean_e2el_ms"] / full["mean_e2el_ms"]) * 100,
                    "p99_e2el_reduction_percent": (1 - suffix["p99_e2el_ms"] / full["p99_e2el_ms"]) * 100,
                }
            )

    return {
        "pairs": len(comparisons),
        "expected_pairs": len(SERVING_MODE_ORDERS) * len(HBM_BLOCK_SCHEDULE),
        "validation_errors": validation_errors,
        "distributions": {
            field: {
                **value_distribution([comparison[field] for comparison in comparisons]),
                "suffix_win_count": sum(comparison[field] > 0 for comparison in comparisons),
            }
            for field in comparison_fields
            if comparisons
        },
        "raw_pairs": comparisons,
    }


def run_benchmark(tmp_path, *, profile_lookup: bool) -> dict[str, Any]:
    runs = []
    for round_index, mode_order in enumerate(SERVING_MODE_ORDERS):
        for order_position, mode in enumerate(mode_order):
            runs.append(
                run_serving_mode(
                    mode,
                    tmp_path,
                    round_index,
                    order_position,
                    profile_lookup=profile_lookup,
                )
            )
    paired = paired_serving_comparison(runs)
    report = {
        "config": {
            "model": MODEL,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "model_runner_v2": False,
            "expert_parallel": False,
            "async_scheduling": False,
            "load_async": False,
            "profile_lookup": profile_lookup,
            "mode_orders": [[mode.value for mode in order] for order in SERVING_MODE_ORDERS],
            "instances_per_mode": len(SERVING_MODE_ORDERS),
            "samples_per_instance": len(HBM_BLOCK_SCHEDULE),
            "requests_per_sample": REQUESTS_PER_SAMPLE,
            "measured_requests_per_mode": (len(SERVING_MODE_ORDERS) * len(HBM_BLOCK_SCHEDULE) * REQUESTS_PER_SAMPLE),
            "max_model_len": MODEL_MAX_LEN,
            "input_tokens_per_request": INPUT_TOKENS,
            "total_blocks": TOTAL_BLOCKS,
            "hbm_block_schedule": HBM_BLOCK_SCHEDULE,
            "external_hit_strata": EXTERNAL_HIT_STRATA,
            "manifest_seed": MANIFEST_SEED,
            "output_tokens": OUTPUT_TOKENS,
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
    return report


@wait_until_npu_memory_free()
def test_vllm_serve_lookup_payload_e2e(tmp_path):
    """Measure serving without profiling contamination on a moderate hit mix."""
    pytest.importorskip("memcache_hybrid")
    report = run_benchmark(tmp_path, profile_lookup=False)
    print("\nKVPP_LOOKUP_PAYLOAD_E2E=" + json.dumps(report, sort_keys=True))
    assert report["valid"], json.dumps(
        {
            "invalid_runs": [
                {"round": run["round"], "mode": run["mode"]} for run in report["runs"] if not run["valid"]
            ],
            "pair_validation_errors": report["paired"]["validation_errors"],
        },
        sort_keys=True,
    )
