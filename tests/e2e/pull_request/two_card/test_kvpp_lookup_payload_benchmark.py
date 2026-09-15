# SPDX-License-Identifier: Apache-2.0
"""Temporary A/B benchmarks for the AscendStore lookup payload RFC."""

import asyncio
import json
import statistics
import time
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
from tests.e2e.common.kvpp import MODEL, PROMPTS, server_args
from tests.e2e.nightly.single_node.models.scripts.kv_pool_runtime import SingleNodeMemcacheManager
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LookupHashMode

pytestmark = pytest.mark.e2e_model(MODEL)

RPC_FULL_HASHES = 4096
RPC_SUFFIX_HASHES = 32
RPC_WARMUPS = 20
RPC_SAMPLES = 200

BLOCK_SIZE = 128
COMMON_PREFIX_BLOCKS = 256
SERVING_REQUESTS = 32
SERVING_OUTPUT_TOKENS = 8
GPU_BLOCKS = 384


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(samples: list[float]) -> dict[str, float]:
    return {
        "mean_us": statistics.fmean(samples) * 1e6,
        "p50_us": percentile(samples, 50) * 1e6,
        "p90_us": percentile(samples, 90) * 1e6,
        "p99_us": percentile(samples, 99) * 1e6,
    }


class LookupStub:
    def __init__(self, expected_calls: int):
        self.expected_calls = expected_calls
        self.calls = 0
        self.hash_counts = {LookupHashMode.FULL: [], LookupHashMode.SUFFIX: []}
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
    samples = {LookupHashMode.FULL: [], LookupHashMode.SUFFIX: []}

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
            order = (
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


def replace_arg(args: list[str], name: str, value: int) -> None:
    args[args.index(name) + 1] = str(value)


def metric_total(metrics_text: str, name: str) -> float:
    return sum(
        float(line.split()[-1]) for line in metrics_text.splitlines() if line.startswith((f"{name}{{", f"{name} "))
    )


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
                "ock.mmc.local_service.world_size": 2,
                "ock.mmc.local_service.protocol": "device_sdma",
                "ock.mmc.local_service.dram.size": "2GB",
            },
        },
    )


def run_serving_mode(mode: LookupHashMode, tmp_path) -> tuple[dict[str, float], list[str]]:
    from tests.e2e.conftest import RemoteOpenAIServer

    config = memcache_config()
    with SingleNodeMemcacheManager(config, f"{tmp_path.name}-{mode.value}") as pool:
        port = get_open_port()
        args = server_args()
        replace_arg(args, "--max-model-len", 34_000)
        replace_arg(args, "--max-num-batched-tokens", 4096)
        replace_arg(args, "--max-num-seqs", SERVING_REQUESTS)
        replace_arg(args, "--num-gpu-blocks-override", GPU_BLOCKS)
        args += [
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
                        "load_async": True,
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
                json={"model": "kvpp-test", "prompt": PROMPTS[1]},
                timeout=30,
            )
            tokenized.raise_for_status()
            seed_tokens = tokenized.json()["tokens"]
            common_size = COMMON_PREFIX_BLOCKS * BLOCK_SIZE
            common_prefix = (seed_tokens * ((common_size + len(seed_tokens) - 1) // len(seed_tokens)))[:common_size]
            suffix_seed = (seed_tokens * ((BLOCK_SIZE + len(seed_tokens) - 1) // len(seed_tokens)))[:BLOCK_SIZE]
            prompts = [common_prefix + suffix_seed[index:] + suffix_seed[:index] for index in range(SERVING_REQUESTS)]

            # Persist every suffix externally, then leave only the shared
            # prefix in HBM. The measured requests therefore perform the same
            # one-block external lookup/load while their RPC payload differs.
            complete(server.url_root, common_prefix)
            expected = [complete(server.url_root, prompt) for prompt in prompts]
            requests.post(server.url_for("reset_prefix_cache"), timeout=30).raise_for_status()
            complete(server.url_root, common_prefix)

            before = requests.get(server.url_for("metrics"), timeout=30)
            before.raise_for_status()
            timings, wall_time = asyncio.run(run_concurrent_requests(server.url_root, prompts))
            after = requests.get(server.url_for("metrics"), timeout=30)
            after.raise_for_status()

            measured = [timing.text for timing in timings]
            assert measured == expected
            summary = serving_summary(timings, wall_time)
            for name in (
                "vllm:prefix_cache_hits_total",
                "vllm:external_prefix_cache_hits_total",
                "vllm:ascend_store_load_get_keys_total",
                "vllm:ascend_store_lookup_hashes_sent_total",
                "vllm:ascend_store_lookup_hashes_omitted_total",
            ):
                summary[name] = metric_total(after.text, name) - metric_total(before.text, name)
            return summary, measured


def test_vllm_serve_lookup_payload_benchmark(tmp_path):
    """Compare complete serving metrics with one remote block after a long HBM hit."""
    pytest.importorskip("memcache_hybrid")
    results = {}
    outputs = {}
    for mode in (LookupHashMode.FULL, LookupHashMode.SUFFIX):
        results[mode.value], outputs[mode.value] = run_serving_mode(mode, tmp_path)

    print("\nKVPP_LOOKUP_PAYLOAD_SERVING_BENCHMARK=" + json.dumps(results, sort_keys=True))
    assert outputs["full"] == outputs["suffix"]
    assert results["full"]["vllm:prefix_cache_hits_total"] > 0
    assert results["suffix"]["vllm:prefix_cache_hits_total"] > 0
    assert results["full"]["vllm:external_prefix_cache_hits_total"] > 0
    assert results["suffix"]["vllm:external_prefix_cache_hits_total"] > 0
    assert results["full"]["vllm:ascend_store_load_get_keys_total"] > 0
    assert results["suffix"]["vllm:ascend_store_load_get_keys_total"] > 0
    assert results["full"]["vllm:ascend_store_lookup_hashes_omitted_total"] == 0
    assert results["suffix"]["vllm:ascend_store_lookup_hashes_omitted_total"] > 0
    assert (
        results["suffix"]["vllm:ascend_store_lookup_hashes_sent_total"]
        < results["full"]["vllm:ascend_store_lookup_hashes_sent_total"] * 0.05
    )
