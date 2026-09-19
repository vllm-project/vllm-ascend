# SPDX-License-Identifier: Apache-2.0

import json
import time

import pytest
import requests
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free
from tests.e2e.nightly.single_node.models.scripts.kv_pool_runtime import SingleNodeMemcacheManager

MODEL = "Qwen/Qwen3-8B"
SERVED_MODEL = "ascend-store-lookup-test"
TENSOR_PARALLEL_SIZE = 2
BLOCK_SIZE = 128
PROMPT_BLOCKS = 12
HBM_BLOCKS = 6
PROMPT_TOKENS = PROMPT_BLOCKS * BLOCK_SIZE
HBM_TOKENS = HBM_BLOCKS * BLOCK_SIZE
OUTPUT_TOKENS = 16
MODEL_MAX_LEN = 4096
METRIC_SETTLE_TIMEOUT_SECONDS = 30
# vLLM recomputes the final prompt token before sampling even when the
# external cache contains the complete prompt.
FINAL_PROMPT_TOKENS_TO_RECOMPUTE = 1
PROMPT_SEED = " ".join(f"AscendStore lookup correctness marker {index}" for index in range(256))

PREFIX_HIT_METRIC = "vllm:prefix_cache_hits_total"
EXTERNAL_HIT_METRIC = "vllm:external_prefix_cache_hits_total"
LOAD_KEYS_METRIC = "vllm:ascend_store_load_get_keys_total"
DELAYED_REQUESTS_METRIC = "vllm:ascend_store_delayed_release_requests"
DELAYED_BLOCKS_METRIC = "vllm:ascend_store_delayed_release_blocks"
OBSERVED_METRICS = (
    PREFIX_HIT_METRIC,
    EXTERNAL_HIT_METRIC,
    LOAD_KEYS_METRIC,
    DELAYED_REQUESTS_METRIC,
    DELAYED_BLOCKS_METRIC,
)

assert PROMPT_TOKENS + OUTPUT_TOKENS <= MODEL_MAX_LEN
pytestmark = pytest.mark.e2e_model(MODEL)


def metric_total(metrics_text: str, name: str) -> float:
    return sum(
        float(line.split()[-1]) for line in metrics_text.splitlines() if line.startswith((f"{name}{{", f"{name} "))
    )


def metric_snapshot(url_root: str) -> dict[str, float]:
    response = requests.get(url_root + "/metrics", timeout=30)
    response.raise_for_status()
    return {name: metric_total(response.text, name) for name in OBSERVED_METRICS}


def stable_metric_snapshot(url_root: str) -> dict[str, float]:
    deadline = time.monotonic() + METRIC_SETTLE_TIMEOUT_SECONDS
    previous = metric_snapshot(url_root)
    stable_reads = 0
    while time.monotonic() < deadline:
        time.sleep(0.2)
        current = metric_snapshot(url_root)
        saves_quiescent = current[DELAYED_REQUESTS_METRIC] == current[DELAYED_BLOCKS_METRIC] == 0
        if current == previous and saves_quiescent:
            stable_reads += 1
            if stable_reads == 2:
                return current
        else:
            stable_reads = 0
            previous = current
    raise TimeoutError(f"AscendStore metrics did not settle: {previous}")


def complete(url_root: str, prompt: list[int], max_tokens: int) -> str:
    response = requests.post(
        url_root + "/v1/completions",
        json={"model": SERVED_MODEL, "prompt": prompt, "temperature": 0, "max_tokens": max_tokens, "ignore_eos": True},
        timeout=600,
    )
    response.raise_for_status()
    result = response.json()
    assert result["choices"][0]["finish_reason"] == "length"
    assert result["usage"]["completion_tokens"] == max_tokens
    return result["choices"][0]["text"]


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
                "ock.mmc.local_service.dram.size": "4GB",
            },
        },
    )


def server_args(port: int) -> list[str]:
    return [
        "--served-model-name",
        SERVED_MODEL,
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--enforce-eager",
        "--max-model-len",
        str(MODEL_MAX_LEN),
        "--max-num-batched-tokens",
        "1024",
        "--max-num-seqs",
        "1",
        "--block-size",
        str(BLOCK_SIZE),
        "--num-gpu-blocks-override",
        "64",
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
                    "lookup_hash_mode": "suffix",
                },
            }
        ),
    ]


@pytest.mark.e2e_coverage(
    arch="dense",
    feature="chunked_prefill,prefix_caching",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_suffix_lookup_preserves_output_with_mixed_hbm_and_external_hits(tmp_path):
    """Exercise an uncached lookup and a cropped mixed-hit lookup in one server."""
    pytest.importorskip("memcache_hybrid")
    config = memcache_config()
    with SingleNodeMemcacheManager(config, tmp_path.name) as pool:
        port = get_open_port()
        with RemoteOpenAIServer(
            maybe_model_redirect(MODEL),
            server_args(port),
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
                server.url_for("tokenize"), json={"model": SERVED_MODEL, "prompt": PROMPT_SEED}, timeout=60
            )
            tokenized.raise_for_status()
            seed_tokens = tokenized.json()["tokens"]
            assert seed_tokens
            prompt = (seed_tokens * ((PROMPT_TOKENS + len(seed_tokens) - 1) // len(seed_tokens)))[:PROMPT_TOKENS]

            expected = complete(server.url_root, prompt, OUTPUT_TOKENS)
            stable_metric_snapshot(server.url_root)
            requests.post(server.url_for("reset_prefix_cache"), timeout=30).raise_for_status()
            stable_metric_snapshot(server.url_root)

            complete(server.url_root, prompt[:HBM_TOKENS], max_tokens=1)
            before = stable_metric_snapshot(server.url_root)
            actual = complete(server.url_root, prompt, OUTPUT_TOKENS)
            after = stable_metric_snapshot(server.url_root)

    assert actual == expected
    assert after[PREFIX_HIT_METRIC] - before[PREFIX_HIT_METRIC] == HBM_TOKENS
    assert after[EXTERNAL_HIT_METRIC] - before[EXTERNAL_HIT_METRIC] == (
        PROMPT_TOKENS - HBM_TOKENS - FINAL_PROMPT_TOKENS_TO_RECOMPUTE
    )
    assert after[LOAD_KEYS_METRIC] > before[LOAD_KEYS_METRIC]
