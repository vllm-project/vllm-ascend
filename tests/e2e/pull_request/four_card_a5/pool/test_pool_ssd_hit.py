# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Pool A/B watch for DeepSeek-V4.1-Flash DP8 x TP1 memcache GVA + optional SSD.

DeepSeek-V4.1-Flash uses CED hybrid attention, which does not support the
layerwise pool path, so ``use_layerwise`` stays false (see the model tutorial).

The SSD tier is behind ``KV_POOL_SSD_ENABLE`` (default off): the standalone
memcache runtime it needs is not in the #22849 base yet. With the switch off
the case still runs the DRAM hit sequence and the A/B perf gate.
"""

import os
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import pytest
import requests
from transformers import AutoTokenizer
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMemcacheManager
from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.kv_pool_watch import (
    POOL_SECRET,
    build_pool_accuracy_prompt,
    build_pool_prompts,
    run_pool_ab,
    run_pool_hit_sequence,
)

MODEL = os.environ.get(
    "DEEPSEEK_V4_1_FLASH_MODEL_PATH",
    "deepseek-ai/DeepSeek-V4.1-Flash-1017",
)
SSD_DEV = os.environ.get("KV_POOL_SSD_DEV", "/dev/loop0")
SSD_ENABLED = os.environ.get("KV_POOL_SSD_ENABLE", "0").lower() in ("1", "true", "yes", "on")
METRICS_PORT = get_open_port()

DATA_PARALLEL_SIZE = 8
TENSOR_PARALLEL_SIZE = 1
POOL_PROMPT_TOKENS = 16388
POOL_PERF_PROMPT_TOKENS = 2 * 16384 + 2 * 16384 // 9
POOL_MAX_MODEL_LEN = max(POOL_PROMPT_TOKENS, POOL_PERF_PROMPT_TOKENS) + 128
# Keep 4,096 chunked prefill on a5-8: each DP8 rank owns a whole die.
POOL_BATCHED_TOKENS = 4096
SSD_PREFIX_COUNT = 8
SSD_MAIN_REQUESTS = 32
CASE_NAME = "dsv4_1_memcache_ssd"

POOL_KV_TRANSFER_CONFIG = {
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "backend": "memcache",
        "lookup_rpc_port": "1",
        "use_layerwise": False,
    },
}


def _build_memcache_pool_config() -> MemcacheKVPoolConfig:
    if not SSD_ENABLED:
        # Default path: co-located DRAM, valid on the #22849 base.
        return MemcacheKVPoolConfig(
            meta_service_port=get_open_port(),
            config_store_port=get_open_port(),
            config={
                "meta": {"ock.mmc.log_level": "error"},
                "local": {
                    "ock.mmc.log_level": "error",
                    "ock.mmc.local_service.world_size": 256,
                    "ock.mmc.local_service.protocol": "device_urma",
                    # Per engine; 16K/36K prompts need well under 1GB of KV, and
                    # 8 DP engines each reserve this much host memory.
                    "ock.mmc.local_service.dram.size": "4GB",
                },
            },
        )
    # SSD path: requires the Nightly-A3 standalone runtime (mode/metrics_port).
    # Build the keywords dynamically so this file also type-checks on the
    # #22849 base, where MemcacheKVPoolConfig does not expose them yet.
    standalone_kwargs: dict[str, Any] = {
        "meta_service_port": get_open_port(),
        "config_store_port": get_open_port(),
        "metrics_port": METRICS_PORT,
        "mode": "standalone",
        "config": {
            "meta": {"ock.mmc.log_level": "error"},
            "local": {
                "ock.mmc.log_level": "error",
                "ock.mmc.local_service.world_size": 256,
                "ock.mmc.local_service.protocol": "device_urma",
                "ock.mmc.local_service.dram.size": "0GB",
            },
            "standalone": {
                "ock.mmc.log_level": "error",
                "ock.mmc.local_service.world_size": 256,
                "ock.mmc.local_service.protocol": "device_urma",
                "ock.mmc.local_service.dram.size": "10GB",
                "ock.mmc.local_service.storage.enabled": "true",
                "ubsio.disk.path": SSD_DEV,
                "ubsio.mem.size_in_gb": 10,
                "ubsio.standalone.device_count": 1,
            },
        },
    }
    try:
        return MemcacheKVPoolConfig(**standalone_kwargs)
    except TypeError as exc:
        raise AssertionError(
            "KV_POOL_SSD_ENABLE=1 requires the Nightly-A3 standalone memcache runtime (mode/metrics_port)"
        ) from exc


@contextmanager
def _leg(use_pool: bool, case_name: str):
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            data_parallel_size=DATA_PARALLEL_SIZE,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            enable_expert_parallel=True,
            max_model_len=POOL_MAX_MODEL_LEN,
            max_num_batched_tokens=POOL_BATCHED_TOKENS,
            max_num_seqs=64,
            tokenizer_mode="deepseek_v41",
            block_size=128,
            gpu_memory_utilization=0.9,
            quantization="deepseek_v4_fp8",
            engram_config={"cpu_offload": True, "dp_shared_memory": True},
            additional_config={"multistream_overlap_shared_expert": True, "combine_quant_mode": 4},
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
            enable_prefix_caching=not use_pool,
            disable_log_stats=False,
        )
        if use_pool:
            pool = stack.enter_context(SingleNodeMemcacheManager(_build_memcache_pool_config(), case_name))
            stack.enter_context(patch.dict(os.environ, pool.server_envs))
            kwargs["kv_transfer_config"] = POOL_KV_TRANSFER_CONFIG
        print(f"[kv_pool_leg] {case_name} pool={use_pool}", flush=True)
        yield stack.enter_context(DPVllmRunner(MODEL, **kwargs))


def _read_pool_metrics() -> dict[str, float]:
    text = requests.get(f"http://127.0.0.1:{METRICS_PORT}/metrics", timeout=30).text
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        values[name.strip()] = float(value)
    return values


def _ssd_prompts(count: int, tokens: int) -> list[list[int]]:
    base, _ = build_pool_prompts(tokens)
    return [[1000 + 7 * index] + base[1:] for index in range(count)]


def _metric(metrics: dict[str, float], *names: str) -> float:
    for name in names:
        if name in metrics:
            return metrics[name]
    return 0.0


def _assert_ssd_evidence(before: dict[str, float], after: dict[str, float]) -> None:
    def delta(*names: str) -> float:
        return _metric(after, *names) - _metric(before, *names)

    assert after.get("memcache_ubs_io_disk_capacity_bytes", 0.0) > 0
    assert after.get("memcache_ubs_io_disk_status", 0.0) == 0
    assert after.get("memcache_ubs_io_fault_disk_num", 0.0) == 0
    # Working set must exceed DRAM: eviction + rewarm is the SSD proof.
    assert delta("memcache_evict_operations_total", "memcache_evict_operations") > 0, (
        "SSD: no eviction; the working set may be smaller than DRAM"
    )
    assert delta("memcache_rewarm_total", "memcache_rewarm") > 0 or delta("memcache_rewarm_bytes_total") > 0, (
        "SSD: no rewarm"
    )
    assert delta("memcache_ubs_io_disk_used_bytes") > 0, "SSD: disk usage did not grow"
    assert delta("memcache_ubs_io_disk_write_bandwidth") > 0 or delta("memcache_ubs_io_disk_read_bandwidth") > 0, (
        "SSD: no disk bandwidth"
    )
    print(
        f"[kv_pool_ssd] evict+{delta('memcache_evict_operations_total', 'memcache_evict_operations'):.0f} "
        f"rewarm+{delta('memcache_rewarm_total', 'memcache_rewarm'):.0f} "
        f"disk_used+{delta('memcache_ubs_io_disk_used_bytes'):.0f}",
        flush=True,
    )


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="prefix_caching",
    parallel="DP,TP,EP",
    deploy="pd_mix",
    hardware="A5",
    quantization="FP8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTHONHASHSEED": "0",
        # A5 (950PR/950DT) needs UBOE/UB comm resources for the pool runtime.
        "ASCEND_GLOBAL_RESOURCE_CONFIG": '{"comm_resource_config.protocol_desc":["uboe:device"]}',
        "ASCEND_LOCAL_COMM_RES": '{"version":"1.3"}',
        # DP8 settings mirrored from PR #22258's known-good A5 config.
        "HCCL_OP_EXPANSION_MODE": "CCU_SCHED",
        "HCCL_BUFFSIZE": "1600",
        "HCCL_CONNECT_TIMEOUT": "1200",
        "HCCL_EXEC_TIMEOUT": "2040",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "3000",
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "10",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v4_1_flash_memcache_layerwise_ssd_hit() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt_ids = build_pool_accuracy_prompt(tokenizer, POOL_PROMPT_TOKENS, POOL_MAX_MODEL_LEN)

    def _accuracy(runner) -> None:
        outputs = runner.generate_greedy(["Hello, my name is"], 16)
        assert outputs and outputs[0][0] and outputs[0][1].strip()

    def _hit(runner) -> dict:
        summary = run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=POOL_PROMPT_TOKENS,
            prompt_ids=prompt_ids,
            replicas=DATA_PARALLEL_SIZE,
            check_output=True,
            answer_secret=POOL_SECRET,
            tag="P7",
        )
        if not SSD_ENABLED:
            print("[kv_pool_ssd] disabled; set KV_POOL_SSD_ENABLE=1 to exercise the SSD tier", flush=True)
            return summary

        pool_before = _read_pool_metrics()
        prompts = _ssd_prompts(SSD_PREFIX_COUNT, POOL_PROMPT_TOKENS)
        runner.generate_greedy(prompts, 1)  # warmup: one request per prefix
        runner.generate_greedy(prompts * (SSD_MAIN_REQUESTS // SSD_PREFIX_COUNT), 1)
        pool_after = _read_pool_metrics()
        _assert_ssd_evidence(pool_before, pool_after)
        return summary

    run_pool_ab(
        case_name=CASE_NAME,
        build_leg=_leg,
        accuracy=_accuracy,
        hit=_hit,
        prompt_tokens=POOL_PROMPT_TOKENS,
        prompt_ids=prompt_ids,
        perf_prompt_tokens=POOL_PERF_PROMPT_TOKENS,
        token_identity=True,
        replicas=DATA_PARALLEL_SIZE,
    )
