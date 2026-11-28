# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Pool A/B watch for Qwen3.8-27B TP2 hybrid Mamba on two A5 dies."""

import os
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMemcacheManager
from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.kv_pool_watch import (
    POOL_SECRET,
    build_pool_accuracy_prompt,
    ensure_memcache_transfer,
    run_pool_ab,
    run_pool_hit_sequence,
)

MODEL = os.environ.get(
    "QWEN38_27B_MODEL_PATH",
    "Eco-Tech/Qwen3.8-27B-w8a8",
)

TENSOR_PARALLEL_SIZE = 2
# Measured on A3: lcm=896, 16,388 tokens -> 16,128 hits (18 lcm blocks).
POOL_LCM_BLOCK_SIZE = 896
POOL_PROMPT_TOKENS = 16388
POOL_EXPECTED_HIT_TOKENS = 16128
# Perf prompt: 18 lcm blocks plus a tail -> 16,128 / 17,920 = 90% hit.
POOL_PERF_PROMPT_TOKENS = 18 * POOL_LCM_BLOCK_SIZE + 18 * POOL_LCM_BLOCK_SIZE // 9
POOL_MAX_MODEL_LEN = max(POOL_PROMPT_TOKENS, POOL_PERF_PROMPT_TOKENS) + 128
POOL_BATCHED_TOKENS = 4096
CASE_NAME = "qwen38_27b_tp2_mamba_memcache"

POOL_KV_TRANSFER_CONFIG = {
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "backend": "memcache",
        "lookup_rpc_port": "1",
        "use_layerwise": False,
        # Hybrid non-layerwise MUST set this false, otherwise the Mamba state
        # copy-after-load is skipped and the hit output is silently wrong.
        "load_async": False,
    },
}


def _build_memcache_pool_config() -> MemcacheKVPoolConfig:
    return MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config={
            "meta": {"ock.mmc.log_level": "error"},
            "local": {
                "ock.mmc.log_level": "error",
                "ock.mmc.local_service.world_size": 256,
                "ock.mmc.local_service.protocol": "device_urma",
                "ock.mmc.local_service.dram.size": "10GB",
            },
        },
    )


@contextmanager
def _leg(use_pool: bool, case_name: str):
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            max_model_len=POOL_MAX_MODEL_LEN,
            max_num_batched_tokens=POOL_BATCHED_TOKENS,
            block_size=128,
            distributed_executor_backend="mp",
            gpu_memory_utilization=0.9,
            quantization="ascend",
            # align depends on the local prefix cache; both legs keep it on.
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            disable_log_stats=False,
        )
        if use_pool:
            pool = stack.enter_context(SingleNodeMemcacheManager(_build_memcache_pool_config(), case_name))
            stack.enter_context(patch.dict(os.environ, pool.server_envs))
            kwargs["kv_transfer_config"] = POOL_KV_TRANSFER_CONFIG
        print(f"[kv_pool_leg] {case_name} pool={use_pool}", flush=True)
        yield stack.enter_context(VllmRunner(MODEL, **kwargs))


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="prefix_caching",
    parallel="TP",
    deploy="pd_mix",
    hardware="A5",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        # HYBM full mesh over the eight on-board dies instead of the
        # PEER2NET/CLOS plane. The A5 CI hosts expose no usable UB CLOS
        # (empty /etc/hccn.conf, no /etc/hixlep + /lib/route.conf), so a plain
        # device_urma local service stays pending in WaitForChannelsReady.
        "TEST_MF_TRANSPORT_FULL_MESH": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTHONHASHSEED": "0",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen38_27b_tp2_mamba_pool_watch() -> None:
    ensure_memcache_transfer("device_urma", TENSOR_PARALLEL_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt_ids = build_pool_accuracy_prompt(tokenizer, POOL_PROMPT_TOKENS, POOL_MAX_MODEL_LEN)

    def _accuracy(runner) -> None:
        outputs = runner.generate_greedy(["Hello, my name is"], 16)
        assert outputs and outputs[0][0] and outputs[0][1].strip()

    def _hit(runner) -> dict:
        return run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=POOL_PROMPT_TOKENS,
            prompt_ids=prompt_ids,
            lcm_block_size=POOL_LCM_BLOCK_SIZE,
            expected_hit_tokens=POOL_EXPECTED_HIT_TOKENS,
            reset_local_cache=True,
            check_output=True,
            answer_secret=POOL_SECRET,
            tag="P9",
        )

    run_pool_ab(
        case_name=CASE_NAME,
        build_leg=_leg,
        accuracy=_accuracy,
        hit=_hit,
        prompt_tokens=POOL_PROMPT_TOKENS,
        prompt_ids=prompt_ids,
        perf_prompt_tokens=POOL_PERF_PROMPT_TOKENS,
        perf_reset_local=True,
    )
