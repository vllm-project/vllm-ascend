# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Pool A/B performance watch for DeepSeek-V4.1-Flash DP8 x TP1 memcache GVA.

DeepSeek-V4.1-Flash uses CED hybrid attention, which does not support the
layerwise pool path, so ``use_layerwise`` stays false (see the model tutorial).
"""

import os
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMemcacheManager
from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.kv_pool_watch import (
    POOL_SECRET,
    build_pool_accuracy_prompt,
    ensure_memcache_transfer,
    run_pool_ab,
    run_pool_hit_sequence,
)

MODEL = os.environ.get(
    "DEEPSEEK_V4_1_FLASH_MODEL_PATH",
    "deepseek-ai/DeepSeek-V4.1-Flash-1017",
)

DATA_PARALLEL_SIZE = 8
TENSOR_PARALLEL_SIZE = 1
POOL_LCM_BLOCK_SIZE = 16384
POOL_PROMPT_TOKENS = POOL_LCM_BLOCK_SIZE + 4
POOL_PERF_PROMPT_TOKENS = 2 * POOL_LCM_BLOCK_SIZE + 2 * POOL_LCM_BLOCK_SIZE // 9
POOL_MAX_MODEL_LEN = max(POOL_PROMPT_TOKENS, POOL_PERF_PROMPT_TOKENS) + 128
# Keep 4,096 chunked prefill on a5-8: each DP8 rank owns a whole die, so the
# 16K prefill chunk no longer splits across TP ranks.
POOL_BATCHED_TOKENS = 4096
CASE_NAME = "dsv4_1_memcache_perf"

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
    return MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config={
            "meta": {"ock.mmc.log_level": "error"},
            "local": {
                "ock.mmc.log_level": "error",
                "ock.mmc.local_service.world_size": 256,
                # A5 targets the UB path: device_urma. device_sdma is A3-only
                # and has no cross-rank data plane on A5, so it is not used as
                # a fallback here.
                "ock.mmc.local_service.protocol": "device_urma",
                # Per engine; 16K/36K prompts need well under 1GB of KV, and
                # 8 DP engines each reserve this much host memory.
                "ock.mmc.local_service.dram.size": "4GB",
            },
        },
    )


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
            # Bound pool-side hangs: workers start sequentially, and a hung
            # generate should surface as ENV_BLOCKED instead of the job timeout.
            dp_start_timeout=600,
            dp_request_timeout=600,
        )
        if use_pool:
            pool = stack.enter_context(SingleNodeMemcacheManager(_build_memcache_pool_config(), case_name))
            stack.enter_context(patch.dict(os.environ, pool.server_envs))
            kwargs["kv_transfer_config"] = POOL_KV_TRANSFER_CONFIG
        print(f"[kv_pool_leg] {case_name} pool={use_pool}", flush=True)
        yield stack.enter_context(DPVllmRunner(MODEL, **kwargs))


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
        # 950PR/950DT uses the UB super plane by default; automatic UB endpoint
        # generation only needs ASCEND_LOCAL_COMM_RES.
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
def test_deepseek_v4_1_flash_memcache_layerwise_perf() -> None:
    ensure_memcache_transfer("device_urma", DATA_PARALLEL_SIZE)
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
            replicas=DATA_PARALLEL_SIZE,
            check_output=True,
            answer_secret=POOL_SECRET,
            tag="P8",
        )

    run_pool_ab(
        case_name=CASE_NAME,
        build_leg=_leg,
        accuracy=_accuracy,
        hit=_hit,
        prompt_tokens=POOL_PROMPT_TOKENS,
        prompt_ids=prompt_ids,
        perf_prompt_tokens=POOL_PERF_PROMPT_TOKENS,
        replicas=DATA_PARALLEL_SIZE,
        # Build the pool leg first so a pool-side failure surfaces without
        # paying for the control leg.
        pool_first=True,
    )
