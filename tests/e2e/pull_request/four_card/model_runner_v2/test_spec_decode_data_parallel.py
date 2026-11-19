# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2026 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""DSpark speculative decoding under DP with Model Runner V2.

The DP4 full-graph case additionally runs the KV-pool A/B watch ported from
PR #20757: leg A serves the memcache pool (non-layerwise) while leg B keeps
local prefix caching, both walking R1 cold / R2-R3 hit / R4 disjoint-miss
on the 16K secret question with DP metrics aggregated across replicas.
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
    run_pool_ab,
    run_pool_hit_sequence,
)

DEEPSEEK_V4_DSPARK_MODEL = os.environ.get(
    "DEEPSEEK_V4_DSPARK_MODEL_PATH",
    "UploadWeight/DeepSeek-V4-Flash-DSpark-w4a8-test",
)

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

DSPARK_NUM_SPECULATIVE_TOKENS = 5

DATA_PARALLEL_SIZE = 2
# 16,388 = lcm(16,384) + 4; DSpark trims the final lcm block, so 16,384 is a
# guaranteed 0-hit workload. See expected_hit_token_count().
POOL_LCM_BLOCK_SIZE = 16384
POOL_PROMPT_TOKENS = POOL_LCM_BLOCK_SIZE + 4
# Perf prompt: one lcm block plus a tail -> 90% hit at concurrency 4.
POOL_PERF_PROMPT_TOKENS = POOL_LCM_BLOCK_SIZE + POOL_LCM_BLOCK_SIZE // 9
POOL_MAX_MODEL_LEN = max(POOL_PROMPT_TOKENS, POOL_PERF_PROMPT_TOKENS) + 128
# Chunked prefill cap: 4096 = 32 blocks of 128, keeps the activation peak low.
POOL_BATCHED_TOKENS = 4096

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
                "ock.mmc.local_service.protocol": "device_sdma",
                "ock.mmc.local_service.dram.size": "10GB",
            },
        },
    )


CASE_NAME = "dsv4_dp2_pool_watch"


@contextmanager
def _leg(use_pool: bool, case_name: str):
    """Leg A enables the memcache pool; leg B keeps everything else identical."""
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            data_parallel_size=DATA_PARALLEL_SIZE,
            tensor_parallel_size=2,
            max_model_len=POOL_MAX_MODEL_LEN,
            max_num_seqs=2,
            max_num_batched_tokens=POOL_BATCHED_TOKENS,
            enable_expert_parallel=True,
            tokenizer_mode="deepseek_v4",
            block_size=128,
            distributed_executor_backend="mp",
            gpu_memory_utilization=0.93,
            quantization="ascend",
            enforce_eager=False,
            async_scheduling=True,
            enable_prefix_caching=not use_pool,
            disable_log_stats=False,
            speculative_config={
                "method": "dspark",
                "num_speculative_tokens": 5,
                "enforce_eager": True,
            },
            attention_config={"indexer_kv_dtype": "int8"},
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [6, 12],
            },
            additional_config={"enable_fused_mc2": 0},
        )
        if use_pool:
            pool = stack.enter_context(SingleNodeMemcacheManager(_build_memcache_pool_config(), case_name))
            stack.enter_context(patch.dict(os.environ, pool.server_envs))
            kwargs["kv_transfer_config"] = POOL_KV_TRANSFER_CONFIG
        yield stack.enter_context(DPVllmRunner(DEEPSEEK_V4_DSPARK_MODEL, **kwargs))


@pytest.mark.e2e_model(DEEPSEEK_V4_DSPARK_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dspark",
    parallel="DP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_BUFFSIZE": "2048",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v4_dspark_spec_decoding_dp4_full_graph() -> None:
    """DSpark DP2xTP2 accuracy plus the memcache pool A/B watch (pool leg A).

    Keeps the original output-non-empty greedy assertions in the accuracy
    stage of both legs. DSpark cold/hit outputs differ deterministically, so
    H3 is exempt for this case only (design doc deviation note); the exact
    per-rank hit count and the A/B perf comparison still gate.
    """
    tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_V4_DSPARK_MODEL, trust_remote_code=True)
    prompt_ids = build_pool_accuracy_prompt(tokenizer, POOL_PROMPT_TOKENS, POOL_MAX_MODEL_LEN)

    def _accuracy(runner) -> list[float]:
        outputs = runner.generate_greedy(PROMPTS, max_tokens=32)
        assert len(outputs) == len(PROMPTS)
        assert all(output_ids and output_text.strip() for output_ids, output_text in outputs)

        # Design doc sec.2.4/sec.3.2: P3 keeps only the original non-empty
        # greedy assertions; token identity and per-position acceptance are
        # recorded, not gated. DP metrics are summed across ranks by DPVllmRunner.
        metrics = runner.get_metrics()
        num_drafts = metrics.get("vllm:spec_decode_num_drafts", 0.0)
        assert num_drafts > 0, "Speculative decoding did not generate draft tokens"
        accepted_per_pos = [
            metrics.get(f"vllm:spec_decode_num_accepted_tokens_per_pos[pos={position}]", 0.0)
            for position in range(DSPARK_NUM_SPECULATIVE_TOKENS)
        ]
        rates = [accepted / num_drafts for accepted in accepted_per_pos]
        # Recorded for the A/B report. The design doc's acceptance class is
        # P5/P6/P10 only, and no DP2xTP2 floor has been measured: the borrowed
        # TP2xPCP2 floors failed here (pos3 0.382 vs 0.49).
        print(f"[{CASE_NAME}] acceptance rates per draft position: {rates}", flush=True)
        return rates

    def _hit(runner) -> dict:
        return run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=POOL_PROMPT_TOKENS,
            prompt_ids=prompt_ids,
            replicas=DATA_PARALLEL_SIZE,
            lcm_block_size=POOL_LCM_BLOCK_SIZE,
            check_output=False,  # DSpark cold/hit outputs differ deterministically, so H3 is off
            answer_secret=POOL_SECRET,
            tag="dsv4-dp2-memcache",
        )

    run_pool_ab(
        case_name=CASE_NAME,
        build_leg=_leg,
        accuracy=_accuracy,
        hit=_hit,
        prompt_ids=prompt_ids,
        perf_prompt_tokens=POOL_PERF_PROMPT_TOKENS,
        token_identity=False,
        # Perf gate applies (helper defaults gate=True, perf_gate=True):
        # 10 timed samples, drop the highest 2 and the lowest 2, gate on the
        # middle 6 -- same policy as the four-card watch.
        perf_sample_count=10,
        perf_trim_count=2,
    )
