#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
import os
from unittest.mock import patch

import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMooncakeManager
from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.kv_pool_watch import run_pool_hit_sequence

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# A3: without fabric mem every cross-rank put fails (halDeviceDisableP2P
# drvRetCode=17 -> aclrtIpcMemImportByKey 107000) and R2 sees 0 hits.
# setdefault keeps it overridable on other platforms.
os.environ.setdefault("ASCEND_ENABLE_USE_FABRIC_MEM", "1")

# Local verification uses a mounted checkpoint; CI uses the HF id.
POOL_MODEL = os.environ.get("DEEPSEEK_V4_MTP_MODEL_PATH", "gdydems/DeepSeek-V4-Flash-w4a8-mtp")

# 16,388 = lcm(16,384) + 4. MTP trims the final lcm block, so a 16,384-token
# prompt would hit 0; the 4-token tail keeps 16,384 hit tokens.
POOL_LCM_BLOCK_SIZE = 16384
POOL_PROMPT_TOKENS = POOL_LCM_BLOCK_SIZE + 4
# max_model_len needs headroom for output tokens.
POOL_MAX_MODEL_LEN = POOL_PROMPT_TOKENS + 128

# DSV4-Flash block-key layerwise is the only mooncake layerwise slot in PR-A3.
POOL_KV_TRANSFER_CONFIG = {
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "backend": "mooncake",
        "lookup_rpc_port": "1",
        "use_layerwise": True,
    },
}


def _build_mooncake_pool_config() -> MooncakeKVPoolConfig:
    return MooncakeKVPoolConfig(
        config={
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "ascend",
            "global_segment_size": "2GB",  # was "5GB"; fabric mem mounts after HBM reservation
            "local_buffer_size": "5GB",
            "preferred_segment": False,
            "prefer_alloc_in_same_node": True,
        },
        master_port=get_open_port(),
        metrics_port=get_open_port(),
    )


@pytest.mark.e2e_model("gdydems/DeepSeek-V4-Flash-w4a8-mtp")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_deepseek_v4_w4a8_tp4_basic_greedy():
    """Verify DeepSeek V4 W4A8 basic greedy generation with TP4 and EP."""
    example_prompts = [
        "Hello, my name is",
        "What is the meaning of life?",
    ]
    max_tokens = 5

    pool_config = _build_mooncake_pool_config()
    with (
        SingleNodeMooncakeManager(pool_config, "dsv4_tp4_pool_watch") as pool,
        patch.dict(os.environ, pool.server_envs),
        VllmRunner(
            POOL_MODEL,
            max_model_len=POOL_MAX_MODEL_LEN,
            max_num_seqs=16,
            max_num_batched_tokens=POOL_PROMPT_TOKENS,
            dtype="auto",
            tensor_parallel_size=4,
            enable_expert_parallel=True,
            gpu_memory_utilization=0.9,
            quantization="ascend",
            tokenizer_mode="deepseek_v4",
            block_size=128,
            # KV pool watch: local prefix cache must be explicitly disabled,
            # and pool metrics require log stats.
            enable_prefix_caching=False,
            disable_log_stats=False,
            kv_transfer_config=POOL_KV_TRANSFER_CONFIG,
            attention_config={"indexer_kv_dtype": "int8"},
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
            },
            speculative_config={"num_speculative_tokens": 1, "method": "mtp"},
        ) as vllm_model,
    ):
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        expected_token_ids = [
            [19923, 14, 1026, 2329, 344, 680, 2852, 95, 305, 342],
            [3085, 344, 270, 5281, 294, 1988, 33, 3955, 361, 582, 3085, 344],
        ]
        assert len(outputs) == len(example_prompts)
        for i, (output_ids, output_str) in enumerate(outputs):
            assert len(output_str) > 0
            assert len(output_ids) > 0
            assert output_ids == expected_token_ids[i]

        # MTP: assert exact hit tokens (16,384), not a 0.9 ratio.
        run_pool_hit_sequence(
            vllm_model,
            block_size=128,
            prompt_tokens=POOL_PROMPT_TOKENS,
            lcm_block_size=POOL_LCM_BLOCK_SIZE,
            tag="dsv4-tp4-layerwise",
        )


@pytest.mark.e2e_model("gdydems/DeepSeek-V4-Flash-w4a8-mtp")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_deepseek_v4_w4a8_tp4_index_cache_freq4():
    """IndexCache freq=4 must produce non-empty greedy outputs identical in
    shape to the baseline test, verifying skip_topk/topk_indices_buffer
    plumbing (DSAModules → AscendDSAImpl) is wired correctly across both
    serial and dual-stream paths.
    """
    example_prompts = [
        "Hello, my name is",
        "The capital of France is",
        "What is the meaning of life?",
    ]
    max_tokens = 5

    with VllmRunner(
        "gdydems/DeepSeek-V4-Flash-w4a8-mtp",
        max_model_len=8192,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        dtype="auto",
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.9,
        quantization="ascend",
        tokenizer_mode="deepseek_v4",
        block_size=128,
        attention_config={"indexer_kv_dtype": "int8"},
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
        hf_overrides={
            "use_index_cache": True,
            "index_topk_freq": 4,
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

        assert len(outputs) == len(example_prompts)
        for output_ids, output_str in outputs:
            assert len(output_str) > 0
            assert len(output_ids) > 0
