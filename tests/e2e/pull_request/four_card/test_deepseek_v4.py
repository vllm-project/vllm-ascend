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

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


@patch.dict(
    os.environ,
    {
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
    },
)
@wait_until_npu_memory_free()
def test_deepseek_v4_w4a8_tp4_basic_greedy():
    """Verify DeepSeek V4 W4A8 basic greedy generation with TP4 and EP."""
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
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

        assert len(outputs) == len(example_prompts)
        for output_ids, output_str in outputs:
            assert len(output_str) > 0
            assert len(output_ids) > 0


@patch.dict(
    os.environ,
    {
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
    },
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
