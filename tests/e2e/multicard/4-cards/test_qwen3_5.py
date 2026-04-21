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

from tests.e2e.conftest import VllmRunner


def test_qwen3_5_27b_distributed_mp_tp4():
    example_prompts = [
        "Hello, my name is",
    ] * 4
    max_tokens = 5
    with VllmRunner(
        "Qwen/Qwen3.5-27B",
        tensor_parallel_size=4,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
        del vllm_model


def test_qwen3_5_35b_distributed_mp_tp4():
    example_prompts = [
        "Hello, my name is",
    ] * 4
    max_tokens = 5
    with VllmRunner(
        "Qwen/Qwen3.5-35B-A3B",
        tensor_parallel_size=4,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
        del vllm_model


def test_qwen3_5_35b_distributed_mp_tp4_full_decode_only_mtp3():
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    max_tokens = 20
    with VllmRunner(
        "Qwen/Qwen3.5-35B-A3B",
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        distributed_executor_backend="mp",
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4, 8, 12, 16],
        },
        speculative_config={
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": 3,
        },
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
        del vllm_model


@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})
def test_qwen3_5_35b_distributed_mp_tp4_full_decode_only_mtp3_flashcomm():
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    max_tokens = 20
    with VllmRunner(
        "Qwen/Qwen3.5-35B-A3B",
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        distributed_executor_backend="mp",
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4, 8, 12, 16],
        },
        speculative_config={
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": 3,
        },
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
        del vllm_model
