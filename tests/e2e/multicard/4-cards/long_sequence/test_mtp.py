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
import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

os.environ["HCCL_BUFFSIZE"] = "512"

prompts = [
    "The capital of France is", "Hello, my name is Tom, I am",
    "The president of United States is", "AI future is"
]
model = "wemaster/deepseek_mtp_main_random_bf16"

@wait_until_npu_memory_free()
def test_pcp_dcp_mtp1_eager():
    with VllmRunner(
            model,
            max_model_len=1024,
            tensor_parallel_size=2,
            prefill_context_parallel_size=2,
            decode_context_parallel_size=2,
            max_num_batched_tokens=1024,
            enable_expert_parallel=True,
            block_size=128,
            speculative_config={
                "num_speculative_tokens": 1,
                "method": "deepseek_mtp",
            },
            enforce_eager=True,
            async_scheduling=False,
    ) as runner:
        runner.generate_greedy(prompts, 32)


@wait_until_npu_memory_free()
def test_pcp_dcp_mtp3_eager():
    with VllmRunner(
            model,
            max_model_len=1024,
            tensor_parallel_size=2,
            prefill_context_parallel_size=2,
            decode_context_parallel_size=2,
            max_num_batched_tokens=1024,
            enable_expert_parallel=True,
            block_size=128,
            async_scheduling=True,
            speculative_config={
                "num_speculative_tokens": 3,
                "method": "deepseek_mtp",
            },
            enforce_eager=True,
    ) as runner:
        runner.generate_greedy(prompts, 32)


@wait_until_npu_memory_free()
def test_pcp_dcp_mtp3_piecewise_graph():
    with VllmRunner(
            model,
            max_model_len=1024,
            tensor_parallel_size=2,
            prefill_context_parallel_size=2,
            decode_context_parallel_size=2,
            max_num_batched_tokens=1024,
            enable_expert_parallel=True,
            block_size=128,
            speculative_config={
                "num_speculative_tokens": 3,
                "method": "deepseek_mtp",
            },
            compilation_config={
                "cudagraph_mode": "PIECEWISE",
                "cudagraph_capture_sizes": [4, 8, 16],
            },
            async_scheduling=False,
    ) as runner:
        runner.generate_greedy(prompts, 32)


@wait_until_npu_memory_free()
def test_pcp_dcp_mtp3_full_graph():
    with VllmRunner(
            model,
            max_model_len=1024,
            tensor_parallel_size=2,
            prefill_context_parallel_size=2,
            decode_context_parallel_size=2,
            max_num_batched_tokens=1024,
            enable_expert_parallel=True,
            block_size=128,
            speculative_config={
                "num_speculative_tokens": 3,
                "method": "deepseek_mtp",
            },
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [4, 8, 16],
            },
            async_scheduling=False,
    ) as runner:
        runner.generate_greedy(prompts, 32)


@wait_until_npu_memory_free()
def test_dcp_mtp3_full_graph():
    with VllmRunner(
            model,
            max_model_len=1024,
            tensor_parallel_size=2,
            decode_context_parallel_size=2,
            max_num_batched_tokens=1024,
            enable_expert_parallel=True,
            block_size=128,
            speculative_config={
                "num_speculative_tokens": 3,
                "method": "deepseek_mtp",
            },
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [4, 8, 16],
            },
            async_scheduling=False,
    ) as runner:
        runner.generate_greedy(prompts, 32)
