#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/entrypoints/llm/test_guided_generate.py
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
#
import gc
import os

import pytest
import torch
from vllm import LLM, SamplingParams

from tests.e2e.conftest import ModelName, cleanup_dist_env_and_memory, model_cache

os.environ["VLLM_BATCH_INVARIANT"] = "1"


@pytest.mark.timeout(1000)
@pytest.mark.model(
    model_name=ModelName.QWEN3_06B,
    quantization=None,
    max_model_len=8192,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    enable_prefix_caching=False,
    max_num_seqs=32,
    tensor_parallel_size=1,
    distributed_executor_backend="mp",
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 32, 64]},
)
def test_qwen3_topk(vllm_runner) -> None:
    example_prompts = [
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0, top_k=50, top_p=0.9)
    vllm_runner.generate(example_prompts, sampling_params)


@pytest.mark.timeout(1000)
@pytest.mark.model(
    model_name=ModelName.QWEN3_06B,
    quantization=None,
    max_model_len=8192,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    enable_prefix_caching=False,
    max_num_seqs=32,
    tensor_parallel_size=1,
    distributed_executor_backend="mp",
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 32, 64]},
)
def test_qwen3_prompt_logprobs(vllm_runner) -> None:
    example_prompts = [
        "Hello, my name is",
    ]
    vllm_runner.generate_greedy_logprobs(example_prompts, max_tokens=5, num_logprobs=1)


@pytest.mark.timeout(1000)
def test_qwen3_exponential_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    # enable_async_exponential is mutually exclusive with VLLM_BATCH_INVARIANT
    # (see vllm_ascend/ascend_config.py). The module-level os.environ setting
    # would silently disable async_exponential, so this test creates its own
    # LLM instance with batch invariant mode turned off.
    model_cache.clear()
    gc.collect()
    torch.npu.empty_cache()

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")

    llm = LLM(
        model=ModelName.QWEN3_06B,
        quantization=None,
        max_model_len=8192,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
        max_num_seqs=32,
        tensor_parallel_size=1,
        distributed_executor_backend="mp",
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8]},
        additional_config={"enable_async_exponential": True},
    )

    example_prompts = [
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(max_tokens=5, temperature=1.0, top_k=50, top_p=0.9)
    llm.generate(example_prompts, sampling_params)

    del llm
    cleanup_dist_env_and_memory()
