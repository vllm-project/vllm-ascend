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
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/e2e/multicard/test_qwen3_moe.py`.
"""

import os

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from vllm_ascend.utils import vllm_version_is

os.environ["HCCL_BUFFSIZE"] = "768"


@pytest.mark.skipif(vllm_version_is('0.12.0'),
                    reason="0.12.0 is not supported for context sequence.")
def test_models_pcp_dcp_basic():
    prompts = [
        "The capital of France is", "Hello, my name is Tom, I am",
        "The president of United States is", "AI future is"
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    enforce_eager=True,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=2,
                    max_num_batched_tokens=1024,
                    enable_expert_parallel=True,
                    block_size=128) as runner:
        runner.model.generate(prompts, sampling_params)

    model = "vllm-ascend/Qwen3-30B-A3B-W8A8"
    with VllmRunner(
            model,
            enforce_eager=True,
            max_model_len=1024,
            tensor_parallel_size=2,
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            enable_expert_parallel=True,
            block_size=128,
            quantization="ascend",
    ) as runner:
        runner.model.generate(prompts, sampling_params)


@pytest.mark.skipif(vllm_version_is('0.12.0'),
                    reason="0.12.0 is not supported for context sequence.")
def test_models_pcp_dcp_full_graph():
    prompts = [
        "The capital of France is", "Hello, my name is Tom, I am",
        "The president of United States is", "AI future is"
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=2,
                    max_num_batched_tokens=1024,
                    enable_expert_parallel=True,
                    block_size=128,
                    compilation_config={
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
                    }) as runner:
        runner.model.generate(prompts, sampling_params)

    model = "vllm-ascend/Qwen3-30B-A3B-W8A8"
    with VllmRunner(model,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=1,
                    enable_expert_parallel=True,
                    block_size=128,
                    quantization="ascend",
                    compilation_config={
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
                    }) as runner:
        runner.model.generate(prompts, sampling_params)


@pytest.mark.skipif(vllm_version_is('0.12.0'),
                    reason="0.12.0 is not supported for context sequence.")
def test_models_pcp_dcp_piece_wise():
    prompts = [
        "The capital of France is", "Hello, my name is Tom, I am",
        "The president of United States is", "AI future is"
    ]
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=2,
                    max_num_batched_tokens=1024,
                    enable_expert_parallel=True,
                    block_size=128) as runner:
        runner.model.generate(prompts, sampling_params)

    model = "vllm-ascend/Qwen3-30B-A3B-W8A8"
    with VllmRunner(model,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    prefill_context_parallel_size=2,
                    decode_context_parallel_size=1,
                    enable_expert_parallel=True,
                    block_size=128,
                    quantization="ascend") as runner:
        runner.model.generate(prompts, sampling_params)
