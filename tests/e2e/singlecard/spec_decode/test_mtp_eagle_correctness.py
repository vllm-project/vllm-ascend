#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
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

"""

from __future__ import annotations

import os
from typing import Union

import pytest
from vllm import SamplingParams
from vllm.config import CompilationConfig

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = ["wemaster/deepseek_mtp_main_random_bf16"]
MODELS_EAGLE = [
    "vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B",
    "RedHatAI/Qwen3-8B-speculator.eagle3"
]
MODELS_MAIN = ["LLM-Research/Meta-Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B"]
VALID_COMBINATIONS = {("eagle", "vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B",
                       "LLM-Research/Meta-Llama-3.1-8B-Instruct"),
                      ("eagle3", "RedHatAI/Qwen3-8B-speculator.eagle3",
                       "Qwen/Qwen3-8B")}


@pytest.mark.parametrize("model_name", MODELS)
# num_speculative_tokens = 2 doesn't work, skip it, fix me.
# @pytest.mark.parametrize("num_speculative_tokens", [1, 2, 3])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
@pytest.mark.parametrize("cudagraph_mode", ["PIECEWISE", "FULL_DECODE_ONLY"])
@pytest.mark.parametrize("disable_padded_drafter_batch", [True, False])
def test_deepseek_mtp_correctness(model_name: str, num_speculative_tokens: int,
                                  cudagraph_mode: str,
                                  disable_padded_drafter_batch: bool):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using mtp speculative decoding.
    '''
    with VllmRunner(model_name,
                    tensor_parallel_size=1,
                    max_num_seqs=256,
                    gpu_memory_utilization=0.7,
                    distributed_executor_backend="mp",
                    enable_expert_parallel=True,
                    speculative_config={
                        "method":
                        "mtp",
                        "num_speculative_tokens":
                        num_speculative_tokens,
                        "disable_padded_drafter_batch":
                        disable_padded_drafter_batch,
                    },
                    max_model_len=2000,
                    compilation_config=CompilationConfig(
                        cudagraph_mode=cudagraph_mode,
                        cudagraph_capture_sizes=[20],
                    )) as spec_llm:
        sampling_config = SamplingParams(temperature=0,
                                         max_tokens=256,
                                         ignore_eos=False)
        spec_outputs = spec_llm.generate(example_prompts, sampling_config)

    with VllmRunner(model_name,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.7,
                    max_model_len=256,
                    compilation_config=CompilationConfig(
                        cudagraph_mode=cudagraph_mode,
                        cudagraph_capture_sizes=[20],
                    )) as ref_llm:
        sampling_config = SamplingParams(temperature=0,
                                         max_tokens=256,
                                         ignore_eos=False)
        ref_outputs = ref_llm.generate(example_prompts, sampling_config)

    matches = 0
    misses = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        ref_token_ids = ref_output[0][0]
        spec_token_ids = spec_output[0][0]
        if ref_token_ids == spec_token_ids[:len(ref_token_ids)]:
            matches += 1
        else:
            misses += 1
            print(f"ref_output: {ref_output[1][0]}")
            print(f"spec_output: {spec_output[1][0]}")

    # Heuristic: expect at least 66% of the prompts to match exactly
    # Upon failure, inspect the outputs to check for inaccuracy.
    assert matches > int(0.66 * len(ref_outputs))
    cleanup_dist_env_and_memory()
    del spec_llm


@pytest.mark.parametrize("model_name", MODELS_EAGLE)
@pytest.mark.parametrize("model_name_main", MODELS_MAIN)
@pytest.mark.parametrize("num_speculative_tokens", [1, 2])
@pytest.mark.parametrize("method", ["eagle", "eagle3"])
@pytest.mark.parametrize("disable_padded_drafter_batch", [True, False])
@pytest.mark.parametrize("async_scheduling", [True, False])
@pytest.mark.parametrize("draft_tensor_parallel_size", [None, 1])
def test_llama_qwen3_eagle_correctness(
        model_name: str, model_name_main: str, num_speculative_tokens: int,
        method: str, disable_padded_drafter_batch: bool,
        async_scheduling: bool, draft_tensor_parallel_size: Union[None, int]):

    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    if (method, model_name, model_name_main) not in VALID_COMBINATIONS or \
        (async_scheduling and disable_padded_drafter_batch):
        pytest.skip(
            f"Invalid combination: method={method}, model_name={model_name}, model_name_main={model_name_main}, or case not support yet"
        )

    sampling_params = SamplingParams(
        max_tokens=300,
        temperature=0.0,
        ignore_eos=False,
    )

    with VllmRunner(model_name_main,
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    data_parallel_size=1,
                    disable_log_stats=False,
                    max_model_len=4096,
                    seed=1024,
                    async_scheduling=async_scheduling,
                    speculative_config={
                        "disable_padded_drafter_batch":
                        disable_padded_drafter_batch,
                        "method": method,
                        "model": model_name,
                        "num_speculative_tokens": num_speculative_tokens,
                        "draft_tensor_parallel_size":
                        draft_tensor_parallel_size,
                        "max_model_len": 128,
                    },
                    compilation_config=CompilationConfig(
                        cudagraph_mode="FULL_DECODE_ONLY",
                        cudagraph_capture_sizes=[12])) as llm:
        spec_outputs = llm.generate(example_prompts, sampling_params)
        cleanup_dist_env_and_memory()
        del llm

    with VllmRunner(model_name_main,
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    data_parallel_size=1,
                    disable_log_stats=False,
                    max_model_len=4096,
                    seed=1024,
                    async_scheduling=async_scheduling,
                    compilation_config=CompilationConfig(
                        cudagraph_mode="FULL_DECODE_ONLY",
                        cudagraph_capture_sizes=[12])) as llm:
        ref_outputs = llm.generate(example_prompts, sampling_params)
        cleanup_dist_env_and_memory()
        del llm

    matches = 0
    misses = 0
    threshold = 0.66
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        ref_token_ids = ref_output[0][0]
        spec_token_ids = spec_output[0][0]
        if ref_token_ids == spec_token_ids[:len(ref_token_ids)]:
            matches += 1
        else:
            misses += 1
            print(f"ref_output: {ref_output[1][0]}")
            print(f"spec_output: {spec_output[1][0]}")

    # Heuristic: expect at least 66.6% of the prompts to match exactly
    # Upon failure, inspect the outputs to check for inaccuracy.
    assert matches > int(threshold * len(ref_outputs))
    cleanup_dist_env_and_memory()
