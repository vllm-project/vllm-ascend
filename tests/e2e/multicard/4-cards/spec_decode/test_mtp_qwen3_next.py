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

Run `pytest tests/e2e/multicard/spec_decode/test_mtp_qwen3_next.py`.
"""

import os

import pytest
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = ["Qwen/Qwen3-Next-80B-A3B-Instruct"]


@pytest.mark.parametrize("model_name", MODELS)
def test_qwen3_next_mtp_acceptance_tp4(model_name):
    golden = [0.85, 0.46, 0.19]

    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    max_tokens = 1024

    with VllmRunner(model_name,
                    tensor_parallel_size=4,
                    max_model_len=4096,
                    gpu_memory_utilization=0.8,
                    distributed_executor_backend="mp",
                    disable_log_stats=False,
                    speculative_config={
                        "method": "qwen3_next_mtp",
                        "num_speculative_tokens": 3,
                    },
                    compilation_config=CompilationConfig(
                        cudagraph_capture_sizes=[20])) as spec_vllm_model:
        _ = spec_vllm_model.generate_greedy(example_prompts, max_tokens)
        metrics = spec_vllm_model.model.get_metrics()

    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * 3
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                num_accepted_tokens_per_pos[pos] += metric.values[pos]

    acceptance_per_pos = [
        num_accepted_tokens / num_drafts
        for num_accepted_tokens in num_accepted_tokens_per_pos
    ]

    match = all((a >= b) or (b - a < 0.06) for a, b in zip(acceptance_per_pos, golden))
    if not match:
        print(f"acceptance_per_pos: {acceptance_per_pos}")
        print(f"golden: {golden}")

    assert match
    cleanup_dist_env_and_memory()


# FIXME: When applying `FULL_DECODE_ONLY` in this e2e, ci will fail.
# The failure can not be reproduced locally.
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("num_speculative_tokens", [1])
@pytest.mark.parametrize("disable_padded_drafter_batch", [True, False])
def test_qwen3_next_mtp_correctness_tp4(model_name: str,
                                        num_speculative_tokens: int,
                                        disable_padded_drafter_batch: bool):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    max_tokens = 20
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using mtp speculative decoding.
    '''
    with VllmRunner(model_name,
                    tensor_parallel_size=4,
                    max_model_len=4096,
                    gpu_memory_utilization=0.8,
                    distributed_executor_backend="mp",
                    speculative_config={
                        "method":
                        "mtp",
                        "num_speculative_tokens":
                        num_speculative_tokens,
                        "disable_padded_drafter_batch":
                        disable_padded_drafter_batch,
                    },
                    compilation_config=CompilationConfig(
                        cudagraph_capture_sizes=[20])) as spec_llm:
        spec_outputs = spec_llm.generate_greedy(example_prompts, max_tokens)
    del spec_llm

    with VllmRunner(model_name,
                    tensor_parallel_size=4,
                    max_model_len=4096,
                    gpu_memory_utilization=0.8,
                    distributed_executor_backend="mp",
                    compilation_config=CompilationConfig(
                        cudagraph_capture_sizes=[20])) as ref_llm:
        ref_outputs = ref_llm.generate_greedy(example_prompts, max_tokens)
    del ref_llm

    matches = 0
    misses = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        ref_token_ids = ref_output[0]
        spec_token_ids = spec_output[0]
        if ref_token_ids == spec_token_ids[:len(ref_token_ids)]:
            matches += 1
        else:
            misses += 1
            print(f"ref_output: {ref_output[1]}")
            print(f"spec_output: {spec_output[1]}")

    # Heuristic: expect at least 66% of the prompts to match exactly
    # Upon failure, inspect the outputs to check for inaccuracy.
    assert matches > int(0.66 * len(ref_outputs))
    cleanup_dist_env_and_memory()


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("num_speculative_tokens", [1])
def test_qwen3_next_mtp_full_decode(model_name: str,
                                        num_speculative_tokens: int):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    max_tokens = 20
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using mtp speculative decoding.
    '''
    with VllmRunner(model_name,
                    tensor_parallel_size=4,
                    max_model_len=4096,
                    gpu_memory_utilization=0.8,
                    distributed_executor_backend="mp",
                    speculative_config={
                        "method": "qwen3_next_mtp",
                        "num_speculative_tokens": num_speculative_tokens,
                    },
                    compilation_config=CompilationConfig(
                        cudagraph_mode="FULL_DECODE_ONLY",
                        cudagraph_capture_sizes=[4])) as llm:
        outputs = llm.generate_greedy(example_prompts, max_tokens)
        print(outputs)
    del llm
    cleanup_dist_env_and_memory()
