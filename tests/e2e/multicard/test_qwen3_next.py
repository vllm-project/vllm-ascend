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

Run `pytest tests/e2e/multicard/test_qwen3_next.py`.
"""

import os
from unittest.mock import patch

from modelscope import snapshot_download  # type: ignore

from tests.e2e.conftest import VllmRunner


def test_qwen3_next_distributed_mp_tp4():
    example_prompts = [
        "Hello, my name is",
    ] * 4
    max_tokens = 5
    with VllmRunner("Qwen/Qwen3-Next-80B-A3B-Instruct",
                    tensor_parallel_size=4,
                    max_model_len=4096,
                    gpu_memory_utilization=0.8,
                    distributed_executor_backend="mp") as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
        del vllm_model


def test_qwen3_next_distributed_mp_full_decode_only_tp4():
    example_prompts = [
        "Hello, my name is",
    ] * 4
    max_tokens = 5
    with VllmRunner("Qwen/Qwen3-Next-80B-A3B-Instruct",
                    tensor_parallel_size=4,
                    max_model_len=4096,
                    gpu_memory_utilization=0.8,
                    distributed_executor_backend="mp",
                    compilation_config={
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [1, 8, 24, 48, 60]
                    }) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
        del vllm_model


# TODO: Fix the accuary of batch chunked prefill
def test_qwen3_next_distributed_mp_eager_mtp_similarity_tp4():
    example_prompts = ["Hello, my name is"]
    max_tokens = 20

    with VllmRunner(
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            tensor_parallel_size=4,
            max_model_len=4096,
            gpu_memory_utilization=0.8,
            distributed_executor_backend="mp",
            enforce_eager=True,
    ) as vllm_model:
        ref_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    with VllmRunner("Qwen/Qwen3-Next-80B-A3B-Instruct",
                    tensor_parallel_size=4,
                    max_model_len=4096,
                    gpu_memory_utilization=0.8,
                    distributed_executor_backend="mp",
                    enforce_eager=True,
                    speculative_config={
                        "method": "qwen3_next_mtp",
                        "num_speculative_tokens": 1
                    }) as spec_vllm_model:
        spec_outputs = spec_vllm_model.generate_greedy(example_prompts,
                                                       max_tokens)
    del spec_vllm_model

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

    assert matches > int(0.66 * len(ref_outputs))


# TODO: will conduct accuracy verification after the subsequent version becomes stable
@patch.dict(os.environ, {"HCCL_BUFFSIZE": "1024"})
def test_qwen3_next_w8a8dynamic_distributed_tp4_ep():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
            snapshot_download("vllm-ascend/Qwen3-Next-80B-A3B-Instruct-W8A8"),
            max_model_len=4096,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.4,
            max_num_seqs=1,
            enable_expert_parallel=True,
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
