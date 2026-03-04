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
import pytest

from tests.e2e.conftest import VllmRunner


TEST_CASES = [
    pytest.param(
        "neuralmagic/Qwen2.5-3B-quantized.w8a8",
        [
            "The president of the United States is the head of state and",
        ],
        id="dense-w8a8",
    ),
    pytest.param(
        "vllm-ascend/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8",
        [
            "The president of the United States is the head of state and",
        ],
        id="moe-w8a8-dynamic",
    ),
    pytest.param(
        "vllm-ascend/Qwen3-30B-A3B-Instruct-2507-quantized.w4a8",
        [
            "The president of the United States is the head of state and",
        ],
        id="moe-w4a8-dynamic",
    ),
    pytest.param(
        "cpatonn-mirror/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit",
        [
            "The president of the United States is the head of state and",
        ],
        id="moe-w4a16-dynamic",
    ),
]


@pytest.mark.parametrize("model_id, golden_results", TEST_CASES)
def test_compressed_tensors_tp2(model_id, golden_results):
    example_prompts = [
        "The president of the United States is",
    ]
    max_tokens = 5
    with VllmRunner(
        model_id,
        max_model_len=4096,
        tensor_parallel_size=2,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        gpu_memory_utilization=0.8,
    ) as vllm_model:
        vllm_output = vllm_model.generate_greedy(example_prompts, max_tokens)

    for i in range(len(vllm_output)):
        assert golden_results[i] == vllm_output[i][1]
        print(f"Generated text: {vllm_output[i][1]!r}")
