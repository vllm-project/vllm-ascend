#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from tests.e2e.conftest import VllmRunner


def test_qwen3_moe_tp4_fp16():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        "Qwen/Qwen3-30B-A3B",
        tensor_parallel_size=4,
        enforce_eager=True,
        dtype="float16",
        max_model_len=16384,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_qwen3_moe_ep4_fp16():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        "Qwen/Qwen3-30B-A3B",
        tensor_parallel_size=4,
        enforce_eager=True,
        dtype="float16",
        enable_expert_parallel=True,
        max_model_len=16384,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_qwen3_moe_tp2_w8a8():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        "vllm-ascend/Qwen3-30B-A3B-W8A8",
        tensor_parallel_size=2,
        enforce_eager=True,
        dtype="float16",
        quantization="ascend",
        max_model_len=16384,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
