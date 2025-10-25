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

from tests.e2e.conftest import VllmRunner


def test_models_distributed_Qwen3_NEXT_TP4():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner("Qwen/Qwen3-Next-80B-A3B-Instruct",
                    tensor_parallel_size=4,
                    max_model_len=4096,
                    gpu_memory_utilization=0.7,
                    distributed_executor_backend="mp",
                    enforce_eager=True) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
