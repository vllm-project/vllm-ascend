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
import os
import pytest

from tests.e2e.conftest import VllmRunner

os.environ["HCCL_BUFFSIZE"] = "512"

QWEN3_5_MODELS = [
    "Qwen/Qwen3.5-35B-A3B",
]
EXAMPLE_PROMPTS = [
        "Hello, my name is",
    ]
MAX_TOKENS = 5
COMPILATION_CONFIG = {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 8, 24, 48, 60],
    }


@pytest.mark.parametrize("model", QWEN3_5_MODELS)
def test_qwen3_5_35b(model) -> None:

    with VllmRunner(
        model,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        max_model_len=512,
        compilation_config=COMPILATION_CONFIG,
        gpu_memory_utilization=0.7,
    ) as runner:
        runner.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)

@pytest.mark.parametrize("model", QWEN3_5_MODELS)
def test_qwen3_5_35b_mtp(model) -> None:
    
    speculative_config = {
        "num_speculative_tokens": 3,
        "method": "qwen3_5_mtp",
    }
    with VllmRunner(
        model,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        max_model_len=512,
        compilation_config=COMPILATION_CONFIG,
        speculative_config=speculative_config,
        gpu_memory_utilization=0.7,
    ) as runner:
        runner.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)