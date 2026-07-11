#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import os

import pytest

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MINIMAX_M2_MODELS = [
    "MiniMaxAI/MiniMax-Max-Text-01-hf",
]


@pytest.mark.parametrize("model", MINIMAX_M2_MODELS)
def test_minimax_m2(model) -> None:
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(model, max_model_len=512, gpu_memory_utilization=0.7) as runner:
        runner.generate_greedy(example_prompts, max_tokens)