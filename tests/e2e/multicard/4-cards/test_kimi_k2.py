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
import os

import pytest

from tests.e2e.conftest import VllmRunner


os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


@pytest.mark.skip(reason="CANN8.5 failed, capture stream failed, fix me")
def test_kimi_k2_thinking_w4a16_tp4():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(
            "vllm-ascend/Kimi-K2-Thinking-Pruning",
            max_model_len=8192,
            dtype="auto",
            tensor_parallel_size=4,
            enable_expert_parallel=True,
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1],
            },
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
