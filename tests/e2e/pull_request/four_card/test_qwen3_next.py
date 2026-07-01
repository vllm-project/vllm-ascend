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
from unittest import mock

import pytest

from tests.e2e.conftest import VllmRunner

TEST_CASES = [
    # case 1: mp, tp4, cudagraph
    {
        "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "tp": 4,
        "gpu_mem": 0.8,
        "backend": "mp",
        "extra_args": {"cudagraph_capture_sizes": [1, 2, 4, 8]},
        "prompt_multiplier": 4,
        "env": {},
    },
    # case 2: FULL_DECODE_ONLY
    {
        "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "tp": 4,
        "gpu_mem": 0.8,
        "backend": "mp",
        "extra_args": {
            "compilation_config": {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 8, 24, 48, 60],
            }
        },
        "prompt_multiplier": 4,
        "env": {},
    },
    # case 3: W8A8 量化 + 专家并行
    {
        "model": "vllm-ascend/Qwen3-Next-80B-A3B-Instruct-W8A8",
        "tp": 4,
        "gpu_mem": 0.4,
        "backend": None,
        "extra_args": {
            "max_num_seqs": 1,
            "enable_expert_parallel": True,
            "cudagraph_capture_sizes": [1, 2, 4, 8],
            "quantization": "ascend",
        },
        "prompt_multiplier": 1,
        "env": {"HCCL_BUFFSIZE": "1024"},
    },
    # case 4: flash_comm + eager
    {
        "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "tp": 4,
        "gpu_mem": 0.7,
        "backend": "mp",
        "extra_args": {
            "enable_expert_parallel": True,
            "enforce_eager": True,
        },
        "prompt_multiplier": 4,
        "env": {
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
            "HCCL_BUFFSIZE": "1024",
        },
    },
    # case 5: graph mode (非 eager) + 专家并行
    {
        "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "tp": 4,
        "gpu_mem": 0.8,
        "backend": "mp",
        "extra_args": {
            "enable_expert_parallel": True,
            "cudagraph_capture_sizes": [1, 2, 8],
            "enforce_eager": False,
        },
        "prompt_multiplier": 4,
        "env": {"HCCL_BUFFSIZE": "1024"},
    },
]


@pytest.mark.parametrize("case", TEST_CASES)
def test_qwen3_next_distributed(case):
    prompt_text = "Hello, my name is"
    example_prompts = [prompt_text] * case["prompt_multiplier"]
    max_tokens = 5

    kwargs = {
        "model": case["model"],
        "tensor_parallel_size": case["tp"],
        "max_model_len": 4096,
        "gpu_memory_utilization": case["gpu_mem"],
    }
    if case["backend"] is not None:
        kwargs["distributed_executor_backend"] = case["backend"]
    for k, v in case["extra_args"].items():
        if v is not None:
            kwargs[k] = v

    with mock.patch.dict(os.environ, case["env"], clear=False), VllmRunner(**kwargs) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
