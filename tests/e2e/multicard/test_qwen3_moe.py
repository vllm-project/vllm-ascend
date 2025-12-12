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

Run `pytest tests/e2e/multicard/test_qwen3_moe.py`.
"""

import os
from unittest.mock import patch

from modelscope import snapshot_download  # type: ignore

from tests.e2e.conftest import VllmRunner


@patch.dict(os.environ, {"HCCL_BUFFSIZE": "1024"})
def test_qwen3_moe_distributed_mp_tp2_ep():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_qwen3_moe_w8a8_distributed_tp2():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
            snapshot_download("vllm-ascend/Qwen3-30B-A3B-W8A8"),
            max_model_len=8192,
            tensor_parallel_size=2,
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@patch.dict(os.environ, {"HCCL_BUFFSIZE": "1024"})
def test_qwen3_moe_w8a8_distributed_tp2_ep():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
            snapshot_download("vllm-ascend/Qwen3-30B-A3B-W8A8"),
            max_model_len=8192,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_qwen3_moe_distributed_aiv_tp2():
    os.environ['HCCL_OP_EXPANSION_MODE'] = 'AIV'
    example_prompts = [
        "Hello, my name is",
    ]
    dtype = "auto"
    max_tokens = 5
    with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            dtype=dtype,
            tensor_parallel_size=2,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
