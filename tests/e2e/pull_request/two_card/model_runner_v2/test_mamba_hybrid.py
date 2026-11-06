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
#
"""
Inference with Model Runner V2 for Mamba/hybrid model.

Run `pytest -sv tests/e2e/pull_request/two_card/model_runner_v2/test_mamba_hybrid.py`.
"""

import os
from unittest.mock import patch

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

QWEN35_DENSE_MODEL = os.environ.get("QWEN35_DENSE_MODEL", "Qwen/Qwen3.5-27B")


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(0.8)
def test_qwen35_27b_eager_mode():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        QWEN35_DENSE_MODEL,
        data_parallel_size=1,
        tensor_parallel_size=2,
        enable_expert_parallel=False,
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        distributed_executor_backend="mp",
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        assert outputs[0][1]


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(0.8)
@pytest.mark.parametrize("enable_batch_sharded_sampling", [True, False])
def test_qwen35_27b_acl_graph(enable_batch_sharded_sampling):
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        QWEN35_DENSE_MODEL,
        data_parallel_size=1,
        tensor_parallel_size=2,
        enable_expert_parallel=False,
        enable_batch_sharded_sampling=enable_batch_sharded_sampling,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        distributed_executor_backend="mp",
        cudagraph_capture_sizes=[1, 2, 4, 8],
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "enforce_eager": True,
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        assert outputs[0][1]
