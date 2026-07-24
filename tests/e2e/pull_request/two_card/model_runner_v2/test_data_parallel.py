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
Data parallel inference with Model Runner V2.

Run `pytest -sv tests/e2e/pull_request/two_card/model_runner_v2/test_data_parallel.py`.
"""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free

MODELS = ["vllm-ascend/DeepSeek-V2-Lite-W8A8"]


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_dsv2_w8a8_dp_eager_mode():
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    max_tokens = 32
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.6, top_k=10)
    with DPVllmRunner(
        MODELS[0],
        data_parallel_size=2,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        max_model_len=1024,
        enforce_eager=True,
        quantization="ascend",
        distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params=sampling_params)


@pytest.mark.parametrize(
    "compilation_config",
    [
        pytest.param(
            {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8]},
            id="full_decode_only",
        ),
        pytest.param({}, id="default_full_and_piecewise"),
    ],
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_dsv2_w8a8_dp_graph_mode(compilation_config: dict):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    max_tokens = 20
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.6, top_p=0.95)
    with DPVllmRunner(
        MODELS[0],
        data_parallel_size=2,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        max_model_len=1024,
        quantization="ascend",
        distributed_executor_backend="mp",
        compilation_config=compilation_config,
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params=sampling_params)
