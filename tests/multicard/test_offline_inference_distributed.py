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

Run `pytest tests/test_offline_inference.py`.
"""
import os
from unittest.mock import patch

import vllm  # noqa: F401
from vllm import SamplingParams

from tests.conftest import VllmRunner

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"


def test_models_distributed_QwQ():
    example_prompts = [
        "Hello, my name is",
    ]
    dtype = "half"
    max_tokens = 5
    with VllmRunner(
            "Qwen/QwQ-32B",
            dtype=dtype,
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_models_distributed_DeepSeek():
    example_prompts = [
        "Hello, my name is",
    ]
    dtype = "half"
    max_tokens = 5
    with VllmRunner(
            "deepseek-ai/DeepSeek-V2-Lite",
            dtype=dtype,
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_TOPK_OPTIMZE": "1"})
def test_models_distributed_topk() -> None:
    example_prompts = [
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
        "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.",
        "Compare and contrast artificial intelligence with human intelligence in terms of processing information.",
    ]
    dtype = "half"
    sampling_params = SamplingParams(max_tokens=5,
                                     temperature=0.0,
                                     top_k=50,
                                     top_p=0.9)

    with VllmRunner(
            "deepseek-ai/DeepSeek-V2-Lite",
            dtype=dtype,
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)
