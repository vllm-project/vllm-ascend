#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
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
"""Tests whether ascend quantization based on MindIE-Turbo is enabled correctly.

Run `pytest tests/quantization/test_mindie_turbo.py`.
"""

import os

import pytest

import vllm  # noqa: F401

import vllm_ascend  # noqa: F401

from tests.conftest import VllmRunner
from tests.quantization.utils import is_mindie_turbo_supported, example_quantization

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]


@pytest.mark.skipif(not is_mindie_turbo_supported(),
                    reason="MindIE-Turbo is not installed.")
@pytest.mark.parametrize("model_name_or_path", MODELS)
@pytest.mark.parametrize("max_tokens", [5])
def test_mindie_turbo(
    model_name_or_path: str,
    max_tokens: int,
) -> None:
    # vLLM must load weights from disk. Hence we need to save the quantized
    # weights at first, and then load it by vLLM.
    temp_path = os.path.join(os.path.dirname(__file__), "temp_weight")
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    example_quantization(model_name_or_path, temp_path)

    prompt = "What's deep learning?"
    example_prompts = [prompt]

    with VllmRunner(temp_path,
                    max_model_len=8192,
                    dtype="bfloat16",
                    enforce_eager=False,
                    gpu_memory_utilization=0.7) as vllm_model:

        output = vllm_model.generate_greedy(example_prompts, max_tokens)
        assert output