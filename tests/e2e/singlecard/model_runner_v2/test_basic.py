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

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

MODELS = ["Qwen/Qwen3-0.6B"]

MAIN_MODELS = ["LLM-Research/Meta-Llama-3.1-8B-Instruct"]
EGALE_MODELS = ["vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [True])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen3_dense_eager_mode(
    model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
            model,
            max_model_len=1024,
            enforce_eager=enforce_eager,
    ) as runner:
        runner.model.generate(prompts, sampling_params)


@pytest.mark.parametrize("model", MAIN_MODELS)
@pytest.mark.parametrize("eagle_model", EGALE_MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [True])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_egale_spec_decoding(
    model: str,
    eagle_model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
            model,
            max_model_len=1024,
            enforce_eager=enforce_eager,
            async_scheduling=True,
            speculative_config={
                "model": eagle_model,
                "method": "eagle",
                "num_speculative_tokens": 3,
            },
    ) as runner:
        runner.model.generate(prompts, sampling_params)
