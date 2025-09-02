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
import pytest
import vllm  # noqa: F401
from vllm import SamplingParams

import vllm_ascend  # noqa: F401
from tests.e2e.conftest import VllmRunner

MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-7B-Instruct"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("max_tokens", [5])
def test_models(model: str, dtype: str, max_tokens: int) -> None:
    example_prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]

    with VllmRunner(model,
                    tensor_parallel_size=1,
                    dtype=dtype,
                    max_model_len=2048,
                    enforce_eager=True,
                    compilation_config={
                        "custom_ops":
                        ["none", "+rms_norm", "+rotary_embedding"]
                    }) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


VL_MODELS = ["Qwen/Qwen2.5-VL-3B-Instruct"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float16"])
def test_vl_model_with_samples(model: str, dtype: str) -> None:
    example_prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]

    with VllmRunner(model,
                    tensor_parallel_size=1,
                    dtype=dtype,
                    max_model_len=2048,
                    enforce_eager=True,
                    compilation_config={
                        "custom_ops":
                        ["none", "+rms_norm", "+rotary_embedding"]
                    }) as vllm_model:
        sampling_params = SamplingParams(max_tokens=100,
                                         top_p=0.95,
                                         top_k=50,
                                         temperature=0.6)
        vllm_model.generate(example_prompts, sampling_params)
