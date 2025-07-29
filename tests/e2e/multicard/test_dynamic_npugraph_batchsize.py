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
#
import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]

TENSOR_PARALLELS = [2]

prompts = [
    "Hello, my name is",
    "The future of AI is",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("temperature", [0.0])
@pytest.mark.parametrize("ignore_eos", [True])
def test_models(model: str, tp_size: int, max_tokens: int, temperature: int,
                ignore_eos: bool) -> None:
    # Create an LLM.
    with VllmRunner(
            model_name=model,
            tensor_parallel_size=tp_size,
    ) as vllm_model:
        # Prepare sampling_parames
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=ignore_eos,
        )

        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects
        outputs = vllm_model.generate(prompts, sampling_params)
        torch.npu.synchronize()
        # The output length should be equal to prompts length.
        assert len(outputs) == len(prompts)
