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
#

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

MODELS = ["deepseek-ai/DeepSeek-V2-Lite", "Qwen/Qwen2.5-0.5B-Instruct"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half", "float16"])
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE": "1"})
def test_models_distributed_topk(model, dtype) -> None:
    example_prompts = [
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
        "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.",
        "Compare and contrast artificial intelligence with human intelligence in terms of processing information.",
    ]
    sampling_params = SamplingParams(max_tokens=5,
                                     temperature=0.0,
                                     top_k=50,
                                     top_p=0.9)

    with VllmRunner(
            model,
            dtype=dtype,
            gpu_memory_utilization=0.7,
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)
