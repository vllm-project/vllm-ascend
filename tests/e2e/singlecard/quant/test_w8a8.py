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

import pytest
from modelscope import snapshot_download  # type: ignore[import-untyped]

from tests.e2e.conftest import VllmRunner

MODELS = [
    "vllm-ascend/DeepSeek-V2-Lite-W8A8",
    "vllm-ascend/Qwen2.5-0.5B-Instruct-W8A8"
]


@pytest.mark.parametrize("model", MODELS)
def test_quant_W8A8(example_prompts, model):
    max_tokens = 5
    model_path = snapshot_download(model)
    with VllmRunner(
            model_path,
            max_model_len=8192,
            enforce_eager=True,
            dtype="auto",
            gpu_memory_utilization=0.7,
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
