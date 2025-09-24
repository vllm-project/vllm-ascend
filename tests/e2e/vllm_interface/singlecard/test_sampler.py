#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/entrypoints/llm/test_guided_generate.py
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
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


def test_models_topk() -> None:
    example_prompts = [
        "The capital of France is",
    ]
    sampling_params = SamplingParams(max_tokens=10,
                                     temperature=0.0,
                                     top_k=10,
                                     top_p=0.9)

    with VllmRunner("Qwen/Qwen3-0.6B",
                    max_model_len=4096,
                    gpu_memory_utilization=0.7) as runner:
        runner.generate(example_prompts, sampling_params)
