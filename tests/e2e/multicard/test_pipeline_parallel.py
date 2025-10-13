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

from tests.e2e.conftest import VllmRunner

MODELS = [
    "Qwen/Qwen3-0.6B",
]

TENSOR_PARALLELS = [1]
PIPELINE_PARALLELS = [2]
DIST_EXECUTOR_BACKEND = ["mp", "ray"]

prompts = [
    "Hello, my name is",
    "The future of AI is",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("pp_size", PIPELINE_PARALLELS)
@pytest.mark.parametrize("distributed_executor_backend", DIST_EXECUTOR_BACKEND)
def test_models(model: str, tp_size: int, pp_size: int,
                distributed_executor_backend: str) -> None:
    with VllmRunner(model,
                    tensor_parallel_size=tp_size,
                    pipeline_parallel_size=pp_size,
                    distributed_executor_backend=distributed_executor_backend,
                    gpu_memory_utilization=0.7) as vllm_model:
        vllm_model.generate_greedy(prompts, 64)
