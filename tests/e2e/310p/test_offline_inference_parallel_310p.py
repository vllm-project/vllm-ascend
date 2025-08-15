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

import vllm_ascend  # noqa: F401
from tests.e2e.conftest import VllmRunner

# Pangu local model path
MODELS = [
    "IntervitensInc/pangu-pro-moe-model",
]
# set additional config for ascend scheduler and torchair graph
ADDITIONAL_CONFIG = [{
    "additional_config": {
        "torchair_graph_config": {
            "enabled": True
        },
        "ascend_scheduler_config": {
            "enabled": True,
        }
    }
}]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enfore_eager", [True, False])
@pytest.mark.parametrize("additional_config", ADDITIONAL_CONFIG)
def test_pangu_model(model: str, dtype: str, max_tokens: int,
                     enfore_eager: bool, additional_config: dict) -> None:
    if enfore_eager:
        additional_config = {}
    example_prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]

    with VllmRunner(model,
                    tensor_parallel_size=4,
                    dtype=dtype,
                    max_model_len=1024,
                    enforce_eager=True,
                    enable_expert_parallel=True,
                    additional_config=additional_config,
                    distributed_executor_backend="mp") as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
