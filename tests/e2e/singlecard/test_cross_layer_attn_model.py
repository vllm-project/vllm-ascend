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
"""
Compare the outputs of cross layer attention model with and without aclgraph.

Run `pytest tests/e2e/singlecard/test_cross_layer_attn_model.py`.
"""

import os

import pytest

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = [
    "google/gemma-3n-E2B-it",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
def test_models_with_aclgraph(
    model: str,
    max_tokens: int,
) -> None:
    prompts = [
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is"
    ]

    with VllmRunner(
            model,
            max_model_len=1024,
            enforce_eager=False,
            cudagraph_capture_sizes=[4],
    ) as vllm_model:
        vllm_aclgraph_outputs = vllm_model.generate_greedy(prompts, max_tokens)

    with VllmRunner(
            model,
            max_model_len=1024,
            enforce_eager=True,
    ) as vllm_model:
        vllm_eager_outputs = vllm_model.generate_greedy(prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs,
        outputs_1_lst=vllm_aclgraph_outputs,
        name_0="vllm_eager_outputs",
        name_1="vllm_aclgraph_outputs",
    )
