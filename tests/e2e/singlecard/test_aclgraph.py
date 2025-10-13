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
Compare the outputs of vLLM with and without aclgraph.

Run `pytest tests/compile/test_aclgraph.py`.
"""

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODELS = [
    "Qwen/Qwen3-0.6B",
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

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
            model,
            max_model_len=1024,
            enforce_eager=False,
    ) as runner:
        vllm_aclgraph_outputs = runner.model.generate(prompts, sampling_params)

    with VllmRunner(
            model,
            max_model_len=1024,
            enforce_eager=True,
    ) as runner:
        vllm_eager_outputs = runner.model.generate(prompts, sampling_params)

    vllm_aclgraph_outputs_list = []
    for output in vllm_aclgraph_outputs:
        vllm_aclgraph_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    vllm_eager_outputs_list = []
    for output in vllm_eager_outputs:
        vllm_eager_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_aclgraph_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_aclgraph_outputs",
    )
