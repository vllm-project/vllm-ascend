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
Compare the outputs of vLLM with and without xlite.

Run `pytest tests/e2e/singlecard/test_xlite.py`.
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
def test_models_with_xlite_decode_only(
    model: str,
    max_tokens: int,
) -> None:
    prompts = [
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is"
    ]

    vllm_xlite_qwen_answers = [
        " Lina. I'm a 22-year-old student from China. I'm interested in studying in the US. I want to know if there are any",
        " the same as the president of the United Nations. This is because the president of the United States is the same as the president of the United Nations. The president",
        " Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China is Beijing. The capital of Japan is Tokyo. The capital",
        " not just a technological challenge but a profound transformation of how we live, work, and interact with the world. As we stand at the intersection of artificial intelligence and"
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
            model,
            block_size=128,
            max_model_len=1024,
            enforce_eager=False,
            additional_config={"xlite_graph_config": {
                "enabled": True
            }},
    ) as runner:
        vllm_xlite_outputs = runner.model.generate(prompts, sampling_params)

    vllm_xlite_outputs_list = []
    for output in vllm_xlite_outputs:
        vllm_xlite_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    vllm_eager_outputs_list = ([
        (0, answer) for answer in vllm_xlite_qwen_answers
    ])

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_xlite_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_xlite_outputs",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
def test_models_with_xlite_full_mode(
    model: str,
    max_tokens: int,
) -> None:
    prompts = [
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is"
    ]

    vllm_xlite_qwen_answers = [
        " Lina. I'm a 22-year-old student from China. I'm interested in studying in the US. I'm looking for a job in the",
        " the same as the president of the United Nations. This is because the president of the United States is the same as the president of the United Nations. The president",
        " Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China is Beijing. The capital of Japan is Tokyo. The capital",
        " not just about the technology itself, but about how we use it to solve real-world problems. As AI continues to evolve, it's important to consider the ethical"
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
            model,
            block_size=128,
            max_model_len=1024,
            enforce_eager=False,
            additional_config={
                "xlite_graph_config": {
                    "enabled": True,
                    "full_mode": True
                }
            },
    ) as runner:
        vllm_xlite_outputs = runner.model.generate(prompts, sampling_params)

    vllm_xlite_outputs_list = []
    for output in vllm_xlite_outputs:
        vllm_xlite_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    vllm_eager_outputs_list = ([
        (0, answer) for answer in vllm_xlite_qwen_answers
    ])

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_xlite_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_xlite_outputs",
    )
