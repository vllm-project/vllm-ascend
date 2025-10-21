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

import os
import random
import string

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODELS = [
    "Qwen/Qwen3-0.6B",
    "vllm-ascend/DeepSeek-V2-Lite-W8A8",
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
    if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
        with VllmRunner(
                model,
                max_model_len=1024,
                enforce_eager=False,
                quantization="ascend",
        ) as runner:
            vllm_aclgraph_outputs = runner.model.generate(
                prompts, sampling_params)

        with VllmRunner(
                model,
                max_model_len=1024,
                enforce_eager=True,
                quantization="ascend",
        ) as runner:
            vllm_eager_outputs = runner.model.generate(prompts,
                                                       sampling_params)
    else:
        with VllmRunner(
                model,
                max_model_len=1024,
                enforce_eager=False,
        ) as runner:
            vllm_aclgraph_outputs = runner.model.generate(
                prompts, sampling_params)

        with VllmRunner(
                model,
                max_model_len=1024,
                enforce_eager=True,
        ) as runner:
            vllm_eager_outputs = runner.model.generate(prompts,
                                                       sampling_params)
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


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [5])
def test_models_with_aclgraph_full_decode_only(
    model: str,
    max_tokens: int,
) -> None:
    if 'HCCL_OP_EXPANSION_MODE' in os.environ:
        del os.environ['HCCL_OP_EXPANSION_MODE']
    # NOTE: Randomly fill the prompt with the requested amount for
    # the specified capture shape to prevent accuracy issues caused by padding
    random_number = random.choice(list(range(6, 47, 8)))
    prompts = [
        ('Solve the following math problem step by step.'
         'The last line of your response should be of the form Answer: '
         '$Answer (without quotes) where $Answer is the answer to the problem.\n\n'
         'In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$'
         'be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$,'
         '$\\angle BDC = 90^\\circ$. Suppose $AD = 1$ and $\\frac{BD}{CD} = \\frac{3}{2}$.'
         'If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$,'
         'where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.'
         ),
        ('Solve the following math problem step by step.'
         'The last line of your response should be of the form Answer: '
         '$Answer (without quotes) where $Answer is the answer to the problem.\n\n'
         'Let $ABCD$ be a unit square in the plane. Points $X$ and $Y$ are chosen'
         'independently and uniformly at random on the perimeter of $ABCD$.'
         'If the expected value of the area of triangle $\\triangle AXY$'
         'can be expressed as $\\frac{m}{n}$, for relatively prime positive'
         'integers $m$ and $n$, compute $m+n$.'),
        ('Solve the following math problem step by step.'
         'The last line of your response should be of the form Answer: '
         '$Answer (without quotes) where $Answer is the answer to the problem.\n\n'
         'Let $a, b, c$ be distinct numbers such that the equations $x^2 + ax + 1 = 0$'
         'and $x^2 + bx + c = 0$ have a common real root, and the equations $x^2 + x + a = 0$'
         'and $x^2 + cx + b = 0$ also have a common real root.'
         'Compute the sum $a + b + c$.')
    ] + [
        ''.join(random.choices(string.ascii_lowercase, k=random.randint(
            1, 25))) for _ in range(random_number)
    ]

    sampling_params = SamplingParams(max_tokens=5,
                                     n=1,
                                     temperature=0.0,
                                     top_p=1.0,
                                     top_k=1)
    if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
        with VllmRunner(
                model,
                max_model_len=1024,
                enforce_eager=False,
                compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
                quantization="ascend",
        ) as runner:
            vllm_aclgraph_outputs = runner.model.generate(
                prompts, sampling_params)

        with VllmRunner(
                model,
                max_model_len=1024,
                enforce_eager=True,
                quantization="ascend",
        ) as runner:
            vllm_eager_outputs = runner.model.generate(prompts,
                                                       sampling_params)
    else:
        with VllmRunner(
                model,
                max_model_len=1024,
                enforce_eager=False,
                compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        ) as runner:
            vllm_aclgraph_outputs = runner.model.generate(
                prompts, sampling_params)

        with VllmRunner(
                model,
                max_model_len=1024,
                enforce_eager=True,
        ) as runner:
            vllm_eager_outputs = runner.model.generate(prompts,
                                                       sampling_params)

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
