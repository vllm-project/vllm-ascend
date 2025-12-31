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

Run `pytest tests/compile/test_aclgraph_accuracy.py`.
"""

import os

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
def test_models_output_between_eager_and_aclgraph(
    model: str,
    max_tokens: int,
) -> None:
    prompts = [
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is"
    ]

    vllm_aclgraph_qwen_answers = [
        " Lina. I'm a 22-year-old student from China. I'm interested in studying in the US. I want to know if there are any",
        ' the same as the president of the United Nations. This is because the president of the United States is the same as the president of the United Nations. The president',
        ' Paris. The capital of France is also the capital of the Republic of France. The capital of France is also the capital of the European Union. The capital of',
        ' not just a technological frontier but a profound transformation of how we live, work, and interact with the world. As we stand at the intersection of artificial intelligence and'
    ]

    vllm_aclgraph_ds_answers = [
        '\nI am a 20 year old student from the UK. I am currently studying for a degree in English Literature and Creative Writing. I have a passion',
        ' a man who has been in the public eye for decades. He has been a senator, a governor, and a businessman. He has also been married to the',
        ' Paris, which is also the largest city in the country. The city is located on the River Seine and is known for its beautiful architecture, museums, and art',
        ' here.\nThe future of AI is here.\nThe future of AI is here.\nThe future of AI is here.\nThe future of AI is'
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
        with VllmRunner(
                model,
                max_model_len=1024,
                cudagraph_capture_sizes=[1, 2, 4, 8],
                quantization="ascend",
        ) as runner:
            vllm_aclgraph_outputs = runner.model.generate(
                prompts, sampling_params)
    else:
        with VllmRunner(
                model,
                max_model_len=1024,
                cudagraph_capture_sizes=[1, 2, 4, 8],
        ) as runner:
            vllm_aclgraph_outputs = runner.model.generate(
                prompts, sampling_params)
    vllm_aclgraph_outputs_list = []
    for output in vllm_aclgraph_outputs:
        vllm_aclgraph_outputs_list.append(
            ([output.outputs[0].index], output.outputs[0].text))

    vllm_eager_outputs_list = ([
        ([0], answer) for answer in vllm_aclgraph_ds_answers
    ] if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8" else [
        ([0], answer) for answer in vllm_aclgraph_qwen_answers
    ])

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_aclgraph_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_aclgraph_outputs",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
def test_models_output_between_eager_and_full_decode_only(
    model: str,
    max_tokens: int,
) -> None:
    if 'HCCL_OP_EXPANSION_MODE' in os.environ:
        del os.environ['HCCL_OP_EXPANSION_MODE']
    # NOTE: Randomly fill the prompt with the requested amount for
    # the specified capture shape to prevent accuracy issues caused by padding
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
    ]
    vllm_aclgraph_qwen_answers = [
        ' \n\nTo solve this problem, we need to use the Law of Sines and Law of Cosines. Let me start by drawing triangle $ABC$ with the',
        ' \n\nTo solve this problem, we can use the following approach: Let $ABCD$ be a unit square with coordinates $A(0,0), B',
        ' \n\nTo solve this problem, we can use the following approach: Let $ \\alpha $ be the common real root of the two equations. Then, we can'
    ]

    vllm_aclgraph_ds_answers = [
        '\n\nSelect an assignment template',
        '\n\nSelect an assignment template',
        '\n\nSelect an assignment template'
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens,
                                     n=1,
                                     temperature=0.0,
                                     top_p=1.0,
                                     top_k=1)
    if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
        with VllmRunner(
                model,
                max_model_len=1024,
                compilation_config={
                    "cudagraph_capture_sizes": [4, 8, 32, 64],
                    "cudagraph_mode": "FULL_DECODE_ONLY"
                },
                quantization="ascend",
        ) as runner:
            vllm_aclgraph_outputs = runner.model.generate(
                prompts, sampling_params)

    else:
        with VllmRunner(
                model,
                max_model_len=1024,
                compilation_config={
                    "cudagraph_capture_sizes": [4, 8, 32, 64],
                    "cudagraph_mode": "FULL_DECODE_ONLY"
                },
        ) as runner:
            vllm_aclgraph_outputs = runner.model.generate(
                prompts, sampling_params)

    vllm_aclgraph_outputs_list = []
    for output in vllm_aclgraph_outputs:
        vllm_aclgraph_outputs_list.append(
            ([output.outputs[0].index], output.outputs[0].text))
    vllm_eager_outputs_list = []
    vllm_eager_outputs_list = ([
        ([0], answer) for answer in vllm_aclgraph_ds_answers
    ] if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8" else [
        ([0], answer) for answer in vllm_aclgraph_qwen_answers
    ])

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_aclgraph_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_aclgraph_outputs",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
def test_models_output_between_eager_and_fullgraph_npugraph_ex(
    model: str,
    max_tokens: int,
) -> None:
    if 'HCCL_OP_EXPANSION_MODE' in os.environ:
        del os.environ['HCCL_OP_EXPANSION_MODE']
    # NOTE: Randomly fill the prompt with the requested amount for
    # the specified capture shape to prevent accuracy issues caused by padding
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
    ]
    vllm_aclgraph_qwen_answers = [
        ' \n\nTo solve this problem, we need to use the Law of Sines and Law of Cosines. Let me start by drawing triangle $ABC$ with the',
        " \n\nTo solve this problem, we can use the fact that the expected value of the area of a triangle formed by two random points on a square's perimeter is",
        ' \n\nTo solve this problem, we can use the following approach: Let $ \\alpha $ be the common real root of the two equations. Then, we can'
    ]

    vllm_aclgraph_ds_answers = [
        '\n\nSelect an assignment template',
        '\n\nSelect an assignment template',
        '\n\nSelect an assignment template'
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens,
                                     n=1,
                                     temperature=0.0,
                                     top_p=1.0,
                                     top_k=1)
    if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
        with VllmRunner(
                model,
                max_model_len=1024,
                compilation_config={
                    "cudagraph_capture_sizes": [4, 8, 32, 64],
                    "cudagraph_mode": "FULL_DECODE_ONLY"
                },
                additional_config={"enable_npugraph_ex": True},
                quantization="ascend",
        ) as runner:
            vllm_aclgraph_outputs = runner.model.generate(
                prompts, sampling_params)

    else:
        with VllmRunner(
                model,
                max_model_len=1024,
                compilation_config={
                    "cudagraph_capture_sizes": [4, 8, 32, 64],
                    "cudagraph_mode": "FULL_DECODE_ONLY"
                },
                additional_config={"enable_npugraph_ex": True},
        ) as runner:
            vllm_aclgraph_outputs = runner.model.generate(
                prompts, sampling_params)

    vllm_aclgraph_outputs_list = []
    for output in vllm_aclgraph_outputs:
        vllm_aclgraph_outputs_list.append(
            ([output.outputs[0].index], output.outputs[0].text))
    vllm_eager_outputs_list = []
    vllm_eager_outputs_list = ([
        ([0], answer) for answer in vllm_aclgraph_ds_answers
    ] if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8" else [
        ([0], answer) for answer in vllm_aclgraph_qwen_answers
    ])

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_aclgraph_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_aclgraph_outputs",
    )


def test_aclgraph_enable():
    # Generally, this test is not belong to e2e, but it is a good way to check if
    # aclgraph is enabled in real environment
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm.engine.arg_utils import EngineArgs

    from vllm_ascend.platform import NPUPlatform

    # vLLM default mode is piecewise cudagraph
    config = EngineArgs()
    VllmConfig = config.create_engine_config()
    assert VllmConfig.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE

    # after check_and_update_config, mode should be VLLM_COMPILE and piecewise cudagraph
    NPUPlatform.check_and_update_config(VllmConfig)
    assert VllmConfig.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert VllmConfig.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
