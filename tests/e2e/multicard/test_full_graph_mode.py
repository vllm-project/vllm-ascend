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
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/e2e/multicard/test_qwen3_moe.py`.
"""

import os

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal


def test_models_distributed_Qwen3_MOE_TP2_WITH_FULLGRAPH():
    if 'HCCL_OP_EXPANSION_MODE' in os.environ:
        del os.environ['HCCL_OP_EXPANSION_MODE']
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
    model = "Qwen/Qwen3-30B-A3B"
    sampling_params = SamplingParams(max_tokens=5,
                                     n=1,
                                     temperature=0.0,
                                     top_p=1.0,
                                     top_k=1)
    with VllmRunner(model,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    enforce_eager=False,
                    gpu_memory_utilization=0.95,
                    compilation_config={
                        "cudagraph_capture_sizes":
                        [4, 8, 12, 16, 24, 32, 40, 48],
                        "cudagraph_mode": "FULL_DECODE_ONLY"
                    }) as runner:
        vllm_fullgraph_outputs = runner.model.generate(prompts,
                                                       sampling_params)
    with VllmRunner(
            model,
            max_model_len=1024,
            tensor_parallel_size=2,
            enforce_eager=True,
            gpu_memory_utilization=0.95,
    ) as runner:
        vllm_eager_outputs = runner.model.generate(prompts, sampling_params)

    vllm_fullgraph_outputs_list = []
    for output in vllm_fullgraph_outputs:
        vllm_fullgraph_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    vllm_eager_outputs_list = []
    for output in vllm_eager_outputs:
        vllm_eager_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_fullgraph_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_fullgraph_outputs",
    )
