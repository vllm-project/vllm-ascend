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

import pytest

from tests.e2e.singlecard.utils import (PROMPTS_LONG, PROMPTS_SHORT,
                                        LLMTestCase, gen_and_valid)

CASE_QWEN_ACLGRAPH = LLMTestCase(
    model="Qwen/Qwen3-0.6B",
    prompts=PROMPTS_SHORT,
    golden_answers=[
        " Lina. I'm a 22-year-old student from China. I'm interested in studying in the US. I want to know if there are any",
        ' the same as the president of the United Nations. This is because the president of the United States is the same as the president of the United Nations. The president',
        ' Paris. The capital of France is also the capital of the Republic of France. The capital of France is also the capital of the European Union. The capital of',
        ' not just a technological frontier but a profound transformation of how we live, work, and interact with the world. As we stand at the intersection of artificial intelligence and'
    ],
)

CASE_DS_ACLGRAPH = LLMTestCase(
    model="vllm-ascend/DeepSeek-V2-Lite-W8A8",
    quantization="ascend",
    prompts=PROMPTS_SHORT,
    golden_answers=[
        '\nI am a 20 year old female, and I have been suffering from depression for 3 years now. I have been on medication for 2',
        ' a man who has been in the public eye for decades. He has been a senator, a governor, and a businessman. He has also been married to the',
        ' Paris, which is also the largest city in the country. The city is located on the River Seine and is known for its beautiful architecture, museums, and art',
        ' here, and it’s not what you think.\nThe future of AI is here, and it’s not what you think.\nThe future of'
    ],
)

CASE_QWEN_FULL_DECODE_ONLY = LLMTestCase(
    model="Qwen/Qwen3-0.6B",
    prompts=PROMPTS_LONG,
    golden_answers=[
        ' \n\nTo solve this problem, we need to use the Law of Sines and Law of Cosines. Let me start by drawing triangle $ABC$ with the',
        " \n\nTo solve this problem, we can use the following approach: Let $ABCD$ be a unit square with coordinates $A(0,0), B",
        ' \n\nTo solve this problem, we can use the following approach: Let $ \\alpha $ be the common real root of the two equations. Then, we can'
    ])

CASE_DS_FULL_DECODE_ONLY = LLMTestCase(
    model="vllm-ascend/DeepSeek-V2-Lite-W8A8",
    quantization="ascend",
    prompts=PROMPTS_LONG,
    golden_answers=[
        '\n\nSelect an assignment template',
        '\n\nSelect an assignment template',
        '\n\nSelect an assignment template'
    ])

CASE_QWEN_EX = LLMTestCase(
    model="Qwen/Qwen3-0.6B",
    prompts=PROMPTS_LONG,
    golden_answers=[
        ' \n\nTo solve this problem, we need to use the Law of Sines and Law of Cosines. Let me start by drawing triangle $ABC$ with the',
        " \n\nTo solve this problem, we can use the fact that the expected value of the area of a triangle formed by two random points on a square's perimeter is",
        ' \n\nTo solve this problem, we can use the following approach: Let $ \\alpha $ be the common real root of the two equations. Then, we can'
    ])

CASE_DS_EX = LLMTestCase(model="vllm-ascend/DeepSeek-V2-Lite-W8A8",
                         quantization="ascend",
                         prompts=PROMPTS_LONG,
                         golden_answers=[
                             '\n\nSelect an assignment template',
                             '\n\nSelect an assignment template',
                             '\n\nSelect an assignment template'
                         ])


@pytest.mark.parametrize("cur_case", [CASE_QWEN_ACLGRAPH, CASE_DS_ACLGRAPH])
def test_piecewise_res_consistency(cur_case: LLMTestCase):
    runner_kwargs = {
        "model_name": cur_case.model,
        "max_model_len": 1024,
        "cudagraph_capture_sizes": [1, 2, 4, 8],
        "quantization": cur_case.quantization,
    }
    gen_and_valid(runner_kwargs=runner_kwargs,
                  prompts=cur_case.prompts,
                  sampling_params=cur_case.sampling_params,
                  golden_answers=cur_case.golden_answers)


@pytest.mark.parametrize(
    "cur_case", [CASE_QWEN_FULL_DECODE_ONLY, CASE_DS_FULL_DECODE_ONLY])
def test_full_decode_only_res_consistency(cur_case: LLMTestCase, monkeypatch):
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    runner_kwargs = {
        "model_name": cur_case.model,
        "max_model_len": 1024,
        "compilation_config": {
            "cudagraph_capture_sizes": [4, 8, 32, 64],
            "cudagraph_mode": "FULL_DECODE_ONLY"
        },
        "quantization": cur_case.quantization,
    }
    gen_and_valid(runner_kwargs=runner_kwargs,
                  prompts=cur_case.prompts,
                  sampling_params=cur_case.sampling_params,
                  golden_answers=cur_case.golden_answers)


@pytest.mark.parametrize("cur_case", [CASE_QWEN_EX, CASE_DS_EX])
def test_npugraph_ex_res_consistency(cur_case: LLMTestCase, monkeypatch):
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    runner_kwargs = {
        "model_name": cur_case.model,
        "quantization": cur_case.quantization,
        "max_model_len": 1024,
        "compilation_config": {
            "cudagraph_capture_sizes": [4, 8, 32, 64],
            "cudagraph_mode": "FULL_DECODE_ONLY"
        },
        "additional_config": {
            "enable_npugraph_ex": True
        },
    }
    gen_and_valid(runner_kwargs=runner_kwargs,
                  prompts=cur_case.prompts,
                  sampling_params=cur_case.sampling_params,
                  golden_answers=cur_case.golden_answers)
