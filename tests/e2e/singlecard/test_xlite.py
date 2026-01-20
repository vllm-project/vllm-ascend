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

import os

import pytest
from vllm import SamplingParams

from tests.e2e.singlecard.utils import (PROMPTS_SHORT, LLMTestCase,
                                        gen_and_valid)

os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"

CASE_DECODE_ONLY = LLMTestCase(
    model="Qwen/Qwen3-0.6B",
    prompts=PROMPTS_SHORT,
    golden_answers=[
        "Hello, my name is Lina. I'm a 22-year-old student from China.",
        'The president of the United States is the same as the president of the United Nations. This is because the president',
        'The capital of France is Paris. The capital of France is also the capital of the French Republic.',
        'The future of AI is not just a technological challenge but a profound transformation of how we live, work'
    ],
    sampling_params=SamplingParams(
        max_tokens=15,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        n=1,
    ))

CASE_FULL_DECODE_ONLY = LLMTestCase(
    model="Qwen/Qwen3-0.6B",
    prompts=[
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is"
    ],
    golden_answers=[
        " Lina. I'm a 22-year-old student from China. I'm interested in studying in the US. I'm looking for a job in the",
        ' the same as the president of the United Nations. This is because the president of the United States is the same as the president of the United Nations. The president',
        ' Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China is Beijing. The capital of Japan is Tokyo. The capital',
        " not just about the technology itself, but about how we use it to solve real-world problems. As AI continues to evolve, it's important to consider the ethical"
    ],
    sampling_params=SamplingParams(
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        n=1,
    ))


@pytest.mark.skip(
    reason="TODO: Re-enable xlite_decode_only e2e test when stable.")
@pytest.mark.parametrize("cur_case", [CASE_DECODE_ONLY])
def test_models_with_xlite_decode_only(cur_case: LLMTestCase):
    runner_kwargs = {
        "model_name": cur_case.model,
        "max_model_len": 1024,
        "block_size": 128,
        "additional_config": {
            "xlite_graph_config": {
                "enabled": True
            }
        },
    }
    gen_and_valid(runner_kwargs=runner_kwargs,
                  prompts=cur_case.prompts,
                  sampling_params=cur_case.sampling_params,
                  golden_answers=cur_case.golden_answers)


@pytest.mark.parametrize("cur_case", [CASE_FULL_DECODE_ONLY])
def test_models_with_xlite_full_mode(cur_case: LLMTestCase):
    runner_kwargs = {
        "model_name": cur_case.model,
        "max_model_len": 1024,
        "block_size": 128,
        "additional_config": {
            "xlite_graph_config": {
                "enabled": True,
                "full_mode": True
            }
        },
    }
    gen_and_valid(runner_kwargs=runner_kwargs,
                  prompts=cur_case.prompts,
                  sampling_params=cur_case.sampling_params,
                  golden_answers=cur_case.golden_answers)
