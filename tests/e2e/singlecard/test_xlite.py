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
Compare the outputs of vLLM with and without xlite via logprob-based accuracy
check (3 tokens: 1 prefill + 2 decode).

Run `pytest tests/e2e/singlecard/test_xlite.py`.
"""

# ruff: noqa: E501

import os

import pytest

from tests.e2e.singlecard.utils import PROMPTS_SHORT, LLMTestCase, compare_logprobs

os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"

CASE_DECODE_ONLY = LLMTestCase(
    model="Qwen/Qwen3-0.6B",
    prompts=PROMPTS_SHORT,
)

CASE_FULL = LLMTestCase(
    model="Qwen/Qwen3-0.6B",
    prompts=PROMPTS_SHORT,
)


@pytest.mark.skip(reason="TODO: Re-enable xlite_decode_only e2e test when stable.")
@pytest.mark.parametrize("cur_case", [CASE_DECODE_ONLY])
def test_models_with_xlite_decode_only(cur_case: LLMTestCase):
    runner_kwargs = {
        "model_name": cur_case.model,
        "max_model_len": 1024,
        "block_size": 128,
        "additional_config": {"xlite_graph_config": {"enabled": True}},
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=cur_case.prompts)


@pytest.mark.parametrize("cur_case", [CASE_FULL])
def test_models_with_xlite_full_mode(cur_case: LLMTestCase):
    runner_kwargs = {
        "model_name": cur_case.model,
        "max_model_len": 1024,
        "block_size": 128,
        "additional_config": {"xlite_graph_config": {"enabled": True, "full_mode": True}},
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=cur_case.prompts)
