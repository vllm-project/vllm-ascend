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

# ruff: noqa: E501

import os

import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.singlecard.utils import PROMPTS_LONG, PROMPTS_SHORT, LLMTestCase, compare_logprobs

# ---------------------------------------------------------------------------
# Test cases – no golden_answers needed; accuracy is verified via logprob
# comparison against an eager-mode baseline.  Token 0 covers the prefill
# forward pass; tokens 1-2 cover decode forward passes.
# ---------------------------------------------------------------------------

CASE_QWEN_ACLGRAPH = LLMTestCase(
    model="Qwen/Qwen3-0.6B",
    prompts=PROMPTS_SHORT,
)

CASE_DS_ACLGRAPH = LLMTestCase(
    model="vllm-ascend/DeepSeek-V2-Lite-W8A8",
    quantization="ascend",
    prompts=PROMPTS_SHORT,
)

CASE_QWEN_FULL = LLMTestCase(
    model="Qwen/Qwen3-0.6B",
    prompts=PROMPTS_SHORT,
)

CASE_DS_FULL = LLMTestCase(
    model="vllm-ascend/DeepSeek-V2-Lite-W8A8",
    quantization="ascend",
    prompts=PROMPTS_SHORT,
)

CASE_QWEN_FULL_DECODE_ONLY = LLMTestCase(
    model="Qwen/Qwen3-0.6B",
    prompts=PROMPTS_LONG,
)

CASE_DS_FULL_DECODE_ONLY = LLMTestCase(
    model="vllm-ascend/DeepSeek-V2-Lite-W8A8",
    quantization="ascend",
    prompts=PROMPTS_LONG,
)

CASE_QWEN_EX = LLMTestCase(
    model="Qwen/Qwen3-0.6B",
    prompts=PROMPTS_LONG,
)

CASE_DS_EX = LLMTestCase(
    model="vllm-ascend/DeepSeek-V2-Lite-W8A8",
    quantization="ascend",
    prompts=PROMPTS_LONG,
)


@wait_until_npu_memory_free(0.7)
@pytest.mark.parametrize("cur_case", [CASE_QWEN_ACLGRAPH, CASE_DS_ACLGRAPH])
def test_piecewise_res_consistency(cur_case: LLMTestCase):
    runner_kwargs = {
        "model_name": cur_case.model,
        "max_model_len": 1024,
        "cudagraph_capture_sizes": [1, 2, 4, 8],
        "quantization": cur_case.quantization,
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=cur_case.prompts)


@wait_until_npu_memory_free(0.7)
@pytest.mark.parametrize("cur_case", [CASE_QWEN_FULL, CASE_DS_FULL])
def test_full_res_consistency(cur_case: LLMTestCase, monkeypatch):
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    runner_kwargs = {
        "model_name": cur_case.model,
        "max_model_len": 1024,
        "compilation_config": {"cudagraph_capture_sizes": [4, 8, 32, 64], "cudagraph_mode": "FULL_DECODE_ONLY"},
        "quantization": cur_case.quantization,
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=cur_case.prompts)


@wait_until_npu_memory_free(0.7)
@pytest.mark.parametrize("cur_case", [CASE_QWEN_FULL_DECODE_ONLY, CASE_DS_FULL_DECODE_ONLY])
def test_full_decode_only_res_consistency(cur_case: LLMTestCase, monkeypatch):
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    runner_kwargs = {
        "model_name": cur_case.model,
        "max_model_len": 1024,
        "compilation_config": {"cudagraph_capture_sizes": [4, 8, 32, 64], "cudagraph_mode": "FULL_DECODE_ONLY"},
        "quantization": cur_case.quantization,
        "additional_config": {"ascend_compilation_config": {"enable_npugraph_ex": False}},
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=cur_case.prompts)


@wait_until_npu_memory_free(0.7)
@pytest.mark.parametrize("cur_case", [CASE_QWEN_EX, CASE_DS_EX])
def test_npugraph_ex_res_consistency(cur_case: LLMTestCase, monkeypatch):
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    runner_kwargs = {
        "model_name": cur_case.model,
        "quantization": cur_case.quantization,
        "max_model_len": 1024,
        "compilation_config": {"cudagraph_capture_sizes": [4, 8, 32, 64], "cudagraph_mode": "FULL_DECODE_ONLY"},
        "additional_config": {"ascend_compilation_config": {"enable_npugraph_ex": True}},
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=cur_case.prompts)


# The accuracy has already been verified in the previous test case.
# This test case is used to check whether the functionality works properly
# after enabling the static kernel and whether it is uninstalled as expected.
@wait_until_npu_memory_free(0.7)
@pytest.mark.parametrize("cur_case", [CASE_QWEN_EX])
def test_npugraph_ex_with_static_kernel(cur_case: LLMTestCase, monkeypatch):
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    runner_kwargs = {
        "model_name": cur_case.model,
        "quantization": cur_case.quantization,
        "max_model_len": 1024,
        "compilation_config": {"cudagraph_capture_sizes": [4, 8], "cudagraph_mode": "FULL_DECODE_ONLY"},
        "additional_config": {
            "ascend_compilation_config": {
                "enable_npugraph_ex": True,
                "enable_static_kernel": True,
            }
        },
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=cur_case.prompts)

    # Check whether the static kernel is properly uninstalled
    ascend_home_path = os.environ["ASCEND_HOME_PATH"]
    static_kernel_install_path = os.path.join(ascend_home_path, "opp/static_kernel/ai_core")
    assert not os.path.exists(static_kernel_install_path)
