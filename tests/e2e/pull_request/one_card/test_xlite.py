#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

Run `pytest tests/e2e/pull_request/one_card/test_xlite.py`.
"""

# ruff: noqa: E501

import os

import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.utils import PROMPTS_SHORT, compare_logprobs

os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"

MODELS: list[str] = ["Qwen/Qwen3-0.6B"]


@pytest.mark.e2e_model(*MODELS)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="xlite",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="xlite_decode_only",
)
@pytest.mark.requires_hardware("A2")
@pytest.mark.requires_npus(1)
@pytest.mark.parametrize("model", MODELS)
@wait_until_npu_memory_free()
def test_models_with_xlite_decode_only(model: str):
    runner_kwargs = {
        "model_name": model,
        "max_model_len": 1024,
        "block_size": 128,
        "additional_config": {"xlite_graph_config": {"enabled": True, "full_mode": False}},
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=PROMPTS_SHORT)


@pytest.mark.e2e_model(*MODELS)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="xlite",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="xlite_full",
)
@pytest.mark.requires_hardware("A2")
@pytest.mark.requires_npus(1)
@pytest.mark.parametrize("model", MODELS)
@wait_until_npu_memory_free()
def test_models_with_xlite_full_mode(model: str):
    runner_kwargs = {
        "model_name": model,
        "max_model_len": 1024,
        "block_size": 128,
        "additional_config": {"xlite_graph_config": {"enabled": True, "full_mode": True}},
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=PROMPTS_SHORT)
