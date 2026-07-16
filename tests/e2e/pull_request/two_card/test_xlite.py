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
# This file is a part of the vllm-ascend project.
#
"""Verify MoE serving with xlite graph mode on two NPU cards.

Run `pytest tests/e2e/pull_request/two_card/test_xlite.py`.
"""

import os

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.utils import PROMPTS_SHORT

os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"

MODELS: list[str] = ["/mnt/sdb/models/Qwen3-30B-A3B"]


@pytest.mark.e2e_model(*MODELS)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="xlite",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="xlite_decode_only",
)
@pytest.mark.parametrize("model", MODELS)
@wait_until_npu_memory_free()
def test_models_with_xlite_decode_only(model: str):
    with VllmRunner(
        model,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        block_size=128,
        max_model_len=2048,
        additional_config={"xlite_graph_config": {"enabled": True, "full_mode": False}},
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(PROMPTS_SHORT, 3)

    assert all(output[1] for output in outputs)


@pytest.mark.e2e_model(*MODELS)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="xlite",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="xlite_full",
)
@pytest.mark.parametrize("model", MODELS)
@wait_until_npu_memory_free()
def test_models_with_xlite_full_mode(model: str):
    with VllmRunner(
        model,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        block_size=128,
        max_model_len=2048,
        additional_config={"xlite_graph_config": {"enabled": True, "full_mode": True}},
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(PROMPTS_SHORT, 3)

    assert all(output[1] for output in outputs)
