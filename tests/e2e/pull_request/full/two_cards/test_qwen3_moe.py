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
import os
from unittest.mock import patch

import pytest

from tests.e2e.conftest import VllmRunner
from vllm_ascend.utils import vllm_version_is


@pytest.mark.skipif(vllm_version_is("0.20.2"), reason="no need to support model_runner for v0.20.2")
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [True])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen3_moe_distributed_tp2_ep2_mrv2(
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    example_prompts = [
        "The president of the United States is",
    ]

    with VllmRunner(
        "Qwen/Qwen3-30B-A3B",
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enforce_eager=enforce_eager,
    ) as vllm_model:
        vllm_output = vllm_model.generate_greedy(example_prompts, max_tokens)

    golden_results = [
        "The president of the United States is the commander in chief of",
    ]

    for i in range(len(vllm_output)):
        assert golden_results[i] == vllm_output[i][1]
