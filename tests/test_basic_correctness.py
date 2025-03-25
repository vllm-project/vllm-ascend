#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
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

"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/basic_correctness/test_basic_correctness.py`.
"""
import os

import pytest

from conftest import VllmRunner
from model_utils import check_outputs_equal

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct"
]

TARGET_TEST_SUITE = os.environ.get("TARGET_TEST_SUITE", "L4")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", ["ASCEND"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [False])
def test_models(
    monkeypatch: pytest.MonkeyPatch,
    hf_runner,
    model: str,
    backend: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    with monkeypatch.context() as m:
        m.setenv("VLLM_ATTENTION_BACKEND", backend)

        prompt = "The following numbers of the sequence " + ", ".join(
            str(i) for i in range(1024)) + " are:"
        example_prompts = [prompt]
        with hf_runner(model) as hf_model:
            hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

        with VllmRunner(model,
                        max_model_len=8192,
                        enforce_eager=enforce_eager,
                        gpu_memory_utilization=0.7) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                      max_tokens)

        check_outputs_equal(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )
