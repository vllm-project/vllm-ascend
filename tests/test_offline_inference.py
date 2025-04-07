#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/tests/basic_correctness/test_basic_correctness.py
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

Run `pytest tests/test_offline_inference.py`.
"""
import os

import pytest
import vllm  # noqa: F401

import vllm_ascend  # noqa: F401
from tests.conftest import VllmRunner

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]
os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"

TARGET_TEST_SUITE = os.environ.get("TARGET_TEST_SUITE", "L4")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half", "float16"])
@pytest.mark.parametrize("max_tokens", [5])
def test_models(
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    os.environ["VLLM_ATTENTION_BACKEND"] = "ASCEND"

    # 5042 tokens for gemma2
    # gemma2 has alternating sliding window size of 4096
    # we need a prompt with more than 4096 tokens to test the sliding window
    prompt = "The following numbers of the sequence " + ", ".join(
        str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    with VllmRunner(model,
                    max_model_len=8192,
                    dtype=dtype,
                    enforce_eager=False,
                    gpu_memory_utilization=0.7) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
