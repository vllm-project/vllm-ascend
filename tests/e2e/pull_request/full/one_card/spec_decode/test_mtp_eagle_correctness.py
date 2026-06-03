#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
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
"""Compare the short outputs of HF and vLLM when using greedy sampling."""

from __future__ import annotations

import os

import pytest
from vllm import SamplingParams
from vllm.config import CompilationConfig

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = ["wemaster/deepseek_mtp_main_random_bf16"]


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("num_speculative_tokens", [3])
@pytest.mark.parametrize("cudagraph_mode", ["FULL_DECODE_ONLY"])
@pytest.mark.parametrize("disable_padded_drafter_batch", [False])
def test_deepseek_mtp(
    model_name: str, num_speculative_tokens: int, cudagraph_mode: str, disable_padded_drafter_batch: bool
):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    """
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using mtp speculative decoding.
    """
    with VllmRunner(
        model_name,
        tensor_parallel_size=1,
        max_num_seqs=256,
        gpu_memory_utilization=0.7,
        distributed_executor_backend="mp",
        enable_expert_parallel=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
            "disable_padded_drafter_batch": disable_padded_drafter_batch,
        },
        max_model_len=2000,
        compilation_config=CompilationConfig(
            cudagraph_mode=cudagraph_mode,
            cudagraph_capture_sizes=[20],
        ),
    ) as spec_llm:
        sampling_config = SamplingParams(temperature=0, max_tokens=256, ignore_eos=False)
        spec_llm.generate(example_prompts, sampling_config)

    cleanup_dist_env_and_memory()
    del spec_llm
