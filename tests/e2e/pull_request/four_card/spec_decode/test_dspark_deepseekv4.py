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
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py`.
"""

import os
from unittest.mock import patch

import pytest
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

MODELS = ["UploadWeight/DeepSeek-V4-Flash-DSpark-w4a8-test"]
ACCEPTANCE_LENGTH_RTOL = 0.05
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Confidence-based dynamic verify-length; keep in sync with
# tests/e2e/pull_request/one_card/spec_decode/test_dynamic.py (dspark).
DSPARK_DYNAMIC_SPEC_CONFIG = {
    "method": "dspark",
    "method_params": {
        "initial_verify_budget_per_req": 3,
        "budget_update_interval": 1,
        "budget_threshold": 0.7,
    },
}


@pytest.mark.skip(reason="Temporarily brought offline. The cases of speculative decoding need to be rectified later.")
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize(
    ("expected_acceptance_length", "num_speculative_tokens", "additional_config"),
    [
        pytest.param(4.14, 5, {"enable_dsa_cp": False}, id="dspark"),
        pytest.param(4.80, 7, {"enable_dsa_cp": True}, id="dsa-cp-dspark"),
        pytest.param(
            4.13,
            5,
            {
                "enable_flashcomm1": False,
                "enable_dsa_cp": False,
                "dynamic_spec_config": DSPARK_DYNAMIC_SPEC_CONFIG,
            },
            id="dspark-dynamic",
        ),
    ],
)
@patch.dict(os.environ, {"HCCL_BUFFSIZE": "1024"})
def test_deepseek_v4_dspark_acceptance_tp4(
    model_name,
    expected_acceptance_length,
    num_speculative_tokens,
    additional_config,
):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Explain why the sky appears blue during the day.",
        "Write a short poem about a quiet winter morning.",
        "Describe how photosynthesis works in simple terms.",
        "What is 17 multiplied by 23? Show the calculation.",
        "Summarize the main causes of the Industrial Revolution.",
        "Give three practical tips for learning a new language.",
        "Translate 'Knowledge is power' into French.",
        "Compare renewable energy with fossil fuels.",
        "Create a Python function that reverses a string.",
        "Why do leaves change color in autumn?",
        "Tell a short story about an astronaut visiting Mars.",
        "Explain the difference between weather and climate.",
        "List the first ten prime numbers.",
        "How does a computer store information in binary?",
        "Suggest a healthy breakfast using common ingredients.",
        "What are the benefits of regular physical exercise?",
    ]

    max_tokens = 1024

    with VllmRunner(
        model_name,
        tensor_parallel_size=4,
        max_model_len=4096,
        enable_expert_parallel=True,
        disable_log_stats=False,
        max_num_seqs=len(example_prompts),
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": num_speculative_tokens,
            "enforce_eager": True,
        },
        compilation_config=CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY"),
        additional_config=additional_config,
    ) as spec_vllm_model:
        _ = spec_vllm_model.generate_greedy(example_prompts, max_tokens)
        metrics = spec_vllm_model.model.get_metrics()

    num_drafts = 0
    num_accepted_tokens = 0
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value

    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    relative_error = abs(acceptance_length - expected_acceptance_length) / expected_acceptance_length

    assert relative_error <= ACCEPTANCE_LENGTH_RTOL, (
        f"acceptance_length {acceptance_length:.3f} does not match expected "
        f"{expected_acceptance_length:.3f} within {ACCEPTANCE_LENGTH_RTOL:.0%} "
        f"relative tolerance (num_drafts={num_drafts}, "
        f"num_accepted_tokens={num_accepted_tokens})"
    )
    cleanup_dist_env_and_memory()
