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
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
"""Validate GLM-5.2 generation with DSpark and MTP speculative decoding.

Run pytest tests/e2e/pull_request/eight_card/test_glm5_2.py.
"""

import os
from unittest.mock import patch

import pytest
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

MAIN_MODEL = "Eco-Tech/GLM-5.2-w4a8"
SPECULATOR_MODEL = "RedHatAI/GLM-5.2-speculator.dspark"
DSPARK_NUM_SPECULATIVE_TOKENS = 7
MTP_NUM_SPECULATIVE_TOKENS = 3
DSPARK_EXPECTED_ACCEPTANCE_LENGTH = 3.57
MTP_EXPECTED_ACCEPTANCE_LENGTH = 2.94
ACCEPTANCE_LENGTH_RTOL = 0.05

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _run_speculative_decoding(
    speculative_config: dict[str, object],
    compilation_config: CompilationConfig,
    expected_acceptance_length: float,
) -> float:
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

    with VllmRunner(
        MAIN_MODEL,
        quantization="ascend",
        tensor_parallel_size=8,
        max_model_len=8192,
        max_num_seqs=len(example_prompts),
        enable_expert_parallel=True,
        disable_log_stats=False,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens=1024)
        metrics = vllm_model.model.get_metrics()

    assert len(outputs) == len(example_prompts)
    assert all(output_ids and output_text for output_ids, output_text in outputs)

    num_drafts = 0
    num_accepted_tokens = 0
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value

    assert num_drafts > 0, "Speculative decoding did not generate any draft tokens"
    acceptance_length = 1 + num_accepted_tokens / num_drafts
    relative_error = abs(acceptance_length - expected_acceptance_length) / expected_acceptance_length
    assert relative_error <= ACCEPTANCE_LENGTH_RTOL, (
        f"acceptance_length {acceptance_length:.3f} does not match expected "
        f"{expected_acceptance_length:.3f} within {ACCEPTANCE_LENGTH_RTOL:.0%} "
        f"relative tolerance (num_drafts={num_drafts}, num_accepted_tokens={num_accepted_tokens})"
    )

    cleanup_dist_env_and_memory()
    return acceptance_length


@pytest.mark.e2e_model(MAIN_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="spec_decode,aclgraph",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "HCCL_BUFFSIZE": "512",
        "HCCL_OP_EXPANSION_MODE": "AIV",
    },
)
def test_glm_5_2_dspark_acceptance_tp8() -> None:
    _run_speculative_decoding(
        speculative_config={
            "method": "dspark",
            "model": SPECULATOR_MODEL,
            "num_speculative_tokens": DSPARK_NUM_SPECULATIVE_TOKENS,
            "enforce_eager": True,
        },
        compilation_config=CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY"),
        expected_acceptance_length=DSPARK_EXPECTED_ACCEPTANCE_LENGTH,
    )


@pytest.mark.e2e_model(MAIN_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,aclgraph",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "HCCL_BUFFSIZE": "512",
        "HCCL_OP_EXPANSION_MODE": "AIV",
    },
)
def test_glm_5_2_mtp_acceptance_tp8() -> None:
    _run_speculative_decoding(
        speculative_config={
            "method": "deepseek_mtp",
            "num_speculative_tokens": MTP_NUM_SPECULATIVE_TOKENS,
            "enforce_eager": True,
        },
        compilation_config=CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY"),
        expected_acceptance_length=MTP_EXPECTED_ACCEPTANCE_LENGTH,
    )
