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
"""Check Qwen3-Next MTP acceptance and RecoverSSM generation.

Run `pytest tests/e2e/pull_request/four_card/spec_decode/test_mtp_qwen3_next.py`.
"""

import os

import pytest
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory
from tests.e2e.pull_request.utils import assert_valid_generation_outputs

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = ["Qwen/Qwen3-Next-80B-A3B-Instruct"]


def _read_mtp_counters(metrics):
    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * 3
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                num_accepted_tokens_per_pos[pos] += metric.values[pos]
    return num_drafts, num_accepted_tokens_per_pos


def _run_mtp_case(model_name, use_replayssm, example_prompts, max_tokens):
    with VllmRunner(
        model_name,
        dtype="bfloat16",
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        distributed_executor_backend="mp",
        disable_log_stats=False,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_ssm_cache_dtype="float32",
        use_replayssm=use_replayssm,
        speculative_config={
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": 3,
        },
        compilation_config=CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[20]),
    ) as spec_vllm_model:
        model = spec_vllm_model.model
        vocab_size = model.llm_engine.model_config.get_vocab_size()
        before = _read_mtp_counters(model.get_metrics())
        outputs = model.generate(example_prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens))
        after = _read_mtp_counters(model.get_metrics())

    assert_valid_generation_outputs(outputs, len(example_prompts), vocab_size)
    num_drafts = after[0] - before[0]
    num_accepted_tokens_per_pos = [after_count - before_count for before_count, after_count in zip(before[1], after[1])]
    return num_drafts, num_accepted_tokens_per_pos


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("use_replayssm", [False, True], ids=["replayssm_off", "replayssm_on"])
def test_qwen3_next_mtp_acceptance_tp4(model_name, use_replayssm):
    golden = [0.85, 0.46, 0.19]

    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    max_tokens = 1024

    num_drafts, num_accepted_tokens_per_pos = _run_mtp_case(model_name, use_replayssm, example_prompts, max_tokens)
    assert num_drafts > 0, f"{model_name}, use_replayssm={use_replayssm}: no draft activity"
    assert sum(num_accepted_tokens_per_pos) > 0, f"{model_name}, use_replayssm={use_replayssm}: no accepted drafts"

    if not use_replayssm:
        acceptance_per_pos = [num_accepted_tokens / num_drafts for num_accepted_tokens in num_accepted_tokens_per_pos]
        match = all((a >= b) or (b - a < 0.06) for a, b in zip(acceptance_per_pos, golden))
        assert match, (
            f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden} (num_drafts={num_drafts})"
        )

    cleanup_dist_env_and_memory()
