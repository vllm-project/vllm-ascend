#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Four-card DP=2/TP=2 coverage for upstream Linear sequence parallelism."""

import os

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free

TEST_MODEL = os.environ.get("SP_TEST_MODEL", "Qwen/Qwen3-30B-A3B")
# A model with real shared experts, covering the decoupled SP/shared-expert
# paths that Qwen3-30B-A3B (no shared experts) cannot exercise.
SHARED_EXPERT_TEST_MODEL = os.environ.get("SP_SHARED_EXPERT_TEST_MODEL", "deepseek-ai/DeepSeek-V2-Lite")


@wait_until_npu_memory_free()
@pytest.mark.parametrize(
    "enable_shared_expert_dp",
    [False, True],
    ids=["sp-only", "sp-with-shared-expert-dp"],
)
def test_sequence_parallel_moe_dp2_tp2_functional(enable_shared_expert_dp: bool) -> None:
    """Verify upstream SP with either independent shared-expert layout."""
    prompts = [
        "The capital of France is",
        "Explain why the sky is blue in one sentence.",
    ]
    with DPVllmRunner(
        TEST_MODEL,
        data_parallel_size=2,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
        additional_config={"enable_shared_expert_dp": enable_shared_expert_dp},
        distributed_executor_backend="mp",
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens=8)

    assert len(outputs) == len(prompts)
    assert all(output[1] for output in outputs)


TEACHER_PAIRS = [
    (
        "The capital of France is",
        " Paris. It is known for the Eiffel Tower, the Louvre Museum, and its cuisine.",
    ),
    (
        "Explain the theory of relativity in one paragraph:",
        " The theory of relativity, developed by Albert Einstein, states that space and time are interwoven.",
    ),
    ("中国的首都是", " 北京。北京是中国的政治、文化和国际交往中心。"),
]


def _teacher_logprobs(
    all2all_backend: str,
    enable_shared_expert_dp: bool,
    model: str = TEST_MODEL,
) -> list[list[dict[int, float]]]:
    with DPVllmRunner(
        model,
        data_parallel_size=2,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        enforce_eager=True,
        all2all_backend=all2all_backend,
        additional_config={"enable_shared_expert_dp": enable_shared_expert_dp},
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    ) as vllm_model:
        outputs = vllm_model.generate_w_logprobs(
            [prompt + continuation for prompt, continuation in TEACHER_PAIRS],
            SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=5),
        )

    return [
        [{token_id: logprob.logprob for token_id, logprob in step.items()} for step in (output[3] or []) if step]
        for output in outputs
    ]


def _assert_logprobs_close(outputs: dict[str, list], baseline: list) -> None:
    for mode, candidate in outputs.items():
        deltas = []
        for pair_idx, (candidate_steps, baseline_steps) in enumerate(zip(candidate, baseline)):
            assert len(candidate_steps) == len(baseline_steps), f"{mode}, pair {pair_idx}: token count mismatch"
            for position, (candidate_dist, baseline_dist) in enumerate(zip(candidate_steps, baseline_steps)):
                for token_id in set(candidate_dist) & set(baseline_dist):
                    deltas.append(
                        (
                            abs(candidate_dist[token_id] - baseline_dist[token_id]),
                            pair_idx,
                            position,
                            token_id,
                        )
                    )

        assert deltas, f"{mode}: no shared top-5 tokens to compare"
        max_delta = max(delta[0] for delta in deltas)
        mean_delta = sum(delta[0] for delta in deltas) / len(deltas)
        assert max_delta < 1.0, f"{mode} distribution corruption: max |delta|={max_delta:.4f}"
        assert mean_delta < 0.15, f"{mode} distribution drift: mean |delta|={mean_delta:.4f}"


@wait_until_npu_memory_free()
def test_sequence_parallel_moe_dp2_tp2_precision() -> None:
    """Validate the SP/shared-expert-DP 2x2 matrix against one baseline."""
    outputs = {
        "shared-expert-dp": _teacher_logprobs("flashinfer_all2allv", True),
        "sequence-parallel": _teacher_logprobs("allgather_reducescatter", False),
        "sequence-parallel-with-shared-expert-dp": _teacher_logprobs("allgather_reducescatter", True),
    }
    baseline = _teacher_logprobs("flashinfer_all2allv", False)

    _assert_logprobs_close(outputs, baseline)


@wait_until_npu_memory_free()
def test_sequence_parallel_moe_shared_expert_dp2_tp2_precision() -> None:
    """Same SP matrix on a model with real shared experts.

    SP-only now runs shared experts with TP-sharded weights (all-gather +
    unpad up front, pad + reduce-scatter on exit), SP+shared-expert-DP keeps
    replicated weights on the SP shard. Both must match the non-SP baseline.
    """
    outputs = {
        "sequence-parallel": _teacher_logprobs("allgather_reducescatter", False, SHARED_EXPERT_TEST_MODEL),
        "sequence-parallel-with-shared-expert-dp": _teacher_logprobs(
            "allgather_reducescatter", True, SHARED_EXPERT_TEST_MODEL
        ),
    }
    baseline = _teacher_logprobs("flashinfer_all2allv", False, SHARED_EXPERT_TEST_MODEL)

    _assert_logprobs_close(outputs, baseline)
