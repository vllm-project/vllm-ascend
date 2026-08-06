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
#
"""DP=2/TP=2 functional coverage for upstream Linear sequence parallelism."""

import os

from vllm import SamplingParams

from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free

TEST_MODEL = os.environ.get("SP_TEST_MODEL", "Qwen/Qwen3-30B-A3B")


@wait_until_npu_memory_free()
def test_sequence_parallel_moe_dp2_tp2_functional() -> None:
    """Verify that DP=2/TP=2 MoE SP serves a deterministic request."""
    prompts = [
        "The capital of France is",
        "Explain why the sky is blue in one sentence.",
    ]
    with DPVllmRunner(
        TEST_MODEL,
        data_parallel_size=2,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
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


def _teacher_logprobs(all2all_backend: str) -> list[list[dict[int, float]]]:
    with DPVllmRunner(
        TEST_MODEL,
        data_parallel_size=2,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        enforce_eager=True,
        all2all_backend=all2all_backend,
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


@wait_until_npu_memory_free()
def test_sequence_parallel_moe_dp2_tp2_precision() -> None:
    """Compare SP and non-SP token distributions under identical teacher forcing."""
    sp_on = _teacher_logprobs("allgather_reducescatter")
    sp_off = _teacher_logprobs("flashinfer_all2allv")

    deltas = []
    for pair_idx, (steps_on, steps_off) in enumerate(zip(sp_on, sp_off)):
        assert len(steps_on) == len(steps_off), f"pair {pair_idx}: token count mismatch"
        for position, (dist_on, dist_off) in enumerate(zip(steps_on, steps_off)):
            for token_id in set(dist_on) & set(dist_off):
                deltas.append((abs(dist_on[token_id] - dist_off[token_id]), pair_idx, position, token_id))

    assert deltas, "no shared top-5 tokens to compare"
    max_delta = max(delta[0] for delta in deltas)
    mean_delta = sum(delta[0] for delta in deltas) / len(deltas)
    assert max_delta < 1.0, f"SP distribution corruption: max |delta|={max_delta:.4f}"
    assert mean_delta < 0.15, f"SP distribution drift: mean |delta|={mean_delta:.4f}"
