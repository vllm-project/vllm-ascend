#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Validate DeepSeek V2 Lite Model Runner V1/V2 greedy parity."""

import os

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory, wait_until_npu_memory_free

MODEL = os.environ.get("DEEPSEEK_V2_MODEL", "vllm-ascend/DeepSeek-V2-Lite-W8A8")
WARMUP_PROMPT = "Warm up the model runner without reusing a parity prompt."
PROMPTS = [
    "The capital of France is",
    "用三句话解释什么是张量并行。",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "A long-context test: " + "alpha beta gamma delta " * 240 + "\nSummarize the repeated pattern in one sentence.",
]


def _logprob_value(entry):
    return entry.logprob if hasattr(entry, "logprob") else entry["logprob"]


def _max_greedy_logprob_delta(left, right):
    deltas = []
    for left_tokens, right_tokens, left_probs, right_probs in zip(
        left["tokens"], right["tokens"], left["logprobs"], right["logprobs"], strict=True
    ):
        assert left_tokens == right_tokens, "MRV1/MRV2 greedy token outputs diverged"
        for token, left_pos, right_pos in zip(left_tokens, left_probs, right_probs, strict=True):
            left_entry = left_pos.get(token, left_pos.get(str(token)))
            right_entry = right_pos.get(token, right_pos.get(str(token)))
            assert left_entry is not None and right_entry is not None, f"missing logprob for token {token}"
            deltas.append(abs(_logprob_value(left_entry) - _logprob_value(right_entry)))
    return max(deltas, default=0.0)


def _measure(model_runner, *, prompts, max_tokens, max_model_len, max_num_batched_tokens,
             tensor_parallel_size, enable_expert_parallel, gpu_memory_utilization):
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0, logprobs=5, seed=1234)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_USE_V2_MODEL_RUNNER", model_runner)
        mp.setenv("OMP_NUM_THREADS", "1")
        mp.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        with VllmRunner(
            MODEL,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            enforce_eager=True,
            tensor_parallel_size=tensor_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            gpu_memory_utilization=gpu_memory_utilization,
        ) as runner:
            runner.generate([WARMUP_PROMPT], SamplingParams(max_tokens=1, temperature=0.0))
            outputs = runner.generate_w_logprobs(prompts, sampling)
    cleanup_dist_env_and_memory()
    return {"tokens": [item[0] for item in outputs], "logprobs": [item[2] for item in outputs]}


def _assert_parity(mrv1, mrv2):
    assert mrv1["tokens"] == mrv2["tokens"], "MRV1/MRV2 greedy token outputs diverged"
    assert _max_greedy_logprob_delta(mrv1, mrv2) <= 1e-4


@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v2_lite_mrv1_mrv2_parity():
    mrv1 = _measure("0", prompts=PROMPTS, max_tokens=32, max_model_len=4096,
                    max_num_batched_tokens=4096, tensor_parallel_size=1,
                    enable_expert_parallel=False, gpu_memory_utilization=0.75)
    mrv2 = _measure("1", prompts=PROMPTS, max_tokens=32, max_model_len=4096,
                    max_num_batched_tokens=4096, tensor_parallel_size=1,
                    enable_expert_parallel=False, gpu_memory_utilization=0.75)
    _assert_parity(mrv1, mrv2)


@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v2_lite_mrv1_mrv2_tp2_ep_parity():
    mrv1 = _measure("0", prompts=PROMPTS, max_tokens=16, max_model_len=1024,
                    max_num_batched_tokens=256, tensor_parallel_size=2,
                    enable_expert_parallel=True, gpu_memory_utilization=0.45)
    mrv2 = _measure("1", prompts=PROMPTS, max_tokens=16, max_model_len=1024,
                    max_num_batched_tokens=256, tensor_parallel_size=2,
                    enable_expert_parallel=True, gpu_memory_utilization=0.45)
    _assert_parity(mrv1, mrv2)
