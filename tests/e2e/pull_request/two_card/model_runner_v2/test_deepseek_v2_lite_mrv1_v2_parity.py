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

"""Validate DeepSeek V2 Lite MRV1/V2 greedy parity with matched prefill batches."""

import os

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import (
    VllmRunner,
    cleanup_dist_env_and_memory,
    wait_until_npu_memory_free,
)

MODEL = os.environ.get("DEEPSEEK_V2_MODEL", "vllm-ascend/DeepSeek-V2-Lite-W8A8")
WARMUP_PROMPT = "Warm up the model runner without reusing a parity prompt."
PROMPTS = [
    "The capital of France is",
    "用三句话解释什么是张量并行。",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "A long-context test: " + "alpha beta gamma delta " * 240 + "\nSummarize the repeated pattern in one sentence.",
]


def _generate(
    model_runner,
    *,
    prompts,
    max_tokens,
    max_model_len,
    max_num_batched_tokens,
    tensor_parallel_size,
    enable_expert_parallel,
    gpu_memory_utilization,
):
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0, logprobs=5, seed=1234)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_USE_V2_MODEL_RUNNER", model_runner)
        mp.setenv("OMP_NUM_THREADS", "1")
        mp.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        with VllmRunner(
            MODEL,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            async_scheduling=True,
            enforce_eager=True,
            tensor_parallel_size=tensor_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            gpu_memory_utilization=gpu_memory_utilization,
        ) as runner:
            runner.generate([WARMUP_PROMPT], SamplingParams(max_tokens=1, temperature=0.0))
            # Rendering/enqueuing is incremental; keep it outside async execution
            # so both runners start with the complete prefill batch.
            core = runner.model.llm_engine.engine_core
            core.call_utility("pause_scheduler", "keep", False)
            try:
                runner.model.enqueue(prompts, sampling_params=sampling, use_tqdm=False)
            finally:
                core.call_utility("resume_scheduler")
            outputs = runner.model.wait_for_completion(use_tqdm=False)
    cleanup_dist_env_and_memory()
    return {
        "tokens": [list(item.outputs[0].token_ids) for item in outputs],
        "logprobs": [item.outputs[0].logprobs for item in outputs],
    }


@pytest.mark.parametrize("tp_size", [1, 2], ids=["single", "tp2_ep"])
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v2_lite_mrv1_mrv2_parity(tp_size):
    options = dict(
        prompts=PROMPTS if tp_size == 1 else PROMPTS[:3],
        max_tokens=32 if tp_size == 1 else 16,
        max_model_len=4096 if tp_size == 1 else 1024,
        max_num_batched_tokens=4096 if tp_size == 1 else 256,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=tp_size == 2,
        gpu_memory_utilization=0.75 if tp_size == 1 else 0.45,
    )
    mrv1 = _generate("0", **options)
    mrv2 = _generate("1", **options)
    assert len(mrv1["tokens"]) == len(mrv2["tokens"]) == len(options["prompts"])
    assert mrv1["tokens"] == mrv2["tokens"], "MRV1/MRV2 greedy token outputs diverged"
    for tokens, left_probs, right_probs in zip(mrv1["tokens"], mrv1["logprobs"], mrv2["logprobs"], strict=True):
        assert tokens, "Model runner returned empty greedy output"
        for token, left_pos, right_pos in zip(tokens, left_probs, right_probs, strict=True):
            assert token in left_pos and token in right_pos, f"Missing greedy logprob for token {token}"
            left = left_pos[token].logprob
            right = right_pos[token].logprob
            assert left == pytest.approx(right, abs=1e-4, rel=0), "MRV1/MRV2 greedy logprobs diverged"
