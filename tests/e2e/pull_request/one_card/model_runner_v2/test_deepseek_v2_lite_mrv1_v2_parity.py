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

import json
import os
import time
from pathlib import Path

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = os.environ.get("DEEPSEEK_V2_MODEL", "/workspace/DeepSeek-V2-Lite-W8A8")
RESULT_PATH = Path(os.environ.get("DEEPSEEK_V2_MRV_RESULT", "/tmp/deepseek_v2_mrv1_v2_parity.json"))

PROMPTS = [
    "The capital of France is",
    "用三句话解释什么是张量并行。",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "A long-context test: " + "alpha beta gamma delta " * 240 + "\nSummarize the repeated pattern in one sentence.",
]


def _measure(model_runner: str):
    greedy = SamplingParams(
        max_tokens=32,
        temperature=0.0,
        logprobs=5,
        seed=1234,
    )
    sampled = SamplingParams(
        max_tokens=32,
        temperature=0.8,
        top_p=0.95,
        top_k=20,
        seed=1234,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_USE_V2_MODEL_RUNNER", model_runner)
        started = time.perf_counter()
        with VllmRunner(
            MODEL,
            max_model_len=4096,
            max_num_batched_tokens=4096,
            enforce_eager=True,
            trust_remote_code=True,
        ) as runner:
            greedy_outputs = runner.generate_w_logprobs(PROMPTS, greedy)
            sampled_outputs = runner.generate(PROMPTS, sampled)
            elapsed = time.perf_counter() - started

    return {
        "model_runner": model_runner,
        "greedy_tokens": [item[0] for item in greedy_outputs],
        "greedy_texts": [item[1] for item in greedy_outputs],
        "greedy_logprobs": [item[2] for item in greedy_outputs],
        "sampled_tokens": [item[0][0] for item in sampled_outputs],
        "sampled_texts": [item[1][0] for item in sampled_outputs],
        "elapsed_s": elapsed,
        "output_tokens": sum(len(item[0]) for item in greedy_outputs)
        + sum(len(item[0][0]) for item in sampled_outputs),
        "hbm_peak_bytes": int(torch.npu.max_memory_allocated()) if torch.npu.is_available() else 0,
    }


@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v2_lite_mrv1_mrv2_parity_and_performance():
    mrv1 = _measure("0")
    mrv2 = _measure("1")

    assert all(mrv1["greedy_texts"])
    assert all(mrv2["greedy_texts"])
    assert all(mrv1["sampled_texts"])
    assert all(mrv2["sampled_texts"])
    assert mrv1["greedy_tokens"] == mrv2["greedy_tokens"], "MRV1/MRV2 greedy token outputs diverged"
    assert mrv1["sampled_tokens"] == mrv2["sampled_tokens"], "MRV1/MRV2 fixed-seed sampling diverged"
    assert mrv2["elapsed_s"] <= mrv1["elapsed_s"] * 1.10, "MRV2 regressed by more than 10%"

    mrv1_rates = mrv1["output_tokens"] / mrv1["elapsed_s"]
    mrv2_rates = mrv2["output_tokens"] / mrv2["elapsed_s"]
    assert mrv2_rates >= mrv1_rates * 0.90, "MRV2 throughput regressed by more than 10%"

    result = {
        "model": MODEL,
        "prompts": PROMPTS,
        "mrv1": mrv1,
        "mrv2": mrv2,
        "mrv1_tokens_per_s": mrv1_rates,
        "mrv2_tokens_per_s": mrv2_rates,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
