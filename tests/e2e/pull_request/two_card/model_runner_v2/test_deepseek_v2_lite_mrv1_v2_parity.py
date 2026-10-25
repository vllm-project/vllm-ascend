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

"""Compare DeepSeek V2 across Model Runner V1 and V2.

Greedy tokens and logprobs are exactness/parity gates. Fixed-seed sampled-token
agreement is recorded as diagnostic data, not asserted exactly: MRV1 samples
with exponential-race RNG, while MRV2 uses hash-based Gumbel noise, so the two
implementations do not share an RNG stream.
"""

import json
import os
import platform
import re
import subprocess
import threading
import time
from importlib.metadata import version
from pathlib import Path

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory, wait_until_npu_memory_free

MODEL = os.environ.get("DEEPSEEK_V2_MODEL", "vllm-ascend/DeepSeek-V2-Lite-W8A8")
RESULT_PATH = Path(os.environ.get("DEEPSEEK_V2_MRV_RESULT", "/tmp/deepseek_v2_mrv1_v2_parity.json"))
TP2_RESULT_PATH = Path(os.environ.get("DEEPSEEK_V2_MRV_TP2_RESULT", "/tmp/deepseek_v2_mrv1_v2_parity_tp2.json"))

PROMPTS = [
    "The capital of France is",
    "用三句话解释什么是张量并行。",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "A long-context test: " + "alpha beta gamma delta " * 240 + "\nSummarize the repeated pattern in one sentence.",
]

WARMUP_PROMPT = "Warm up the model runner without reusing a parity prompt."

_HBM_USAGE_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


class _NpuMemoryMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_hbm_mb = 0

    def _sample(self) -> int:
        output = subprocess.check_output(["npu-smi", "info"], text=True, stderr=subprocess.DEVNULL)
        matches = []
        for line in output.splitlines():
            if re.match(r"^\|\s*\d+\s+\d+\s+\|", line):
                values = _HBM_USAGE_RE.findall(line)
                if values:
                    matches.append(int(values[-1][0]))
        return max(matches, default=0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_hbm_mb = max(self.peak_hbm_mb, self._sample())
            except (OSError, subprocess.SubprocessError):
                return
            self._stop.wait(0.5)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join(timeout=2)


def _json_default(value):
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def _greedy_logprob_deltas(mrv1, mrv2):
    deltas = []
    for tokens1, tokens2, logprobs1, logprobs2 in zip(
        mrv1["greedy_tokens"],
        mrv2["greedy_tokens"],
        mrv1["greedy_logprobs"],
        mrv2["greedy_logprobs"],
        strict=True,
    ):
        for token1, token2, position1, position2 in zip(tokens1, tokens2, logprobs1, logprobs2, strict=True):
            if token1 != token2:
                continue
            entry1 = position1.get(token1, position1.get(str(token1)))
            entry2 = position2.get(token2, position2.get(str(token2)))
            value1 = entry1.logprob if hasattr(entry1, "logprob") else entry1["logprob"]
            value2 = entry2.logprob if hasattr(entry2, "logprob") else entry2["logprob"]
            deltas.append(abs(value1 - value2))
    return max(deltas, default=0.0), sum(deltas) / len(deltas) if deltas else 0.0


def _measure(
    model_runner: str,
    *,
    prompts: list[str] = PROMPTS,
    max_tokens: int = 32,
    max_model_len: int = 4096,
    max_num_batched_tokens: int = 4096,
    tensor_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    gpu_memory_utilization: float = 0.75,
):
    greedy = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        logprobs=5,
        seed=1234,
    )
    sampled = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.8,
        top_p=0.95,
        top_k=20,
        seed=1234,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_USE_V2_MODEL_RUNNER", model_runner)
        mp.setenv("OMP_NUM_THREADS", "1")
        mp.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        with (
            _NpuMemoryMonitor() as memory_monitor,
            VllmRunner(
                MODEL,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                enforce_eager=True,
                tensor_parallel_size=tensor_parallel_size,
                enable_expert_parallel=enable_expert_parallel,
                gpu_memory_utilization=gpu_memory_utilization,
            ) as runner,
        ):
            runner.generate([WARMUP_PROMPT], SamplingParams(max_tokens=1, temperature=0.0))
            started = time.perf_counter()
            greedy_outputs = runner.generate_w_logprobs(prompts, greedy)
            sampled_outputs = runner.generate(prompts, sampled)
            elapsed = time.perf_counter() - started

    cleanup_dist_env_and_memory()
    time.sleep(5)

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
        "hbm_peak_mb": memory_monitor.peak_hbm_mb,
        "tensor_parallel_size": tensor_parallel_size,
        "expert_parallel": enable_expert_parallel,
        "gpu_memory_utilization": gpu_memory_utilization,
        "timing_scope": "generation_after_warmup",
    }


@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v2_lite_mrv1_mrv2_parity_and_performance():
    mrv1 = _measure("0")
    mrv2 = _measure("1")

    sampled_total = sum(len(tokens) for tokens in mrv1["sampled_tokens"])
    sampled_matches = sum(
        left == right
        for left_tokens, right_tokens in zip(mrv1["sampled_tokens"], mrv2["sampled_tokens"], strict=True)
        for left, right in zip(left_tokens, right_tokens, strict=True)
    )
    sampled_match_ratio = sampled_matches / sampled_total
    greedy_logprob_max_delta, greedy_logprob_mean_delta = _greedy_logprob_deltas(mrv1, mrv2)

    mrv1_rates = mrv1["output_tokens"] / mrv1["elapsed_s"]
    mrv2_rates = mrv2["output_tokens"] / mrv2["elapsed_s"]

    result = {
        "model": MODEL,
        "software": {
            "vllm": version("vllm"),
            "vllm_ascend": version("vllm-ascend"),
            "torch": torch.__version__,
            "torch_npu": version("torch-npu"),
            "platform": platform.platform(),
        },
        "runtime": {
            "model_runner_v1": "VLLM_USE_V2_MODEL_RUNNER=0",
            "model_runner_v2": "VLLM_USE_V2_MODEL_RUNNER=1",
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "expert_parallel": False,
            "enforce_eager": True,
        },
        "prompts": PROMPTS,
        "mrv1": mrv1,
        "mrv2": mrv2,
        "mrv1_tokens_per_s": mrv1_rates,
        "mrv2_tokens_per_s": mrv2_rates,
        "sampled_token_match_ratio": sampled_match_ratio,
        "greedy_logprob_max_abs_delta": greedy_logprob_max_delta,
        "greedy_logprob_mean_abs_delta": greedy_logprob_mean_delta,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))

    assert all(mrv1["greedy_texts"])
    assert all(mrv2["greedy_texts"])
    assert all(mrv1["sampled_texts"])
    assert all(mrv2["sampled_texts"])
    assert mrv1["greedy_tokens"] == mrv2["greedy_tokens"], "MRV1/MRV2 greedy token outputs diverged"
    assert greedy_logprob_max_delta <= 1e-4, "MRV1/MRV2 greedy logprobs diverged"
    assert sampled_total > 0
    assert mrv2["elapsed_s"] <= mrv1["elapsed_s"] * 1.10, "MRV2 regressed by more than 10%"
    assert mrv2_rates >= mrv1_rates * 0.90, "MRV2 throughput regressed by more than 10%"


@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v2_lite_mrv1_mrv2_tp2_ep_parity_and_performance():
    prompts = PROMPTS[:3]
    mrv1 = _measure(
        "0",
        prompts=prompts,
        max_tokens=16,
        max_model_len=1024,
        max_num_batched_tokens=256,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.45,
    )
    mrv2 = _measure(
        "1",
        prompts=prompts,
        max_tokens=16,
        max_model_len=1024,
        max_num_batched_tokens=256,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.45,
    )

    sampled_total = sum(len(tokens) for tokens in mrv1["sampled_tokens"])
    sampled_matches = sum(
        left == right
        for left_tokens, right_tokens in zip(mrv1["sampled_tokens"], mrv2["sampled_tokens"], strict=True)
        for left, right in zip(left_tokens, right_tokens, strict=True)
    )
    sampled_match_ratio = sampled_matches / sampled_total
    greedy_logprob_max_delta, greedy_logprob_mean_delta = _greedy_logprob_deltas(mrv1, mrv2)
    mrv1_rates = mrv1["output_tokens"] / mrv1["elapsed_s"]
    mrv2_rates = mrv2["output_tokens"] / mrv2["elapsed_s"]

    result = {
        "model": MODEL,
        "runtime": {
            "model_runner_v1": "VLLM_USE_V2_MODEL_RUNNER=0",
            "model_runner_v2": "VLLM_USE_V2_MODEL_RUNNER=1",
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
            "expert_parallel": True,
            "enforce_eager": True,
        },
        "prompts": prompts,
        "mrv1": mrv1,
        "mrv2": mrv2,
        "mrv1_tokens_per_s": mrv1_rates,
        "mrv2_tokens_per_s": mrv2_rates,
        "sampled_token_match_ratio": sampled_match_ratio,
        "greedy_logprob_max_abs_delta": greedy_logprob_max_delta,
        "greedy_logprob_mean_abs_delta": greedy_logprob_mean_delta,
    }
    TP2_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TP2_RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))

    assert mrv1["greedy_tokens"] == mrv2["greedy_tokens"], "MRV1/MRV2 TP2 greedy token outputs diverged"
    assert greedy_logprob_max_delta <= 1e-4, "MRV1/MRV2 TP2 greedy logprobs diverged"
    assert sampled_total > 0
    assert mrv2["elapsed_s"] <= mrv1["elapsed_s"] * 1.10, "MRV2 TP2 regressed by more than 10%"
    assert mrv2_rates >= mrv1_rates * 0.90, "MRV2 TP2 throughput regressed by more than 10%"
