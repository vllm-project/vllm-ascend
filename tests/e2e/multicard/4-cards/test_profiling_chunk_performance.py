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
#
"""Performance guard for profiling-based dynamic chunk sizing (PP scenario).

Measures Time-To-First-Token (TTFT) on 64k-token prefill requests with
profiling_chunk_config enabled.  The test runs against
DeepSeek-V2-Lite-Chat served with PP=2, TP=2 (4 NPU cards total).

Test flow:
  1. Create an LLM engine with profiling_chunk_config enabled.
  2. Run NUM_WARMUP sequential requests (64k tokens, max_tokens=1) to warm
     up both the NPU and the profiling predictor.
  3. Run NUM_TEST sequential requests, recording TTFT for each.
  4. Assert that the median TTFT does not exceed BASELINE_TTFT_S seconds.
"""

import os
import statistics
import time

import pytest

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"

# ~64k tokens: DeepSeek-V2 tokenizer averages ~3.5 chars/token for English;
# 64000 * 3.5 ≈ 224000 chars.  We use a plain repeated word to keep it simple.
_WORD = "hello "
INPUT_64K_TOKENS = _WORD * (224_000 // len(_WORD))

NUM_WARMUP = 5
NUM_TEST = 5

# NOTE: Any changes to this baseline must be approved by team members.
# Measured on DeepSeek-V2-Lite-Chat, PP=2, TP=2, 64k prefill, profiling_chunk enabled.
BASELINE_TTFT_S = 5.0


def test_profiling_chunk_ttft_performance() -> None:
    with VllmRunner(
        MODEL,
        max_model_len=70000,
        dtype="auto",
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        block_size=128,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=65536,
        distributed_executor_backend="mp",
        additional_config={"profiling_chunk_config": {"enabled": True}},
    ) as vllm_model:
        # With max_tokens=1, total latency ≈ prefill time ≈ TTFT
        prompts = [INPUT_64K_TOKENS]

        # ── Warmup ──────────────────────────────────────────────────────────
        for _ in range(NUM_WARMUP):
            vllm_model.generate_greedy(prompts, max_tokens=1)

        # ── Measurement ─────────────────────────────────────────────────────
        ttfts: list[float] = []
        for _ in range(NUM_TEST):
            start = time.perf_counter()
            vllm_model.generate_greedy(prompts, max_tokens=1)
            ttfts.append(time.perf_counter() - start)

        median_ttft = statistics.median(ttfts)
        ttft_str = ", ".join(f"{t:.2f}s" for t in ttfts)
        print(
            f"\n[profiling_chunk perf] TTFT per request: [{ttft_str}]"
            f"\n[profiling_chunk perf] Median TTFT: {median_ttft:.2f}s  "
            f"(baseline: {BASELINE_TTFT_S}s)"
        )

        assert median_ttft <= BASELINE_TTFT_S, (
            f"TTFT performance regression: median TTFT {median_ttft:.2f}s "
            f"exceeds baseline {BASELINE_TTFT_S}s. "
            f"Individual TTFTs: [{ttft_str}]"
        )
