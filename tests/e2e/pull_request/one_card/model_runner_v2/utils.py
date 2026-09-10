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
#

from __future__ import annotations

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Keep default cudagraph_mode (FULL_AND_PIECEWISE when unspecified) but pin
# capture sizes. The workload is 4 prompts; leaving sizes unset enumerates
# graphs up to min(max_num_seqs * (1+K), 512) and dominates runtime.
CUDAGRAPH_CAPTURE_SIZES = [4, 8]
FULL_DECODE_ONLY = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": CUDAGRAPH_CAPTURE_SIZES,
}
DEFAULT_PIECEWISE = {"cudagraph_capture_sizes": CUDAGRAPH_CAPTURE_SIZES}
# Matches the 4-prompt batch; also shrinks KV/draft padding vs default 256.
MAX_NUM_SEQS = 8
# Default max_num_batched_tokens=8192 makes torch.compile use range (1, 8192).
MAX_NUM_BATCHED_TOKENS = 256


def calculate_acceptance_per_pos(
    metrics: list,
    num_speculative_tokens: int,
    counter_type: type,
    vector_type: type,
) -> list[float]:
    num_drafts = 0
    accepted_per_pos = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, counter_type)
            num_drafts += metric.value  # type: ignore[attr-defined]
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, vector_type)
            for pos in range(len(metric.values)):  # type: ignore[attr-defined]
                accepted_per_pos[pos] += metric.values[pos]  # type: ignore[attr-defined]
    return [a / num_drafts for a in accepted_per_pos]
