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
"""MC2 fused matmul-collective ops for dense SP (Qwen3.5 dense family).

The fused AllGather+Matmul / Matmul+ReduceScatter custom ops live in
``vllm_ascend.models.common.ops.sequence_parallel`` and are installed on
eligible projections when dense SP (FlashComm switch) is active.  The fusion
threshold (128 total tokens) directs prefill and mixed batches to the fused
kernel and pure decode to the unfused fallback; both paths must produce
identical numerics.

These tests exercise the fused path with prefill-sized batches (token count
well above 128) under both eager and FULL_DECODE_ONLY graph modes, validating
numerical consistency against the plain eager baseline without FlashComm.
"""

import os

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.utils import compare_logprobs

MODEL = os.environ.get("DENSE_SP_TEST_MODEL", "Qwen/Qwen3.8-27B")

# Prefill-sized prompts whose token counts exceed the 128-token fusion
# threshold, ensuring the fused matmul-collective kernel is exercised.
PREFILL_PROMPTS = [
    "Explain the concept of sequence parallelism in distributed training. "
    "Cover the motivation, the communication pattern, and how it interacts "
    "with tensor parallelism. Provide a concrete example with code snippets.",
    "Write a detailed essay about the history of computing, from the earliest "
    "mechanical calculators to modern AI accelerators. Include key milestones, "
    "important figures, and how each generation of hardware influenced software "
    "development practices.",
    "Describe the architecture of a transformer model in detail. Explain the "
    "attention mechanism, positional encoding, layer normalization, and the "
    "feed-forward network. Discuss how these components are composed in both "
    "encoder and decoder blocks.",
    "Provide a comprehensive guide to optimizing large language model inference. "
    "Cover quantization, KV cache management, batching strategies, speculative "
    "decoding, and hardware-specific optimizations for NPU accelerators.",
]

RUNNER_KWARGS = {
    "model_name": MODEL,
    "max_model_len": 4096,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 2048,
    "tensor_parallel_size": 2,
    "additional_config": {
        "enable_flashcomm1": True,
    },
}


@wait_until_npu_memory_free(0.7)
def test_dense_sp_mc2_fusion_tp2_eager(monkeypatch) -> None:
    """MC2 fusion in eager mode: prefill-sized prompts trigger the fused
    matmul-collective kernel; output must match the unfused eager baseline."""
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)

    compare_logprobs(
        runner_kwargs={
            **RUNNER_KWARGS,
            "enforce_eager": True,
        },
        prompts=PREFILL_PROMPTS,
    )


@wait_until_npu_memory_free(0.7)
def test_dense_sp_mc2_fusion_tp2_graph(monkeypatch) -> None:
    """MC2 fusion with FULL_DECODE_ONLY graphs: the fused custom ops must
    compile, warmup, capture and replay identically, keeping numerics
    consistent with the unfused eager baseline."""
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)

    compare_logprobs(
        runner_kwargs={
            **RUNNER_KWARGS,
            "compilation_config": {
                "cudagraph_capture_sizes": [4],
                "cudagraph_mode": "FULL_DECODE_ONLY",
            },
        },
        prompts=PREFILL_PROMPTS,
    )
