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
#
"""Qwen3-8B DSpark: adaptive verification combined with batch-sharded sampling.

Both features trim the sampler inputs (adaptive verification on device, batch
sharding on the host). This e2e only checks that the two features can be enabled
together and that a full decode-graph run finishes without errors, which matches
the coverage level of the other ``enable_adaptive_verification=True`` e2e cases.

Run "pytest tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_8b_dspark_batch_sharded.py".
"""

import os
from unittest.mock import patch

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_8b_block7"
NUM_SPECULATIVE_TOKENS = 7
MAX_TOKENS = 32
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@patch.dict(
    os.environ,
    {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_8b_dspark_adaptive_verification_batch_sharded() -> None:
    # Batch-sharded sampling and adaptive verification must be able to run together.
    with VllmRunner(
        TARGET_MODEL,
        tensor_parallel_size=2,
        max_model_len=1024,
        max_num_seqs=len(PROMPTS),
        enforce_eager=False,
        async_scheduling=True,
        enable_prefix_caching=False,
        # Full decode graphs must keep working with variable-length verification.
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 16, 32]},
        enable_batch_sharded_sampling=True,
        speculative_config={
            "model": DRAFT_MODEL,
            "method": "dspark",
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "enable_adaptive_verification": True,
        },
    ) as runner:
        runner.model.generate(PROMPTS, SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0))
