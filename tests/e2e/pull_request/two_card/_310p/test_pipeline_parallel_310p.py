# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from unittest.mock import patch

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

SMALL_MODEL = "Qwen/Qwen3-0.6B"


@patch.dict(os.environ, {"OMP_NUM_THREADS": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_pipeline_parallel_matches_single_stage():
    prompts = [
        "Complete the sequence: 2, 4, 6, 8,",
        "Name three moons in the Solar System.",
        "The quick brown fox jumps over the lazy dog. " * 36 + "Continue the story:",
    ]
    settings = dict(
        dtype="float16",
        enforce_eager=True,
        distributed_executor_backend="mp",
        max_model_len=768,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        gpu_memory_utilization=0.35,
        enable_prefix_caching=False,
    )
    with VllmRunner(SMALL_MODEL, pipeline_parallel_size=1, **settings) as baseline:
        expected = baseline.generate_greedy(prompts, 16)
    with VllmRunner(SMALL_MODEL, pipeline_parallel_size=2, **settings) as candidate:
        actual = candidate.generate_greedy(prompts, 16)
    assert [tokens for tokens, _ in actual] == [tokens for tokens, _ in expected]
