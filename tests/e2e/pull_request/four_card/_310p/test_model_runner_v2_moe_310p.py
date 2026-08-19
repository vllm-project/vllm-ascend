# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from unittest.mock import patch

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.utils_310p import hybrid_runner_kwargs

# First-release Model Runner V2 E2E coverage for MoE models on 310P.
# Mirrors the V1 tests in test_moe_model_310p.py with VLLM_USE_V2_MODEL_RUNNER=1.


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_qwen3_moe_tp2_eager():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        "Qwen/Qwen3-30B-A3B",
        tensor_parallel_size=2,
        enforce_eager=True,
        dtype="float16",
        max_model_len=8192,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        enable_prefix_caching=False,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        follow_up_outputs = vllm_model.generate_greedy(["Count to two."], max_tokens=2)

    assert outputs[0][0]
    assert follow_up_outputs[0][0]


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_qwen3_moe_tp2_aclgraph():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        "Qwen/Qwen3-30B-A3B",
        tensor_parallel_size=2,
        enforce_eager=False,
        dtype="float16",
        max_model_len=8192,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        enable_prefix_caching=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        follow_up_outputs = vllm_model.generate_greedy(["Count to two."], max_tokens=2)

    assert outputs[0][0]
    assert follow_up_outputs[0][0]


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_qwen3_moe_w8a8_dynamic_tp2_aclgraph():
    # This checkpoint's expert quantization description must be
    # W8A8_DYNAMIC. Static W8A8/W8A8SC expert descriptions are rejected.
    with VllmRunner(
        "vllm-ascend/Qwen3-30B-A3B-W8A8",
        tensor_parallel_size=2,
        enforce_eager=False,
        dtype="float16",
        quantization="ascend",
        max_model_len=8192,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        enable_prefix_caching=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(["Hello, my name is"], max_tokens=5)
        follow_up_outputs = vllm_model.generate_greedy(["Count to two."], max_tokens=2)

    assert outputs[0][0]
    assert follow_up_outputs[0][0]


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_qwen3_5_moe_tp4_eager():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        "Qwen/Qwen3.5-35B-A3B",
        tensor_parallel_size=4,
        enforce_eager=True,
        dtype="float16",
        max_model_len=8192,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        enable_prefix_caching=False,
        **hybrid_runner_kwargs("Qwen/Qwen3.5-35B-A3B"),
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        follow_up_outputs = vllm_model.generate_greedy(["Count to two."], max_tokens=2)

    assert outputs[0][0]
    assert follow_up_outputs[0][0]


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_qwen3_5_moe_tp4_aclgraph():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        "Qwen/Qwen3.5-35B-A3B",
        tensor_parallel_size=4,
        enforce_eager=False,
        dtype="float16",
        max_model_len=8192,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        enable_prefix_caching=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
        **hybrid_runner_kwargs("Qwen/Qwen3.5-35B-A3B"),
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        follow_up_outputs = vllm_model.generate_greedy(["Count to two."], max_tokens=2)

    assert outputs[0][0]
    assert follow_up_outputs[0][0]
