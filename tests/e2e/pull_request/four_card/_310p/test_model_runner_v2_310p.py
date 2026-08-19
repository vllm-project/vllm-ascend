# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from unittest.mock import patch

import pytest

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.utils_310p import (
    FULL_DECODE_ONLY_GRAPH,
    hybrid_runner_kwargs,
    run_vl_model_test,
)


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@pytest.mark.parametrize("model", ["Qwen/Qwen3-8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"])
def test_model_runner_v2_tp2_chunked_prefill_aclgraph(model: str) -> None:
    prompts = [("The following ledger contains numbered entries. " * 96) + "Summarize entry one."] * 4
    with VllmRunner(
        model,
        tensor_parallel_size=2,
        dtype="float16",
        max_model_len=4096,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        enable_prefix_caching=False,
        compilation_config=FULL_DECODE_ONLY_GRAPH,
        **hybrid_runner_kwargs(model),
    ) as runner:
        outputs = runner.generate_greedy(prompts, max_tokens=4)
        follow_up_outputs = runner.generate_greedy(["Give one short greeting."], max_tokens=2)

    assert all(output[0] for output in outputs)
    assert follow_up_outputs[0][0]


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@pytest.mark.parametrize(
    "model",
    [
        "vllm-ascend/Qwen3-8B-W8A8",
        "vllm-ascend/Qwen3-8B-W8A8-Dynamic",
    ],
)
def test_model_runner_v2_tp2_quantized_aclgraph(model: str) -> None:
    with VllmRunner(
        model,
        tensor_parallel_size=2,
        dtype="float16",
        quantization="ascend",
        max_model_len=2048,
        enable_prefix_caching=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
    ) as runner:
        outputs = runner.generate_greedy(["Hello, my name is"], max_tokens=4)
        follow_up_outputs = runner.generate_greedy(["Count to two."], max_tokens=2)

    assert outputs[0][0]
    assert follow_up_outputs[0][0]


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_tp2_w8a8sc_aclgraph() -> None:
    with VllmRunner(
        "vllm-ascend/Qwen3-8B-w8a8sc-310-vllm-tp2",
        tensor_parallel_size=2,
        dtype="float16",
        quantization="ascend",
        load_format="sharded_state",
        max_model_len=2048,
        enable_prefix_caching=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
    ) as runner:
        outputs = runner.generate_greedy(["Hello, my name is"], max_tokens=4)
        follow_up_outputs = runner.generate_greedy(["Count to two."], max_tokens=2)

    assert outputs[0][0]
    assert follow_up_outputs[0][0]


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_qwen3_5_27b_tp4_aclgraph() -> None:
    with VllmRunner(
        "Qwen/Qwen3.5-27B",
        tensor_parallel_size=4,
        dtype="float16",
        max_model_len=2048,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        enable_prefix_caching=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
        **hybrid_runner_kwargs("Qwen/Qwen3.5-27B"),
    ) as runner:
        outputs = runner.generate_greedy(["Hello, my name is"], max_tokens=4)
        follow_up_outputs = runner.generate_greedy(["Give one short greeting."], max_tokens=2)

    assert outputs[0][0]
    assert follow_up_outputs[0][0]


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_tp4_w8a8sc_aclgraph() -> None:
    with VllmRunner(
        "vllm-ascend/Qwen3-32B-w8a8sc-310-vllm-tp4",
        tensor_parallel_size=4,
        dtype="float16",
        quantization="ascend",
        load_format="sharded_state",
        max_model_len=2048,
        max_num_seqs=4,
        enable_prefix_caching=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
    ) as runner:
        outputs = runner.generate_greedy(["Hello, my name is"], max_tokens=4)
        follow_up_outputs = runner.generate_greedy(["Count to two."], max_tokens=2)

    assert outputs[0][0]
    assert follow_up_outputs[0][0]


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_qwen3_embedding_8b_tp2_aclgraph() -> None:
    queries = ["What is the capital of China?", "Explain gravity"]
    with VllmRunner(
        "Qwen/Qwen3-Embedding-8B",
        runner="pooling",
        tensor_parallel_size=2,
        dtype="float16",
        max_model_len=512,
        enable_prefix_caching=False,
        cudagraph_capture_sizes=[2],
    ) as runner:
        embeddings = runner.embed(queries)

    assert len(embeddings) == len(queries)
    assert all(embedding for embedding in embeddings)


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@pytest.mark.parametrize("model", ["Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"])
def test_model_runner_v2_qwen3_vl_tp2(model: str) -> None:
    run_vl_model_test(
        model_name=model,
        tensor_parallel_size=2,
        max_tokens=5,
        enable_prefix_caching=False,
    )


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@pytest.mark.parametrize("model", ["Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"])
def test_model_runner_v2_qwen3_vl_tp2_aclgraph(model: str) -> None:
    run_vl_model_test(
        model_name=model,
        tensor_parallel_size=2,
        max_tokens=5,
        enforce_eager=False,
        enable_prefix_caching=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
    )


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_qwen3_5_vl_tp2_aclgraph() -> None:
    run_vl_model_test(
        model_name="Qwen/Qwen3.5-2B",
        tensor_parallel_size=2,
        max_tokens=5,
        enforce_eager=False,
        enable_prefix_caching=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
        **hybrid_runner_kwargs("Qwen/Qwen3.5-2B"),
    )


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_qwen3_vl_8b_w8a8sc_tp2_aclgraph() -> None:
    run_vl_model_test(
        model_name="vllm-ascend/Qwen3-VL-8B-Instruct-w8a8sc-310-vllm-tp2",
        tensor_parallel_size=2,
        max_tokens=5,
        enforce_eager=False,
        enable_prefix_caching=False,
        quantization="ascend",
        load_format="sharded_state",
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
    )
