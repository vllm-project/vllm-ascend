# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from unittest.mock import patch

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.model_utils import check_outputs_equal

QWEN3_5_PREFIX_MAMBA_PROMPT = (
    "You are reading a compact synthetic operations ledger. "
    "Use only the rows below when answering the final question.\n"
    + "\n".join(
        f"Row {i}: route R{i:03d} moves cargo from zone {i % 11} to zone {(i * 7) % 13}; priority is {i % 5}."
        for i in range(64)
    )
    + "\n"
)

QWEN3_5_PREFIX_MAMBA_PROMPTS = [
    QWEN3_5_PREFIX_MAMBA_PROMPT + "Question: What route is listed in row 17? Answer briefly.",
    QWEN3_5_PREFIX_MAMBA_PROMPT + "Question: What priority is listed in row 42? Answer briefly.",
]


def _generate_qwen35_mrv2_prefix_mamba_outputs(enable_prefix_caching: bool) -> list[tuple[list[int], str]]:
    outputs: list[tuple[list[int], str]] = []

    if enable_prefix_caching:
        with (
            patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"}),
            VllmRunner(
                "Qwen/Qwen3.5-4B",
                tensor_parallel_size=2,
                enforce_eager=True,
                dtype="float16",
                max_model_len=2048,
                max_num_batched_tokens=2048,
                enable_prefix_caching=True,
                mamba_cache_mode="align",
                mamba_ssm_cache_dtype="float16",
            ) as vllm_model,
        ):
            for prompt in QWEN3_5_PREFIX_MAMBA_PROMPTS:
                outputs.extend(vllm_model.generate_greedy([prompt], max_tokens=8))
    else:
        with (
            patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"}),
            VllmRunner(
                "Qwen/Qwen3.5-4B",
                tensor_parallel_size=2,
                enforce_eager=True,
                dtype="float16",
                max_model_len=2048,
                max_num_batched_tokens=2048,
                enable_prefix_caching=False,
                mamba_ssm_cache_dtype="float16",
            ) as vllm_model,
        ):
            for prompt in QWEN3_5_PREFIX_MAMBA_PROMPTS:
                outputs.extend(vllm_model.generate_greedy([prompt], max_tokens=8))
    return outputs


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen3_dense_mrv2_tp2_aclgraph_fp16():
    with VllmRunner(
        "Qwen/Qwen3-8B",
        tensor_parallel_size=2,
        enforce_eager=False,
        enable_prefix_caching=True,
        dtype="float16",
        max_model_len=16384,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        },
    ) as vllm_model:
        vllm_model.generate_greedy(["Hello, my name is"], max_tokens=5)


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen3_moe_mrv2_tp2_aclgraph_w8a8():
    with VllmRunner(
        "vllm-ascend/Qwen3-30B-A3B-W8A8",
        tensor_parallel_size=2,
        enforce_eager=False,
        enable_prefix_caching=True,
        dtype="float16",
        quantization="ascend",
        max_model_len=16384,
        max_num_batched_tokens=2048,
        max_num_seqs=256,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        },
    ) as vllm_model:
        vllm_model.generate_greedy(["Hello, my name is"], max_tokens=5)


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen35_hybrid_mrv2_tp2_aclgraph_fp16():
    with VllmRunner(
        "Qwen/Qwen3.5-4B",
        tensor_parallel_size=2,
        enforce_eager=False,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        dtype="float16",
        max_model_len=8192,
        mamba_ssm_cache_dtype="float16",
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        },
    ) as vllm_model:
        vllm_model.generate_greedy(["Hello, my name is"], max_tokens=5)


@wait_until_npu_memory_free(0.7)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen35_hybrid_prefix_mamba_cache_mrv2_tp2_fp16():
    """MRv2 Qwen3.5 prefix-cache parity vs MRv1 one-card APC test."""
    prefix_cache_outputs = _generate_qwen35_mrv2_prefix_mamba_outputs(enable_prefix_caching=True)
    no_prefix_cache_outputs = _generate_qwen35_mrv2_prefix_mamba_outputs(enable_prefix_caching=False)

    assert len(prefix_cache_outputs) == len(no_prefix_cache_outputs) == len(QWEN3_5_PREFIX_MAMBA_PROMPTS)
    check_outputs_equal(
        outputs_0_lst=no_prefix_cache_outputs,
        outputs_1_lst=prefix_cache_outputs,
        name_0="no_prefix_cache_outputs",
        name_1="prefix_cache_outputs",
    )
