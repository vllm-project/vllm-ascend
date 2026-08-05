# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from unittest.mock import patch

import pytest

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.utils_310p import run_vl_model_test


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@pytest.mark.parametrize("model", ["Qwen/Qwen3-8B", "Qwen/Qwen3.5-4B"])
def test_model_runner_v2_tp2_chunked_prefill_aclgraph(model: str) -> None:
    prompts = [("The following ledger contains numbered entries. " * 96) + "Summarize entry one."] * 4
    kwargs = {"mamba_ssm_cache_dtype": "float16"} if model == "Qwen/Qwen3.5-4B" else {}
    with VllmRunner(
        model,
        tensor_parallel_size=2,
        dtype="float16",
        max_model_len=4096,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        enable_prefix_caching=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
        },
        **kwargs,
    ) as runner:
        outputs = runner.generate_greedy(prompts, max_tokens=4)

    assert all(output[0] for output in outputs)


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_qwen3_vl_tp2() -> None:
    run_vl_model_test(
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        tensor_parallel_size=2,
        max_tokens=5,
        enable_prefix_caching=False,
    )
