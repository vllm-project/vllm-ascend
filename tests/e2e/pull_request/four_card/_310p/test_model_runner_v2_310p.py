# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from unittest.mock import patch

from tests.e2e.conftest import VllmRunner


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen3_dense_mrv2_tp2_aclgraph_fp16():
    with VllmRunner(
        "Qwen/Qwen3-8B",
        tensor_parallel_size=2,
        enforce_eager=False,
        enable_prefix_caching=False,
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
        enable_prefix_caching=False,
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
