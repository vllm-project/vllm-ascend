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
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py`.
"""

import os
from unittest.mock import patch

import pytest
from vllm.config import CompilationConfig

from tests.e2e.pull_request.utils import _run_speculative_decoding

MODELS = ["UploadWeight/DeepSeek-V4-Flash-DSpark-w4a8-test"]
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Confidence-based dynamic verify-length; keep in sync with
# tests/e2e/pull_request/one_card/spec_decode/test_dynamic.py (dspark).
DSPARK_DYNAMIC_SPEC_CONFIG = {
    "method": "dspark",
    "method_params": {
        "initial_verify_budget_per_req": 3,
        "budget_update_interval": 1,
        "budget_threshold": 0.7,
    },
}


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize(
    ("expected_acceptance_length", "num_speculative_tokens", "additional_config"),
    [
        pytest.param(3.33, 5, {"enable_dsa_cp": False}, id="dspark"),
        pytest.param(3.45, 7, {"enable_dsa_cp": True}, id="dsa-cp-dspark"),
        pytest.param(
            3.35,
            5,
            {
                "enable_flashcomm1": False,
                "enable_dsa_cp": False,
                "dynamic_spec_config": DSPARK_DYNAMIC_SPEC_CONFIG,
            },
            id="dspark-dynamic",
        ),
    ],
)
@patch.dict(
    os.environ,
    {
        "HCCL_BUFFSIZE": "1024",
        "LCCL_DETERMINISTIC": "1",
        "HCCL_DETERMINISTIC": "true",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "CLOSE_MATMUL_K_SHIFT": "1",
    },
)
def test_deepseek_v4_dspark_acceptance_tp4(
    model_name,
    expected_acceptance_length,
    num_speculative_tokens,
    additional_config,
):
    _run_speculative_decoding(
        model_name=model_name,
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": num_speculative_tokens,
            "enforce_eager": True,
        },
        expected_acceptance_length=expected_acceptance_length,
        runner_kwargs={
            "tensor_parallel_size": 4,
            "max_model_len": 4096,
            "attention_config": {"indexer_kv_dtype": "fp8"},
            "compilation_config": CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY"),
            "additional_config": additional_config,
        },
    )
