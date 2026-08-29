#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Validate GLM-5.2 generation with DSpark and MTP speculative decoding.

Run pytest tests/e2e/pull_request/eight_card/test_glm5_2.py.
"""

import os
from unittest.mock import patch

import pytest
from vllm.config import CompilationConfig

from tests.e2e.pull_request.utils import _run_speculative_decoding

MAIN_MODEL = "Eco-Tech/GLM-5.2-w4a8"
SPECULATOR_MODEL = "RedHatAI/GLM-5.2-speculator.dspark"
DSPARK_NUM_SPECULATIVE_TOKENS = 7
MTP_NUM_SPECULATIVE_TOKENS = 3
DSPARK_EXPECTED_ACCEPTANCE_LENGTH = 3.61
MTP_EXPECTED_ACCEPTANCE_LENGTH = 3.11

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


@pytest.mark.e2e_model(MAIN_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="spec_decode,aclgraph",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "HCCL_BUFFSIZE": "512",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "LCCL_DETERMINISTIC": "1",
        "HCCL_DETERMINISTIC": "true",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "CLOSE_MATMUL_K_SHIFT": "1",
    },
)
def test_glm_5_2_dspark_acceptance_tp8() -> None:
    _run_speculative_decoding(
        model_name=MAIN_MODEL,
        speculative_config={
            "method": "dspark",
            "model": SPECULATOR_MODEL,
            "num_speculative_tokens": DSPARK_NUM_SPECULATIVE_TOKENS,
            "enforce_eager": True,
        },
        expected_acceptance_length=DSPARK_EXPECTED_ACCEPTANCE_LENGTH,
        runner_kwargs={
            "quantization": "ascend",
            "tensor_parallel_size": 8,
            "max_model_len": 8192,
            "compilation_config": CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY"),
        },
    )


@pytest.mark.e2e_model(MAIN_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,aclgraph",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "HCCL_BUFFSIZE": "512",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "LCCL_DETERMINISTIC": "1",
        "HCCL_DETERMINISTIC": "true",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "CLOSE_MATMUL_K_SHIFT": "1",
    },
)
def test_glm_5_2_mtp_acceptance_tp8() -> None:
    _run_speculative_decoding(
        model_name=MAIN_MODEL,
        speculative_config={
            "method": "deepseek_mtp",
            "num_speculative_tokens": MTP_NUM_SPECULATIVE_TOKENS,
            "enforce_eager": True,
        },
        expected_acceptance_length=MTP_EXPECTED_ACCEPTANCE_LENGTH,
        runner_kwargs={
            "quantization": "ascend",
            "tensor_parallel_size": 8,
            "max_model_len": 8192,
            "compilation_config": CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY"),
        },
    )
