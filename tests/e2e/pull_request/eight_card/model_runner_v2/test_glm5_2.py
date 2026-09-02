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

import os
from unittest.mock import patch

import pytest
from vllm.config import CompilationConfig

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.utils import _run_speculative_decoding

MODEL = "Eco-Tech/GLM-5.2-w4a8"
DRAFT_MODEL = "RedHatAI/GLM-5.2-speculator.dspark"
EXPECTED_ACCEPTANCE_LENGTH = 3.0
DSPARK_EXPECTED_ACCEPTANCE_LENGTH = 3.5


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free()
def test_glm5_2_mtp_full_decode_only() -> None:
    _run_speculative_decoding(
        model_name=MODEL,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
        },
        expected_acceptance_length=EXPECTED_ACCEPTANCE_LENGTH,
        runner_kwargs={
            "quantization": "ascend",
            "tensor_parallel_size": 8,
            "max_model_len": 8192,
            "compilation_config": CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY"),
        },
    )


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dspark",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="eager",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_BUFFSIZE": "1024",
    },
)
@wait_until_npu_memory_free()
def test_glm5_2_dspark_eager() -> None:
    _run_speculative_decoding(
        model_name=MODEL,
        speculative_config={
            "method": "dspark",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": 7,
            "enforce_eager": True,
        },
        expected_acceptance_length=DSPARK_EXPECTED_ACCEPTANCE_LENGTH,
        runner_kwargs={
            "quantization": "ascend",
            "tensor_parallel_size": 8,
            "max_model_len": 4096,
            "max_num_batched_tokens": 2048,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "async_scheduling": False,
        },
        acceptance_length_rtol=0.1,
    )
