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
"""Qwen3.6-35B-A3B BF16 DFLASH acceptance on MRV2 over two NPUs.

Run "pytest tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_6_dflash.py".
"""

import os
from unittest.mock import patch

import pytest
from vllm.config import CompilationConfig

from tests.e2e.pull_request.utils import (
    ACCEPTANCE_LENGTH_RTOL,
    SPEC_DECODE_PROMPTS,
    _run_speculative_decoding,
)

QWEN36_MOE_MODEL = "Qwen/Qwen3.6-35B-A3B"
QWEN36_DFLASH_DRAFT_MODEL = "rainney/AEON-DFlash-Qwen3.6-35B-A3B"
MODELS = [QWEN36_MOE_MODEL]
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MAX_MODEL_LEN = 72320
MAX_NUM_BATCHED_TOKENS = 16384
GPU_MEMORY_UTILIZATION = 0.95
# The adaptive case is a functional smoke test, so accept the full valid
# acceptance-length range [1, num_speculative_tokens + 1].
ADAPTIVE_ACCEPTANCE_LENGTH_RTOL = 0.78
ADAPTIVE_SMOKE_MAX_TOKENS = 512


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize(
    (
        "expected_acceptance_length",
        "acceptance_length_rtol",
        "num_speculative_tokens",
        "enable_adaptive_verification",
        "max_tokens",
        "additional_config",
    ),
    [
        pytest.param(
            4.0,
            ACCEPTANCE_LENGTH_RTOL,
            7,
            False,
            8192,
            {"ascend_compilation_config": {"enable_npugraph_ex": False}},
            id="dflash-qwen36-35b",
        ),
        pytest.param(
            4.5,
            ADAPTIVE_ACCEPTANCE_LENGTH_RTOL,
            7,
            True,
            ADAPTIVE_SMOKE_MAX_TOKENS,
            {"ascend_compilation_config": {"enable_npugraph_ex": False}},
            id="dflash-qwen36-35b-adaptive-verification",
        ),
    ],
)
@patch.dict(
    os.environ,
    {
        "OMP_NUM_THREADS": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_BUFFSIZE": "1024",
        "TASK_QUEUE_ENABLE": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "LCCL_DETERMINISTIC": "1",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "HCCL_DETERMINISTIC": "true",
        "CLOSE_MATMUL_K_SHIFT": "1",
    },
)
def test_qwen36_35b_dflash_acceptance_tp2(
    model_name,
    expected_acceptance_length,
    acceptance_length_rtol,
    num_speculative_tokens,
    enable_adaptive_verification,
    max_tokens,
    additional_config,
):
    speculative_config: dict[str, object] = {
        "method": "dflash",
        "model": QWEN36_DFLASH_DRAFT_MODEL,
        "num_speculative_tokens": num_speculative_tokens,
    }
    if enable_adaptive_verification:
        speculative_config["enable_adaptive_verification"] = True

    _run_speculative_decoding(
        model_name=model_name,
        speculative_config=speculative_config,
        example_prompts=SPEC_DECODE_PROMPTS,
        expected_acceptance_length=expected_acceptance_length,
        acceptance_length_rtol=acceptance_length_rtol,
        runner_kwargs={
            "tensor_parallel_size": 2,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "generation_config": "vllm",
            "compilation_config": CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY"),
            "additional_config": additional_config,
            "enable_prefix_caching": False,
            "async_scheduling": True,
        },
        is_moe=True,
        max_tokens=max_tokens,
    )
