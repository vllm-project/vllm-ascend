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
"""Qwen3.6-27B BF16 EAGLE3 acceptance on MRV2 over two NPUs.

Run "pytest tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_6_eagle3.py".
"""

import os
from unittest.mock import patch

import pytest
from vllm.config import CompilationConfig

from tests.e2e.pull_request.utils import SPEC_DECODE_PROMPTS, _run_speculative_decoding

QWEN36_DENSE_MODEL = "Qwen/Qwen3.6-27B"
QWEN36_EAGLE3_DRAFT_MODEL = "Ex0bit/Qwen3.6-27B-PRISM-EAGLE3/compressed"
MODELS = [QWEN36_DENSE_MODEL]
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MAX_MODEL_LEN = 72320
MAX_NUM_BATCHED_TOKENS = 16384
GPU_MEMORY_UTILIZATION = 0.95

_EAGER = os.environ.get("QWEN36_EAGLE3_EAGER") == "1"


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize(
    ("expected_acceptance_length", "num_speculative_tokens", "additional_config"),
    [
        pytest.param(
            2.43,
            3,
            {"ascend_compilation_config": {"enable_npugraph_ex": False}},
            id="eagle3-qwen36-27b",
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
        "VLLM_ASCEND_ENABLE_FUSED_MC2": "1",
        "ASCEND_RT_VISIBLE_DEVICES": "14,15",
    },
)
def test_qwen36_27b_eagle3_acceptance_tp2(
    model_name,
    expected_acceptance_length,
    num_speculative_tokens,
    additional_config,
):
    _run_speculative_decoding(
        model_name=model_name,
        speculative_config={
            "method": "eagle3",
            "model": QWEN36_EAGLE3_DRAFT_MODEL,
            "num_speculative_tokens": num_speculative_tokens,
        },
        example_prompts=SPEC_DECODE_PROMPTS,
        expected_acceptance_length=expected_acceptance_length,
        runner_kwargs={
            "tensor_parallel_size": 2,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "generation_config": "vllm",
            "compilation_config": (
                CompilationConfig(
                    cudagraph_mode="FULL_DECODE_ONLY",
                    cudagraph_capture_sizes=[
                        8, 12, 16, 20, 24, 28, 30, 34, 36, 38, 40,
                        52, 64, 72, 80, 88, 96, 100, 104, 108, 112, 116, 120,
                    ],
                )
            ),
            "additional_config": additional_config,
            "enable_prefix_caching": False,
            "async_scheduling": True,
        },
        is_moe=False,
        max_tokens=512,
    )

