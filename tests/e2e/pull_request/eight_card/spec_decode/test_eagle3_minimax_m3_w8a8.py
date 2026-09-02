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

Run `pytest tests/e2e/pull_request/eight_card/spec_decode/test_eagle3_minimax_m3.py`.
"""

import os
from unittest.mock import patch

import pytest

from tests.e2e.pull_request.utils import _run_speculative_decoding

MODELS = ["/mnt/weight/MiniMax-M3-w8a8"]
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize(
    ("expected_acceptance_length", "num_speculative_tokens", "additional_config"),
    [
        pytest.param(
            2.68,
            3,
            {
                "ascend_compilation_config": {"enable_npugraph_ex": False}
            },
            id="eagle3-minimax-m3-w8a8",
        ),
    ],
)
@patch.dict(
    os.environ,
    {
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "1",
        "TASK_QUEUE_ENABLE": "1",
        "VLLM_LOGGING_LEVEL": "INFO",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "LCCL_DETERMINISTI": "1",
        "HCCL_DETERMINISTIC": "true",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "CLOSE_MATMUL_K_SHIFT": "1",
        "HCCL_BUFFSIZE": "1024",
        "VLLM_ASCEND_ENABLE_FUSED_MC2": "1",
        "ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7,8,9,10,11",
        "ASCEND_LAUNCH_BLOCKING": "1",
    },
)
def test_minimax_m3_eagle3_acceptance_tp8(
    model_name,
    expected_acceptance_length,
    num_speculative_tokens,
    additional_config,
):
    _run_speculative_decoding(
        model_name=model_name,
        speculative_config={
            "method": "eagle3",
            "model": "/mnt/weight/MiniMax-M3-EAGLE3",
            "num_speculative_tokens": num_speculative_tokens,
            "enforce_eager": True,
            "attention_backend": "FLASH_ATTN",
        },
        expected_acceptance_length=expected_acceptance_length,
        runner_kwargs={
            "tensor_parallel_size": 8,     
            "max_model_len": 8192,     
            "max_num_batched_tokens": 4096, 
            "enforce_eager": True, 
            "additional_config": additional_config,
        },
    )