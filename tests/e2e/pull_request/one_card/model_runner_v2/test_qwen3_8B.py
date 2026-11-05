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

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.utils import ACCEPTANCE_LENGTH_RTOL, _run_speculative_decoding

MODEL = "Qwen/Qwen3-8B"
EAGLE3_SPECULATOR = "RedHatAI/Qwen3-8B-speculator.eagle3"
EXPECTED_ACCEPTANCE_LENGTH = 2.03


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="eagle3",
    parallel="TP",
    deploy="pd_mix",
    quantization="",
    hardware="A2",
    graph_mode="eager",
)
@patch.dict(
    os.environ,
    {
        "HCCL_BUFFSIZE": "1024",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "10",
        "TASK_QUEUE_ENABLE": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "LCCL_DETERMINISTIC": "1",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "HCCL_DETERMINISTIC": "true",
        "CLOSE_MATMUL_K_SHIFT": "1",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_eagle3_dsd() -> None:
    """Verify Qwen3-8B Eagle3 DSD acceptance with ModelRunner V2."""
    num_speculative_tokens = 3
    acceptance_length = _run_speculative_decoding(
        model_name=MODEL,
        speculative_config={
            "method": "eagle3",
            "model": EAGLE3_SPECULATOR,
            "num_speculative_tokens": num_speculative_tokens,
            "enforce_eager": True,
            "num_speculative_tokens_per_batch_size": [
                [1, 6, 3],
                [7, 11, 2],
                [12, 16, 2],
            ],
        },
        expected_acceptance_length=EXPECTED_ACCEPTANCE_LENGTH,
        runner_kwargs={
            "max_model_len": 15500,
            "dtype": "auto",
            "tensor_parallel_size": 1,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": True,
            "async_scheduling": True,
            "generation_config": "vllm",
            "additional_config": {"enable_cpu_binding": True},
            "compilation_config": {"cudagraph_mode": "NONE"},
        },
        acceptance_length_rtol=ACCEPTANCE_LENGTH_RTOL,
        is_moe=False,
    )
    assert acceptance_length > 1.0, f"acceptance_length {acceptance_length:.4f} must be greater than 1.0"
