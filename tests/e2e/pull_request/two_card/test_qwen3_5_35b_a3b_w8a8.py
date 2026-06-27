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

import os
from unittest.mock import patch

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

EXAMPLE_PROMPTS = [
    "Hello, my name is",
]


@pytest.mark.e2e_features("multimodal", "full_decode_only", "w8a8", "mtp", "mamba_ssm")
@pytest.mark.e2e_model("Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp")
@patch.dict(os.environ, {"HCCL_BUFFSIZE": "1024"})
@wait_until_npu_memory_free()
def test_qwen3_5_35b_a3b_w8a8_tp2_without_ep():
    with VllmRunner(
        "Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp",
        max_model_len=4096,
        tensor_parallel_size=2,
        enable_expert_parallel=False,
        quantization="ascend",
        gpu_memory_utilization=0.9,
        distributed_executor_backend="mp",
        cudagraph_capture_sizes=[1, 2, 4, 8],
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(EXAMPLE_PROMPTS, max_tokens=5)

    assert outputs[0][1]
