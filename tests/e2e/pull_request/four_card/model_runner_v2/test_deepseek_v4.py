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
#
import os
from unittest.mock import patch

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = "gdydems/DeepSeek-V4-Flash-w4a8-mtp"


@pytest.mark.skip("Temporarily skip this DeepSeek V4 test.")
@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp",
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
    },
)
@wait_until_npu_memory_free()
def test_deepseek_v4_mtp_eager():
    """Verify DeepSeek V4 MTP decoding with ModelRunner V2."""
    prompts = [
        "Hello, my name is",
        "What is the meaning of life?",
    ]

    with VllmRunner(
        MODEL,
        max_model_len=8192,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        dtype="auto",
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.9,
        quantization="ascend",
        tokenizer_mode="deepseek_v4",
        block_size=128,
        enforce_eager=True,
        speculative_config={"num_speculative_tokens": 3, "method": "mtp"},
        additional_config={"enable_dsa_cp": False},
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens=5)

    expected_token_ids = [
        [19923, 14, 1026, 2329, 344, 680, 2852, 95, 305, 342],
        [3085, 344, 270, 5281, 294, 1988, 33, 3955, 361, 582, 3085, 344],
    ]
    assert len(outputs) == len(prompts)
    for (output_ids, output_str), expected_ids in zip(outputs, expected_token_ids):
        assert output_str
        assert output_ids == expected_ids
