# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2026 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""DSpark speculative decoding under DP with Model Runner V2."""

import os
from unittest.mock import patch

import pytest

from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free

DEEPSEEK_V4_DSPARK_MODEL = os.environ.get(
    "DEEPSEEK_V4_DSPARK_MODEL_PATH",
    "UploadWeight/DeepSeek-V4-Flash-DSpark-w4a8-test",
)

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@pytest.mark.e2e_model(DEEPSEEK_V4_DSPARK_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dspark",
    parallel="DP,EP",
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
        "HCCL_BUFFSIZE": "2048",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v4_dspark_spec_decoding_dp4_full_graph() -> None:
    with DPVllmRunner(
        DEEPSEEK_V4_DSPARK_MODEL,
        data_parallel_size=2,
        tensor_parallel_size=2,
        max_model_len=4096,
        max_num_seqs=2,
        max_num_batched_tokens=512,
        enable_expert_parallel=True,
        tokenizer_mode="deepseek_v4",
        block_size=128,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.93,
        quantization="ascend",
        enforce_eager=False,
        async_scheduling=True,
        enable_prefix_caching=False,
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": 5,
            "enforce_eager": True,
        },
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [6, 12],
        },
        additional_config={
            "enable_dsa_cp": False,
            "enable_fused_mc2": 0,
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(PROMPTS, max_tokens=32)

    assert len(outputs) == len(PROMPTS)
    assert all(output_ids and output_text.strip() for output_ids, output_text in outputs)
