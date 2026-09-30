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

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_GRAPH_ENV = {
    "HCCL_BUFFSIZE": "1024",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
}


def _run_full_decode_graph_smoke(model: str, *, enable_expert_parallel: bool) -> None:
    with VllmRunner(
        model,
        tensor_parallel_size=2,
        enable_expert_parallel=enable_expert_parallel,
        distributed_executor_backend="mp",
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.8,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(["Explain what an NPU is in one sentence."], max_tokens=4)

    assert len(outputs) == 1
    output_ids, _ = outputs[0]
    assert output_ids


@pytest.mark.e2e_model("google/gemma-4-26B-A4B-it")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@patch.dict(os.environ, _GRAPH_ENV)
@wait_until_npu_memory_free()
def test_gemma4_26b_a4b_moe_full_decode_graph() -> None:
    """Verify Gemma4 MoE startup, graph capture, and short generation."""
    _run_full_decode_graph_smoke("google/gemma-4-26B-A4B-it", enable_expert_parallel=True)
