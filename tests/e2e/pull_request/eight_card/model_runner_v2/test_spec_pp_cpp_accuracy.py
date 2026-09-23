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
"""CPP-enabled prefill smoke variant of test_spec_pp_accuracy.py.

This file is the chunk pipeline parallel (profiling_chunk_config) companion
to tests/e2e/pull_request/eight_card/model_runner_v2/test_spec_pp_accuracy.py.
The two files are intentionally kept separate so the base DSpark e2e case
and the CPP e2e case can fail independently without affecting each other.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

DEEPSEEK_V4_MODEL = os.environ.get(
    "DEEPSEEK_V4_DSPARK_MODEL_PATH",
    "UploadWeight/DeepSeek-V4-Flash-DSpark-w4a8-test",
)

PREFILL_PROMPT = "Ali had $21. Leila gave him half of her $100. How much does Ali have now?"


def _assert_single_token_output(outputs) -> None:
    assert len(outputs) == 1
    output_ids, _ = outputs[0]
    assert len(output_ids) == 1


@pytest.mark.e2e_model(DEEPSEEK_V4_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="spec_decode",
    parallel="PP,TP,EP",
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
        "HCCL_BUFFSIZE": "2048",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v4_dspark_pp_cpp_prefill() -> None:
    with VllmRunner(
        DEEPSEEK_V4_MODEL,
        max_model_len=4096,
        max_num_seqs=2,
        max_num_batched_tokens=512,
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        quantization="ascend",
        tokenizer_mode="deepseek_v4",
        block_size=128,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        attention_config={
            "indexer_kv_dtype": "int8",
        },
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": 5,
            "enforce_eager": True,
        },
        additional_config={
            "enable_dsa_cp": False,
            "enable_fused_mc2": 0,
            "scheduler_config": {"profiling_chunk_config": {"enabled": True}},
        },
    ) as runner:
        outputs = runner.generate_greedy([PREFILL_PROMPT], max_tokens=1)

    _assert_single_token_output(outputs)
