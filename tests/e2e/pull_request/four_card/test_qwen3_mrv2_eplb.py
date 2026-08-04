# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os
from typing import Any
from unittest.mock import patch

import pytest

from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free
from tests.e2e.model_utils import check_outputs_equal

MODEL = os.environ.get("QWEN3_MRV2_EPLB_MODEL_PATH", "vllm-ascend/Qwen3-30B-A3B-W8A8")
PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at a temperature of",
    "The author of Pride and Prejudice is",
    "The chemical symbol for gold is",
    "The square root of 144 is",
    "The opposite of hot is",
    "The first month of the year is",
]


def _run_dp2_tp2(*, enable_eplb: bool):
    runner_kwargs: dict[str, Any] = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "enable_expert_parallel": True,
        "max_model_len": 1024,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 1024,
        "enforce_eager": True,
        "quantization": "ascend",
        "distributed_executor_backend": "mp",
        "async_scheduling": False,
        "gpu_memory_utilization": 0.7,
        "block_size": 128,
        "dp_start_timeout": 1800,
        "dp_request_timeout": 1800,
    }
    if enable_eplb:
        runner_kwargs.update(
            {
                "enable_eplb": True,
                "eplb_config": {
                    "window_size": 2,
                    "step_interval": 2,
                    "num_redundant_experts": 16,
                    "log_balancedness": True,
                    "log_balancedness_interval": 1,
                    "use_async": False,
                },
                "additional_config": {"eplb_config": {"load_collection_phase": "prefill"}},
            }
        )

    with DPVllmRunner(MODEL, **runner_kwargs) as runner:
        return runner.generate_greedy(PROMPTS, max_tokens=16)


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="eplb",
    parallel="DP,TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "1024",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.7, max_wait_seconds=180)
def test_qwen3_moe_w8a8_dp2_tp2_sync_eplb_accuracy():
    baseline_outputs = _run_dp2_tp2(enable_eplb=False)
    eplb_outputs = _run_dp2_tp2(enable_eplb=True)

    check_outputs_equal(
        outputs_0_lst=eplb_outputs,
        outputs_1_lst=baseline_outputs,
        name_0="MRV2 synchronous EPLB",
        name_1="MRV2 baseline",
    )
