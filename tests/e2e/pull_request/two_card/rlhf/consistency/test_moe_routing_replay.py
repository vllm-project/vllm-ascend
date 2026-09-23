#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""End-to-end routing replay consistency tests for MoE models.

Tests that routed experts output is valid and consistent. Only the MRV2 model
runner is exercised: after vLLM #45635, routed-experts capture is owned by the
AuxOutput connector, which MRV1 does not wire up (and upstream deleted the MRV1
side outright), so requesting it with MRV1 is rejected during config validation.
"""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from tests.e2e.conftest import VllmRunner

MODELS = [
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3-30B-A3B",
]


@pytest.mark.parametrize("model", MODELS)
@patch.dict(os.environ, {"OMP_NUM_THREADS": "1", "VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_moe_routing_replay(model):
    """Routed experts output is valid and non-empty for MoE models (MRV2)."""
    prompts = ["Hello, please introduce yourself."]
    with VllmRunner(
        model,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        distributed_executor_backend="mp",
        enable_return_routed_experts=True,
        async_scheduling=False,
    ) as vllm_model:
        sampling_params = SamplingParams(
            max_tokens=5,
            temperature=0.8,
            top_p=0.95,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        inputs = vllm_model.get_inputs(prompts=prompts)
        outputs = vllm_model.model.generate(prompts=inputs, sampling_params=sampling_params)
        assert outputs[0].finished
        assert len(outputs[0].outputs[0].text) > 0
        routed_experts = outputs[0].outputs[0].routed_experts
        assert routed_experts is not None
        assert routed_experts.size > 0
