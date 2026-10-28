#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""End-to-end routing replay consistency tests for MoE models.

Only the V2 model runner is exercised: routed-experts capture (R3 / routing
replay) is owned by vLLM's AuxOutput connector, which is only integrated in the
V2 model runner, so the V1 model runner no longer serves R3.
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
    """Every token row carries its own routing decision (regression for #15451).

    With DP == 1 and TP > 1 the TP-sharded strategies still pad and split the
    token dimension inside the TP group, so ``select_experts`` only routes this
    rank's shard; a capture that treats that shard as the whole batch leaves the
    rows owned by the other TP ranks unwritten.

    Both strategies are covered from this one engine. ``set_moe_comm_method``
    picks MC2 for a batch of at most ``mc2_tokens_capacity`` tokens and ALLTOALL
    above it, and ``set_mc2_tokens_capacity`` derives that capacity from
    ``max_cudagraph_capture_size`` when ``cudagraph_capture_sizes`` is set -- 8
    tokens here, split over the two TP ranks. The prompts are submitted one at a
    time so each prefill is a step of its own: the short one prefills inside the
    size-8 graph (still MC2), the long one goes through ALLTOALL.
    """
    prompts = [
        "Hello, please introduce yourself.",  # 6 tokens -> MC2
        "Hello, please introduce yourself. " * 8,  # 49 tokens -> ALLTOALL
    ]
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
        for prompt in prompts:
            inputs = vllm_model.get_inputs(prompts=[prompt])
            outputs = vllm_model.model.generate(prompts=inputs, sampling_params=sampling_params)
            assert outputs[0].finished
            completion = outputs[0].outputs[0]
            assert len(completion.text) > 0
            routed_experts = completion.routed_experts
            assert routed_experts is not None
            assert routed_experts.size > 0
            # The last sampled token has not been forwarded yet, so it has no routing.
            expected_rows = len(outputs[0].prompt_token_ids) + len(completion.token_ids) - 1
            assert routed_experts.shape[0] == expected_rows
            empty_rows = [row for row in range(expected_rows) if not routed_experts[row].any()]
            assert not empty_rows, (
                f"rows {empty_rows} of {expected_rows} carry no routing data for a "
                f"{len(outputs[0].prompt_token_ids)}-token prompt"
            )
