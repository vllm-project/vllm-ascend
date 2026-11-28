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

CASES = [
    ("Qwen/Qwen3.5-35B-A3B", "default"),
    ("Qwen/Qwen3-30B-A3B", "default"),
    ("Qwen/Qwen3.5-35B-A3B", "allgather"),
    ("Qwen/Qwen3-30B-A3B", "fused_mc2"),
]


@pytest.mark.parametrize("model,comm_case", CASES)
@patch.dict(os.environ, {"OMP_NUM_THREADS": "1"})
def test_moe_routing_replay(model, comm_case):
    """Every token row carries its own routing decision (regression for #15451).

    With DP == 1 and TP > 1 the TP-sharded strategies still pad and split the
    token dimension inside the TP group, so ``select_experts`` only routes this
    rank's shard; a capture that treats that shard as the whole batch leaves the
    rows owned by the other TP ranks unwritten.

    The default EP cases cover MC2 and ALLTOALL in one engine. Disabling EP
    selects ALLGATHER; the fused configuration selects FUSED_MC2. Prompts are
    submitted one at a time so each prefill is a separate step.
    """
    prompts = ["Hello, please introduce yourself."]
    if comm_case == "default":
        # With graph sizes up to 8, the long prefill exceeds MC2 capacity.
        prompts.append("Hello, please introduce yourself. " * 8)
    additional_config = {"enable_fused_mc2": 1} if comm_case == "fused_mc2" else None
    with VllmRunner(
        model,
        tensor_parallel_size=2,
        enable_expert_parallel=comm_case != "allgather",
        additional_config=additional_config,
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
