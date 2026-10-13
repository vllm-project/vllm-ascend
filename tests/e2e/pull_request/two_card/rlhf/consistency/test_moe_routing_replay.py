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

# The TP-sharded capture paths pinned by the case below are model independent, so it
# runs on a single model: `test_moe_routing_replay` already loads the other one, and
# loading a 30B+ model a second time would double this file's runtime for no extra
# coverage.
TP_SHARDED_MODEL = "Qwen/Qwen3-30B-A3B"


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


@pytest.mark.parametrize("model", MODELS)
@patch.dict(os.environ, {"OMP_NUM_THREADS": "1", "VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_moe_routing_replay_covers_every_token(model):
    """Every token row carries its own routing decision, under MC2 and ALLTOALL.

    Regression test for #15451. Both strategies pad and split the token dimension
    inside the TP group even with DP == 1, so ``select_experts`` only routes this
    rank's shard. A capture that treats that shard as the complete batch writes
    only the first ``batch / tp_size`` rows and leaves the rows owned by the other
    TP ranks unwritten, which routing replay then serves as zeros.

    The two strategies are picked by ``max_num_tokens <= mc2_tokens_capacity``
    (``set_moe_comm_method``), and in eager mode without CUDA-graph sizes that
    capacity is ``max_num_reqs * decode_query_len``, i.e. ``max_num_seqs`` here.
    Pinning ``max_num_seqs`` to 8 therefore gives a deterministic threshold: the
    short prompt routes through MC2 and the long one through ALLTOALL.

    Eager mode is deliberate: graph warm-up would pre-fill the capture buffer, so
    an unwritten row would hold stale expert ids instead of zeros and the
    assertion below could no longer tell the two apart.

    Runs on one model only, see ``TP_SHARDED_MODEL``.
    """
    if model != TP_SHARDED_MODEL:
        pytest.skip(f"TP-sharded capture is already covered on {TP_SHARDED_MODEL}")
    prompts = [
        "Hello, please introduce yourself.",  # 6 tokens -> MC2
        "Hello, please introduce yourself. " * 8,  # > 8 tokens -> ALLTOALL
    ]
    with VllmRunner(
        model,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enforce_eager=True,
        max_num_seqs=8,
        distributed_executor_backend="mp",
        enable_return_routed_experts=True,
        async_scheduling=False,
    ) as vllm_model:
        sampling_params = SamplingParams(
            max_tokens=8,
            temperature=0.0,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        for prompt in prompts:
            inputs = vllm_model.get_inputs(prompts=[prompt])
            outputs = vllm_model.model.generate(prompts=inputs, sampling_params=sampling_params)
            output = outputs[0].outputs[0]
            routed_experts = output.routed_experts
            assert routed_experts is not None
            # The last sampled token has not been forwarded yet, so it has no routing.
            expected_rows = len(outputs[0].prompt_token_ids) + len(output.token_ids) - 1
            assert routed_experts.shape[0] == expected_rows
            empty_rows = [row for row in range(expected_rows) if not routed_experts[row].any()]
            assert not empty_rows, (
                f"rows {empty_rows} of {expected_rows} carry no routing data for a "
                f"{len(outputs[0].prompt_token_ids)}-token prompt"
            )
