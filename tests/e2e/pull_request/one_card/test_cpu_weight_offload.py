# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""End-to-end tests for CPU weight offloading on Ascend NPU.

Covers both the prefetch backend (AscendPrefetchOffloader).
Tests verify that offloading produces the same outputs
as the baseline (no offloading).
"""

from vllm.outputs import RequestOutput

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request import utils as e2e_utils
from tests.e2e.pull_request.utils import PROMPTS_SHORT

MODEL = "Qwen/Qwen3-0.6B"


def _eager_baseline_kwargs(nz_mode: int) -> dict:
    return {
        "model_name": MODEL,
        "max_model_len": 512,
        "enforce_eager": True,
        "additional_config": {"weight_nz_mode": nz_mode},
    }


def _generate_eager_baseline(nz_mode: int) -> list[RequestOutput]:
    with VllmRunner(**_eager_baseline_kwargs(nz_mode)) as runner:
        return runner.model.generate(
            prompts=PROMPTS_SHORT,
            sampling_params=e2e_utils._LOGPROB_SAMPLING_PARAMS,
        )


def _assert_offload_matches_baseline(
    baseline_outputs: list[RequestOutput],
    runner_kwargs: dict,
    atol: float = 0.0689,
    decode_atol: float | None = None,
    num_runs: int = 3,
) -> None:
    """Generate with offload enabled and compare logprobs to a shared baseline."""
    if decode_atol is None:
        decode_atol = 2 * atol

    with VllmRunner(**runner_kwargs) as runner:
        for run_idx in range(num_runs):
            offload_outputs = runner.model.generate(
                prompts=PROMPTS_SHORT,
                sampling_params=e2e_utils._LOGPROB_SAMPLING_PARAMS,
            )

            for prompt_idx, (base_out, offload_out) in enumerate(zip(baseline_outputs, offload_outputs)):
                base_seq = base_out.outputs[0]
                offload_seq = offload_out.outputs[0]

                assert base_seq.logprobs is not None and offload_seq.logprobs is not None, (
                    f"logprobs not returned for prompt {prompt_idx} (run {run_idx})"
                )
                assert len(base_seq.token_ids) == len(offload_seq.token_ids) == 3, (
                    f"Expected 3 tokens for prompt {prompt_idx} (run {run_idx}), "
                    f"got baseline={len(base_seq.token_ids)}, "
                    f"offload={len(offload_seq.token_ids)}"
                )

                e2e_utils._check_prefill_token(base_seq, offload_seq, prompt_idx, atol)
                for token_idx in range(1, 3):
                    e2e_utils._check_decode_token(base_seq, offload_seq, token_idx, prompt_idx, decode_atol)


def _prefetch_kwargs(*, enforce_eager: bool, nz_mode: int) -> dict:
    runner_kwargs: dict = {
        "model_name": MODEL,
        "max_model_len": 512,
        "offload_backend": "prefetch",
        "offload_group_size": 4,
        "offload_num_in_group": 1,
        "additional_config": {"weight_nz_mode": nz_mode},
    }
    if enforce_eager:
        runner_kwargs["enforce_eager"] = True
    else:
        runner_kwargs["cudagraph_capture_sizes"] = [1, 2, 4, 8]
    return runner_kwargs


# -------------------- Prefetch backend tests --------------------


@wait_until_npu_memory_free()
def test_prefetch_offload_nd_accuracy():
    """ND prefetch offload vs one eager baseline (eager, graph, selective MLP)."""
    baseline_outputs = _generate_eager_baseline(nz_mode=0)

    _assert_offload_matches_baseline(
        baseline_outputs,
        _prefetch_kwargs(enforce_eager=True, nz_mode=0),
    )
    _assert_offload_matches_baseline(
        baseline_outputs,
        _prefetch_kwargs(enforce_eager=False, nz_mode=0),
    )
    _assert_offload_matches_baseline(
        baseline_outputs,
        {
            "model_name": MODEL,
            "max_model_len": 512,
            "enforce_eager": True,
            "offload_backend": "prefetch",
            "offload_group_size": 8,
            "offload_num_in_group": 2,
            "offload_prefetch_step": 1,
            "offload_params": {"gate_up_proj", "down_proj"},
            "additional_config": {"weight_nz_mode": 0},
        },
        num_runs=1,
    )


@wait_until_npu_memory_free()
def test_prefetch_offload_nz_eager_accuracy():
    """NZ prefetch offload vs one eager baseline."""
    baseline_outputs = _generate_eager_baseline(nz_mode=2)
    _assert_offload_matches_baseline(
        baseline_outputs,
        _prefetch_kwargs(enforce_eager=True, nz_mode=2),
    )


@wait_until_npu_memory_free()
def test_prefetch_offload_nz_graph_accuracy():
    """NZ+graph prefetch matches the eager baseline on Model Runner V2.

    Qwen3 defaults to MRv2, whose prefetch path accepts NZ graph capture.
    Model Runner V1 still fail-fasts in AscendPrefetchOffloader for this combo.
    """
    baseline_outputs = _generate_eager_baseline(nz_mode=2)
    _assert_offload_matches_baseline(
        baseline_outputs,
        _prefetch_kwargs(enforce_eager=False, nz_mode=2),
    )
