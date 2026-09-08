# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""End-to-end tests for CPU weight offloading on Ascend NPU.

Covers the prefetch backend (AscendPrefetchOffloader).
Tests verify that offloading produces the same outputs
as the baseline (no offloading).
"""

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request import utils as e2e_utils
from tests.e2e.pull_request.utils import PROMPTS_SHORT

MODEL = "Qwen/Qwen3-0.6B"
# 4 short prompts x 3 tokens. Unpinned capture / default max_num_batched_tokens
# make torch.compile and ACL graphs dominate runtime.
_BASE_KWARGS = {
    "model_name": MODEL,
    "max_model_len": 512,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 256,
}
_CUDAGRAPH_CAPTURE_SIZES = [1, 2, 4, 8]
_PREFETCH_KWARGS = {
    "offload_backend": "prefetch",
    "offload_group_size": 4,
    "offload_num_in_group": 1,
}


def _generate(runner_kwargs: dict):
    with VllmRunner(**runner_kwargs) as runner:
        return runner.model.generate(
            prompts=PROMPTS_SHORT,
            sampling_params=e2e_utils._LOGPROB_SAMPLING_PARAMS,
        )


def _eager_baseline_kwargs(nz_mode: int) -> dict:
    return {
        **_BASE_KWARGS,
        "enforce_eager": True,
        "additional_config": {"weight_nz_mode": nz_mode},
    }


def _assert_offload_logprobs(baseline_outputs, offload_outputs, atol: float = 0.0689) -> None:
    decode_atol = 2 * atol
    for prompt_idx, (base_out, offload_out) in enumerate(zip(baseline_outputs, offload_outputs)):
        base_seq = base_out.outputs[0]
        offload_seq = offload_out.outputs[0]

        assert base_seq.logprobs is not None and offload_seq.logprobs is not None, (
            f"logprobs not returned for prompt {prompt_idx}"
        )
        assert len(base_seq.token_ids) == len(offload_seq.token_ids) == 3, (
            f"Expected 3 tokens for prompt {prompt_idx}, "
            f"got baseline={len(base_seq.token_ids)}, offload={len(offload_seq.token_ids)}"
        )

        e2e_utils._check_prefill_token(base_seq, offload_seq, prompt_idx, atol)
        for token_idx in range(1, 3):
            e2e_utils._check_decode_token(base_seq, offload_seq, token_idx, prompt_idx, decode_atol)


@pytest.fixture(scope="module")
def nd_baseline_outputs():
    """Eager, no offload, weight_nz_mode=0. Shared by ND eager and graph cases."""
    return _generate(_eager_baseline_kwargs(0))


@pytest.fixture(scope="module")
def nz_baseline_outputs():
    """Eager, no offload, weight_nz_mode=1 (Ascend default). Shared by NZ and selective."""
    return _generate(_eager_baseline_kwargs(1))


def _prefetch_runner_kwargs(nz_mode: int, enforce_eager: bool) -> dict:
    kwargs = {
        **_BASE_KWARGS,
        **_PREFETCH_KWARGS,
        "additional_config": {"weight_nz_mode": nz_mode},
    }
    if enforce_eager:
        kwargs["enforce_eager"] = True
    else:
        kwargs["cudagraph_capture_sizes"] = _CUDAGRAPH_CAPTURE_SIZES
    return kwargs


@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "graph"])
@wait_until_npu_memory_free()
def test_prefetch_offload_accuracy_nd(nd_baseline_outputs, enforce_eager):
    """Prefetch offload vs shared ND eager baseline (eager and graph)."""
    offload_outputs = _generate(_prefetch_runner_kwargs(nz_mode=0, enforce_eager=enforce_eager))
    _assert_offload_logprobs(nd_baseline_outputs, offload_outputs)


@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "graph"])
@wait_until_npu_memory_free()
def test_prefetch_offload_accuracy_nz(nz_baseline_outputs, enforce_eager):
    """Prefetch offload vs shared NZ eager baseline (eager and graph)."""
    offload_outputs = _generate(_prefetch_runner_kwargs(nz_mode=1, enforce_eager=enforce_eager))
    _assert_offload_logprobs(nz_baseline_outputs, offload_outputs)


@wait_until_npu_memory_free()
def test_prefetch_offload_selective_params(nz_baseline_outputs):
    """Offload MLP weights only; compare against the shared NZ no-offload baseline."""
    offload_outputs = _generate(
        {
            **_BASE_KWARGS,
            "enforce_eager": True,
            "additional_config": {"weight_nz_mode": 1},
            "offload_backend": "prefetch",
            "offload_group_size": 8,
            "offload_num_in_group": 2,
            "offload_prefetch_step": 1,
            "offload_params": {"gate_up_proj", "down_proj"},
        }
    )
    _assert_offload_logprobs(nz_baseline_outputs, offload_outputs)
