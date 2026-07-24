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

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request import utils as e2e_utils
from tests.e2e.pull_request.utils import PROMPTS_SHORT, compare_logprobs

MODEL = "Qwen/Qwen3-0.6B"

_OFFLOAD_KEYS = {
    "offload_backend",
    "offload_group_size",
    "offload_num_in_group",
    "offload_prefetch_step",
    "offload_params",
    "cpu_offload_gb",
}


def _compare_offload_logprobs(
    runner_kwargs: dict,
    prompts: list[str],
    atol: float = 0.0689,
    decode_atol: float | None = None,
) -> None:
    """Compare prefetch/offload run against a no-offload eager baseline.

    Unlike ``compare_logprobs``, this keeps ``additional_config`` (e.g.
    ``weight_nz_mode``) on both sides and strips offload-related kwargs from
    the baseline so accuracy of the offloader itself is exercised.
    """
    if decode_atol is None:
        decode_atol = 2 * atol

    baseline_kwargs = {k: v for k, v in runner_kwargs.items() if k not in _OFFLOAD_KEYS}
    baseline_kwargs.pop("cudagraph_capture_sizes", None)
    baseline_kwargs["enforce_eager"] = True

    # baseline(eager, no offload)
    with VllmRunner(**baseline_kwargs) as runner:
        baseline_outputs = runner.model.generate(
            prompts=prompts,
            sampling_params=e2e_utils._LOGPROB_SAMPLING_PARAMS,
        )

    # enabled offload
    with VllmRunner(**runner_kwargs) as runner:
        offload_outputs = runner.model.generate(
            prompts=prompts,
            sampling_params=e2e_utils._LOGPROB_SAMPLING_PARAMS,
        )

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


# -------------------- Prefetch backend tests --------------------


@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "graph"])
@pytest.mark.parametrize("nz_mode", [0, 1], ids=["ND", "NZ"])
@pytest.mark.requires_hardware("A2")
@pytest.mark.requires_npus(1)
@wait_until_npu_memory_free()
def test_prefetch_offload_accuracy(enforce_eager, nz_mode):
    """Test prefetch CPU offloading across eager/graph × ND/NZ.

    Compares outputs between:
    1. Baseline (eager, no offloading, same weight_nz_mode)
    2. Prefetch offloading (group_size=4, num_in_group=1)

    NZ uses weight_nz_mode=2 so BF16 weights are converted to FRACTAL_NZ
    (mode 1 only enables NZ for quantized weights).
    """
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

    _compare_offload_logprobs(runner_kwargs=runner_kwargs, prompts=PROMPTS_SHORT)


@wait_until_npu_memory_free()
@pytest.mark.requires_hardware("A2")
@pytest.mark.requires_npus(1)
def test_prefetch_offload_selective_params():
    """Test selective parameter offloading (MLP weights only).

    Only offloads gate_up_proj and down_proj parameters, leaving
    attention weights on NPU.
    """
    runner_kwargs = {
        "model_name": MODEL,
        "max_model_len": 512,
        "enforce_eager": True,
        "offload_backend": "prefetch",
        "offload_group_size": 8,
        "offload_num_in_group": 2,
        "offload_prefetch_step": 1,
        "offload_params": {"gate_up_proj", "down_proj"},
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=PROMPTS_SHORT)
