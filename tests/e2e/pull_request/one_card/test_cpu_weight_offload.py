# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""End-to-end tests for CPU weight offloading on Ascend NPU.

Covers both the prefetch backend (NPUPrefetchOffloader) and the UVA
backend (functional_call fallback path, since UVA is not available on
NPU hardware).  Tests verify that offloading produces the same outputs
as the baseline (no offloading).
"""

import os

import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.utils import PROMPTS_SHORT, compare_logprobs

MODEL = "Qwen/Qwen3-0.6B"


# -------------------- Prefetch backend tests --------------------


@wait_until_npu_memory_free()
def test_prefetch_offload_eager():
    """Test prefetch CPU offloading in eager mode.

    Compares outputs between:
    1. Baseline (eager, no offloading)
    2. Prefetch offloading (group_size=4, num_in_group=1)
       with enforce_eager=True (no ACL graph capture)
    """
    runner_kwargs = {
        "model_name": MODEL,
        "max_model_len": 512,
        "enforce_eager": True,
        "offload_backend": "prefetch",
        "offload_group_size": 4,
        "offload_num_in_group": 1,
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=PROMPTS_SHORT)


@wait_until_npu_memory_free()
def test_prefetch_offload_aclgraph():
    """Test prefetch CPU offloading with ACL graph capture.

    Compares outputs between:
    1. Baseline (eager, no offloading)
    2. Prefetch offloading (group_size=4, num_in_group=1)
       with ACL graph capture enabled (default, non-eager)
    """
    runner_kwargs = {
        "model_name": MODEL,
        "max_model_len": 512,
        "cudagraph_capture_sizes": [1, 2, 4, 8],
        "offload_backend": "prefetch",
        "offload_group_size": 4,
        "offload_num_in_group": 1,
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=PROMPTS_SHORT)


@wait_until_npu_memory_free()
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


# -------------------- UVA backend tests --------------------
# UVA (Unified Virtual Addressing) is not available on Ascend NPU, so
# the UVA offloader falls back to the functional_call path that moves
# weights to device on-demand.  Tests below mirror the upstream
# test_cpu_offload.py parametrization but only exercise the non-UVA
# (functional_call) path with enforce_eager, as NPU does not support
# UVA zero-copy.


@pytest.mark.parametrize("disable_pin_memory", [False, True])
@wait_until_npu_memory_free()
def test_uva_offload_functional_call(disable_pin_memory):
    """Test UVA offloader's functional_call fallback on NPU.

    With UVA disabled (forced by env var), the UVA offloader falls back
    to moving weights to device inside a functional_call wrapper.
    enforce_eager is required because this fallback is incompatible
    with graph capture.

    Parametrized over pin_memory to cover both pinned and unpinned
    CPU storage paths.
    """
    old_uva = os.environ.get("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA")
    old_pin = os.environ.get("VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY")
    try:
        os.environ["VLLM_WEIGHT_OFFLOADING_DISABLE_UVA"] = "1"
        os.environ["VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY"] = str(int(disable_pin_memory))
        runner_kwargs = {
            "model_name": MODEL,
            "max_model_len": 512,
            "enforce_eager": True,
            "cpu_offload_gb": 1,
        }
        compare_logprobs(runner_kwargs=runner_kwargs, prompts=PROMPTS_SHORT)
    finally:
        if old_uva is None:
            os.environ.pop("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", None)
        else:
            os.environ["VLLM_WEIGHT_OFFLOADING_DISABLE_UVA"] = old_uva
        if old_pin is None:
            os.environ.pop("VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY", None)
        else:
            os.environ["VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY"] = old_pin
