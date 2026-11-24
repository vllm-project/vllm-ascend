# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch

from tests.e2e.models.minimax_m31_lora.m31_observation import AddObservation
from vllm_ascend.lora.lora_ops import _LORA_WRAPPER_IDS, _LORA_WRAPPERS, lora_linear
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU


def test_observes_branch_even_when_bf16_addition_loses_it():
    output = torch.tensor([[128.0, 2.0]], dtype=torch.bfloat16)
    delta = torch.tensor([[0.25, 0.5]], dtype=torch.bfloat16)
    before = output.clone()
    with AddObservation(output) as observation:
        torch.zeros_like(output).add_(7)
        output.add_(delta)
    assert torch.equal(observation.base, before)
    assert torch.equal(observation.delta, delta)
    assert output[0, 0] == before[0, 0]
    assert torch.equal(output, before + delta)


def test_rejects_ambiguous_additions():
    output = torch.zeros(2)
    with pytest.raises(AssertionError, match="one unscaled"), AddObservation(output):
        output.add_(torch.ones_like(output))
        output.add_(torch.ones_like(output))


def test_observation_survives_the_actual_custom_op_boundary():
    wrapper = PunicaWrapperNPU.__new__(PunicaWrapperNPU)
    wrapper._single_lora_slot = True
    wrapper._single_lora_mask = torch.ones(3, 1, dtype=torch.bfloat16)
    identity = next(_LORA_WRAPPER_IDS)
    _LORA_WRAPPERS[identity] = wrapper
    observations = []
    original = wrapper._lora_linear_matmul

    def collect(y, *args):
        with AddObservation(y) as observation:
            original(y, *args)
        observations.append(observation)

    wrapper._lora_linear_matmul = collect
    x = torch.ones(3, 4, dtype=torch.bfloat16)
    y = torch.full_like(x, 128)
    a = torch.ones(1, 1, 2, 4, dtype=x.dtype)
    b = torch.full((1, 1, 4, 2), 1 / 32, dtype=x.dtype)
    try:
        lora_linear(identity, y, x, [a], [b], 1.0, [4], None, None, True)
    finally:
        del _LORA_WRAPPERS[identity]
    assert len(observations) == 1
    assert torch.equal(observations[0].delta, torch.full_like(x, 0.25))
    assert torch.equal(y, torch.full_like(x, 128))
