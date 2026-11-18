# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import pytest
import torch
from ple_residual import ple_residual_bf16


def test_first_sum_is_materialized():
    gated = torch.tensor([1.0], dtype=torch.bfloat16)
    conv = torch.tensor([0.00390625], dtype=torch.bfloat16)
    outer = torch.tensor([-1.0], dtype=torch.bfloat16)
    # Halfway at 1.0 rounds to even before cancellation with outer.
    assert ple_residual_bf16(gated, conv, outer).item() == 0
    assert (gated.float() + conv.float() + outer.float()).bfloat16().item() == 0.00390625
    assert gated.item() == 1.0 and outer.item() == -1.0


def test_broadcasting_is_rejected():
    with pytest.raises(ValueError):
        ple_residual_bf16(torch.ones(2).bfloat16(), torch.ones(1).bfloat16(), torch.ones(2).bfloat16())
