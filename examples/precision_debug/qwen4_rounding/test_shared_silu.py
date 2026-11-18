# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import pytest
import torch
from shared_silu import silu_bf16_then_mul


def test_rounding_before_multiplication():
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(37, 26, generator=generator).bfloat16()
    gate, up = x.chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate.float()).bfloat16().float() * up.float()).bfloat16()
    result = silu_bf16_then_mul(x)
    assert torch.equal(result, expected)
    fused = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
    assert not torch.equal(result, fused)


def test_invalid_shape():
    with pytest.raises(ValueError):
        silu_bf16_then_mul(torch.ones(1, 3).bfloat16())
