# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Rounding boundaries in the actual Ascend runtime entry points."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.ops.activation import AscendSiluAndMul
from vllm_ascend.ops.triton_ple import ple_gate


def test_shared_silu_scope():
    torch.manual_seed(42)
    x = torch.randn(17, 1280, device="npu", dtype=torch.bfloat16)
    with set_current_vllm_config(VllmConfig()):
        op = AscendSiluAndMul()
    assert torch.equal(op.forward_oot(x), torch_npu.npu_swiglu(x))
    op.materialize_bf16_silu = True
    gate, up = x.chunk(2, -1)
    gate = gate.float()
    expected = ((gate * gate.sigmoid()).bfloat16().float() * up.float()).bfloat16()
    assert torch.equal(op.forward_oot(x), expected)
    fp16 = x.half()
    assert torch.equal(op.forward_oot(fp16), torch_npu.npu_swiglu(fp16))


@pytest.mark.parametrize("tokens,width", [(0, 13), (1, 13), (17, 2560)])
def test_gate_shapes_and_scalar_boundary(tokens, width):
    # Constant norms give exactly representable products and dot, allowing a
    # scalar reference independent of the kernel's reduction implementation.
    key = torch.ones(tokens, 4 * width, device="npu", dtype=torch.bfloat16)
    value = torch.ones(tokens, width, device="npu", dtype=torch.bfloat16)
    weights = [torch.zeros(4 * width, device="npu", dtype=torch.bfloat16) for _ in range(3)]
    gated, normed = ple_gate(key, value, key, *weights, 1e-6)
    assert gated.shape == normed.shape == key.shape
    if tokens:
        dot = torch.tensor(float(width)).bfloat16().double()
        scaled = (dot / width**0.5).bfloat16().double()
        root = scaled.sqrt().bfloat16().double()
        gate = root.sigmoid().bfloat16()
        assert torch.equal(gated.cpu(), gate.expand_as(gated.cpu()))
        assert torch.isfinite(normed).all()
