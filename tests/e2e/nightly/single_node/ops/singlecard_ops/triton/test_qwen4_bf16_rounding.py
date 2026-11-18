# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Rounding boundaries in the actual Ascend runtime entry points."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.ops.activation import AscendSiluAndMul
from vllm_ascend.ops.triton_ple import ple_conv_decode, ple_conv_prefill, ple_conv_spec, ple_gate


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


@pytest.mark.parametrize("mode", ["prefill", "decode", "spec"])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_residual_rounding_and_state(mode, mapped, alias):
    torch.manual_seed(42)
    channels, count = 13, 3
    total = 7 if mapped else count
    indices = torch.tensor([1, 3, 5], device="npu", dtype=torch.int32) if mapped else None
    x = torch.randn(total, channels, device="npu", dtype=torch.bfloat16)
    weights = torch.randn(channels, 4, device="npu", dtype=torch.bfloat16)
    initial = torch.randn(total, channels, device="npu", dtype=torch.bfloat16)
    residual = initial.clone()
    outer = residual if alias else torch.randn_like(residual)
    original_outer = outer.clone()
    state = torch.randn(5, 6, channels, device="npu", dtype=torch.bfloat16)
    reference_state = state.clone()
    slots = torch.tensor([1, 2, 3] if mode == "decode" else [1], device="npu", dtype=torch.int32)
    kwargs = dict(dilation=1, token_indices=indices)
    if mode != "decode":
        kwargs["query_start_loc"] = torch.tensor([0, count], device="npu", dtype=torch.int32)
        kwargs["has_initial_states"] = torch.ones(1, device="npu", dtype=torch.bool)
    if mode == "spec":
        kwargs["num_accepted_tokens"] = torch.tensor([1], device="npu", dtype=torch.int32)
        kwargs["spec_query_len"] = count
    fn = {"prefill": ple_conv_prefill, "decode": ple_conv_decode, "spec": ple_conv_spec}[mode]
    activated = torch.empty_like(initial)
    fn(x, activated, reference_state, weights, slots, None, **kwargs)
    fn(x, residual, state, weights, slots, outer, **kwargs)
    rows = indices.long() if mapped else torch.arange(count, device="npu")
    expected = (initial[rows].float() + activated[rows].float()).bfloat16()
    expected = (expected.float() + original_outer[rows].float()).bfloat16()
    assert torch.equal(residual[rows], expected)
    assert torch.equal(state, reference_state)
    if mapped:
        assert torch.equal(residual[::2], initial[::2])
