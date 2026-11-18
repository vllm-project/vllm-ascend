# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import pytest
import torch


@pytest.mark.parametrize("tokens,width", [(0, 13), (1, 13), (17, 2560)])
def test_materialized_gate(tokens, width):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires an allocated Ascend NPU")
    from ple_gate import ple_gate_materialized

    generator = torch.Generator().manual_seed(42)
    hc = 4
    kv = torch.randn(tokens, (hc + 1) * width, generator=generator).bfloat16().npu()
    hidden = torch.randn(tokens, hc * width, generator=generator).bfloat16().npu()
    weights = [torch.zeros(hc * width, dtype=torch.bfloat16, device="npu") for _ in range(3)]
    before = hidden.clone()
    gated, normed, scalars = ple_gate_materialized(kv[:, : hc * width], kv[:, hc * width :], hidden, *weights)
    assert torch.equal(hidden, before)
    assert gated.shape == normed.shape == hidden.shape
    assert all(s.dtype == torch.bfloat16 and s.shape == (tokens, hc) for s in scalars)
    if tokens:
        dot, scaled, magnitude, sigmoid = [s.cpu().double() for s in scalars]
        # Check each stored scalar against an independent FP64 reference using
        # the actual preceding BF16 boundary, not a different reduction order.
        assert torch.equal((dot / width**0.5).bfloat16(), scaled.bfloat16())
        expected_magnitude = scaled.sign() * scaled.abs().clamp_min(1e-6).sqrt()
        assert torch.equal(expected_magnitude.bfloat16(), magnitude.bfloat16())
        assert torch.equal(magnitude.sigmoid().bfloat16(), sigmoid.bfloat16())
        expected_gated = (sigmoid[:, :, None] * kv[:, hc * width :].cpu().double()[:, None, :]).bfloat16()
        assert torch.equal(gated.cpu().view(tokens, hc, width), expected_gated)
        assert torch.isfinite(normed).all()
