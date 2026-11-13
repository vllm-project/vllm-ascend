# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.ops.layernorm import LayerNormFn


@pytest.mark.parametrize("tokens", [1, 8, 17])
@pytest.mark.parametrize("activation", ["swish", "sigmoid"])
@pytest.mark.parametrize("before", [False, True])
@torch.inference_mode()
def test_gdn_strided_z_matches_packed_and_reference(tokens, activation, before):
    torch.manual_seed(42)
    backing = torch.randn(tokens + 2, 128, 128, device="npu", dtype=torch.bfloat16)
    z = backing[1 : tokens + 1, 80:128]
    x = torch.randn(tokens, 48, 128, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(128, device="npu", dtype=torch.bfloat16)
    actual = LayerNormFn.apply(x, weight, None, z, 1e-5, None, before, True, activation)
    packed = LayerNormFn.apply(x, weight, None, z.contiguous(), 1e-5, None, before, True, activation)
    torch.testing.assert_close(actual, packed, rtol=0, atol=0)
    xf, zf, wf = x.cpu().float(), z.cpu().float(), weight.cpu().float()
    gate = zf.sigmoid()
    if activation == "swish":
        gate *= zf
    if not before:
        xf *= gate
    expected = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5) * wf
    if before:
        expected *= gate
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=0.02, atol=0.02)
