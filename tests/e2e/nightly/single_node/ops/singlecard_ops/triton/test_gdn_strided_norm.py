# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.ops.layernorm import LayerNormFn


@pytest.mark.parametrize("tokens", [1, 4, 17])
@pytest.mark.parametrize("activation", ["swish", "sigmoid"])
@pytest.mark.parametrize("before", [False, True])
@torch.inference_mode()
def test_gdn_strided_z_matches_packed_and_reference(tokens, activation, before):
    torch.manual_seed(42)
    qkv_width = 5120
    num_v_heads = 24
    head_v_dim = 128
    mixed_qkvz = torch.randn(
        tokens,
        qkv_width + num_v_heads * head_v_dim,
        device="npu",
        dtype=torch.bfloat16,
    )
    z = mixed_qkvz[:, qkv_width:].reshape(tokens, num_v_heads, head_v_dim)
    x = torch.randn(tokens, num_v_heads, head_v_dim, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(head_v_dim, device="npu", dtype=torch.bfloat16)

    actual = LayerNormFn.apply(x, weight, None, z, 1e-5, None, before, True, activation)
    packed = LayerNormFn.apply(
        x.reshape(-1, head_v_dim),
        weight,
        None,
        z.contiguous().reshape(-1, head_v_dim),
        1e-5,
        None,
        before,
        True,
        activation,
    ).reshape_as(x)
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
