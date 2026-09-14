# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.kda.kda import rms_norm_gated


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens, heads, head_dim", [(1, 1, 128), (16, 4, 128), (37, 4, 128), (2, 1, 1024)])
@pytest.mark.parametrize("strided_gate", [False, True])
def test_kimi_kda_fused_rms_norm_sigmoid_gate(dtype, tokens, heads, head_dim, strided_gate):
    torch.manual_seed(20260801)
    eps = 1e-6
    weight = torch.randn(head_dim, dtype=dtype, device="npu")
    core_attn_out = torch.randn(1, tokens, heads, head_dim, dtype=dtype, device="npu")
    output_gate = torch.randn(tokens, heads, head_dim, dtype=dtype, device="npu")
    if strided_gate:
        # K3's packed projection leaves gaps between consecutive gate rows.
        packed_gate = torch.full((tokens, 2 * heads, head_dim), torch.nan, dtype=dtype, device="npu")
        packed_gate[:, :heads].copy_(output_gate)
        output_gate = packed_gate[:, :heads]
    core_attn_out_before = core_attn_out.clone()
    output_gate_before = output_gate.clone()

    actual = rms_norm_gated(core_attn_out, output_gate, weight, None, activation="sigmoid", eps=eps)

    x_float = core_attn_out_before.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    expected = x_float * torch.rsqrt(variance + eps)
    expected = expected * weight.float()
    expected = expected * output_gate.float().sigmoid().unsqueeze(0)

    torch.testing.assert_close(actual, expected.to(dtype), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(core_attn_out, core_attn_out_before, rtol=0, atol=0)
    torch.testing.assert_close(output_gate, output_gate_before, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("residual_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_fused_rms_norm_silu_gate_preserves_prenorm_contract(residual_dtype, elementwise_affine):
    torch.manual_seed(20260801)
    x = torch.randn(1, 3, 2, 128, dtype=torch.bfloat16, device="npu")
    gate = torch.randn_like(x)
    residual = torch.randn_like(x, dtype=residual_dtype) if residual_dtype is not None else None
    weight = torch.randn(128, dtype=x.dtype, device="npu") if elementwise_affine else None
    before = x.clone()
    eps = 1e-6

    actual, residual_out = rms_norm_gated(
        x, gate, weight, None, activation="silu", residual=residual, prenorm=True, residual_in_fp32=True, eps=eps
    )

    summed = before.float() if residual is None else before.float() + residual.float()
    expected = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        expected *= weight.float()
    expected *= gate.float() * gate.float().sigmoid()
    expected_residual_dtype = torch.float32 if residual is None else residual.dtype

    torch.testing.assert_close(actual, expected.to(x.dtype), rtol=2e-3, atol=2e-3)
    assert residual_out.dtype == expected_residual_dtype
    torch.testing.assert_close(residual_out, summed.to(expected_residual_dtype), rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
