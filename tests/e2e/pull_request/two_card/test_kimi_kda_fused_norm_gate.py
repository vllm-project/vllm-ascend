# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.kda.kda import layer_norm_gated_fwd, rms_norm_gated


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


@torch.inference_mode()
@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 32, 37, 64, 768])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [False, True])
def test_kimi_norm_gate_strides_and_direct_output(tokens, dtype, inplace):
    """Read projection gaps and preserve caller-owned padding across replays."""
    heads, dim = 12, 128
    x_storage = torch.randn(1, tokens + 2, heads * 2, dim, dtype=dtype, device="npu")
    x = x_storage[:, :tokens, ::2]
    projection = torch.randn(tokens, heads + 2 * heads * dim, dtype=dtype, device="npu")
    gate = projection[:, heads + heads * dim :].view(tokens, heads, dim)
    weight = torch.randn(dim, dtype=dtype, device="npu")
    output_storage = torch.full_like(x_storage, 7)
    out = x if inplace else output_storage[:, :tokens, ::2]

    def reference():
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        return (value * weight.float() * gate.float().sigmoid()).to(dtype)

    expected = reference()
    legacy, *_ = layer_norm_gated_fwd(
        x.contiguous().reshape(-1, dim),
        gate.contiguous().reshape(-1, dim),
        weight,
        None,
        activation="sigmoid",
        eps=1e-6,
        out_dtype=x.dtype,
        is_rms_norm=True,
    )
    actual = rms_norm_gated(x, gate, weight, None, "sigmoid", out=out)
    assert actual is out
    torch.testing.assert_close(actual, legacy.reshape(x.shape), rtol=0, atol=0)
    # BF16 rounding at a midpoint may differ by one ULP from the torch FP32
    # reference, while the original and strided Triton paths must agree exactly.
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=2e-3)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        rms_norm_gated(x, gate, weight, None, "sigmoid", out=out)
    for step in range(3):
        x.fill_(0.25 * (step + 1))
        gate.fill_(0.5 * step)
        expected = reference()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)
    if not inplace:
        torch.testing.assert_close(output_storage[:, tokens:], torch.full_like(output_storage[:, tokens:], 7))
        torch.testing.assert_close(output_storage[:, :tokens, 1::2], torch.full_like(out, 7))
