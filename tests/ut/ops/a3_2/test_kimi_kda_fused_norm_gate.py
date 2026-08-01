# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.kda.fused_norm_gate import apply_kda_rms_norm_sigmoid_gate


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kimi_kda_fused_rms_norm_sigmoid_gate(dtype: torch.dtype):
    torch.manual_seed(20260801)
    tokens, heads, head_dim = 37, 4, 128
    eps = 1e-6
    core_attn_out = torch.randn(
        1,
        tokens,
        heads,
        head_dim,
        dtype=dtype,
        device="npu",
    )
    output_gate = torch.randn(
        tokens,
        heads,
        head_dim,
        dtype=dtype,
        device="npu",
    )
    weight = torch.randn(head_dim, dtype=dtype, device="npu")

    actual = apply_kda_rms_norm_sigmoid_gate(
        core_attn_out,
        output_gate,
        weight,
        eps,
    )

    x_float = core_attn_out.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    expected = x_float * torch.rsqrt(variance + eps)
    expected = expected * weight.float()
    expected = expected * output_gate.float().sigmoid().unsqueeze(0)
    expected = expected.to(dtype)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
