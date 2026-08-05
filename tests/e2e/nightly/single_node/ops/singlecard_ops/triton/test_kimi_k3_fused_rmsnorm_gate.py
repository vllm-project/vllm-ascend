# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.ops.triton.layernorm_gated import layer_norm_fwd_npu


def test_kimi_k3_fused_rmsnorm_sigmoid_gate():
    torch.manual_seed(0)
    eps = 1e-5
    x = torch.randn(33, 128, dtype=torch.bfloat16).npu()
    gate = torch.randn_like(x)
    weight = torch.randn(128, dtype=torch.bfloat16).npu()

    actual, _, _ = layer_norm_fwd_npu(
        x,
        weight,
        None,
        eps,
        z=gate,
        norm_before_gate=True,
        is_rms_norm=True,
        activation="sigmoid",
    )

    x_fp32 = x.float()
    expected = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + eps)
    expected = (expected * weight.float() * gate.float().sigmoid()).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
