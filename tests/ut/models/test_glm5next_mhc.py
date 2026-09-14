# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu
from torch import nn
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch

from vllm_ascend.models.glm5next.model import Glm5NextDecoderLayer


def _rms_norm(x, weight, epsilon):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + epsilon) * weight.float()).to(x.dtype)


@pytest.mark.parametrize("post_scale", [2.0, 1.5])
@pytest.mark.parametrize("with_norm", [False, True])
def test_mhc_native_ops_preserve_deferred_mixing_and_input_norm(monkeypatch, post_scale, with_norm):
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    nn.Module.__init__(layer)
    layer.n = 4
    layer.mhc_sinkhorn_iterations = 20
    layer.rms_norm_eps = 1e-6
    layer.hc_eps = 1e-6
    layer.mhc_post_mult_value = post_scale
    torch.manual_seed(0)
    residual = torch.randn(3, 4, 8).bfloat16()
    original = residual.clone()
    fn = torch.randn(24, 32) * 0.1
    scale = torch.tensor([0.5, 0.6, 0.7])
    base = torch.randn(24) * 0.1
    weight = torch.linspace(0.5, 1.5, 8).bfloat16() if with_norm else None
    norm_eps = 1e-5

    def native_pre(x, fn, scale, base, hc_mult, iters, rms_eps, hc_eps):
        assert hc_mult == 4
        post, comb, y = mhc_pre_torch(x, fn, scale, base, rms_eps, hc_eps, hc_eps, 2.0, iters)
        return y, post.squeeze(-1), comb

    def native_post(x, residual, post, comb):
        assert post.ndim == 3 and residual.ndim == 4
        return mhc_post_torch(x, residual, post.unsqueeze(-1), comb)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_hc_pre_v2", native_pre, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_hc_post", native_post, raising=False)
    monkeypatch.setattr(torch_npu, "npu_rms_norm", lambda x, weight, epsilon: (_rms_norm(x, weight, epsilon), None))

    expected_post, expected_comb, expected_y = mhc_pre_torch(
        residual, fn, scale, base, layer.rms_norm_eps, layer.hc_eps, layer.hc_eps, post_scale, 20
    )
    if with_norm:
        expected_y = _rms_norm(expected_y, weight, norm_eps)
    post, comb, y = layer.hc_pre(residual, fn, scale, base, norm_weight=weight, norm_eps=norm_eps)
    for actual, expected in zip((post, comb, y), (expected_post, expected_comb, expected_y)):
        torch.testing.assert_close(actual, expected)

    mixed = mhc_post_torch(expected_y, residual, expected_post, expected_comb)
    next_post, next_comb, next_y = mhc_pre_torch(
        mixed, fn, scale, base, layer.rms_norm_eps, layer.hc_eps, layer.hc_eps, post_scale, 20
    )
    if with_norm:
        next_y = _rms_norm(next_y, weight, norm_eps)
    actual = layer.hc_post_pre(y, residual, post, comb, fn, scale, base, norm_weight=weight, norm_eps=norm_eps)
    for value, expected in zip(actual, (mixed, next_post, next_comb, next_y)):
        torch.testing.assert_close(value, expected)
    torch.testing.assert_close(residual, original)
