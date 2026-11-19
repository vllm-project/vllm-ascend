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


def test_mega_mhc_post_pre_maps_glm_inputs_and_outputs(monkeypatch):
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    nn.Module.__init__(layer)
    layer.n = 4
    layer.mhc_sinkhorn_iterations = 20
    layer.rms_norm_eps = 1e-6
    layer.hc_eps = 2e-6
    layer.mhc_post_mult_value = 2.0
    layer._use_mega_mhc = True

    x = torch.randn(3, 8).bfloat16()
    residual = torch.randn(3, 4, 8).bfloat16()
    post = torch.randn(3, 4, 1)
    comb = torch.randn(3, 4, 4)
    fn = torch.randn(24, 32)
    scale = torch.tensor([0.5, 0.6, 0.7])
    base = torch.randn(24)
    norm_weight = torch.linspace(0.5, 1.5, 8).bfloat16()
    norm_eps = 1e-5
    expected = (
        torch.randn_like(residual),
        torch.randn_like(post),
        torch.randn_like(comb),
        torch.randn_like(x),
    )
    captured = {}

    def mega_mhc(*args):
        captured["args"] = args
        return expected

    monkeypatch.setattr(torch.ops._C_ascend, "npu_mega_mhc", mega_mhc, raising=False)
    monkeypatch.setattr(layer, "hc_post", lambda *args: pytest.fail("legacy hc_post should not be called"))
    monkeypatch.setattr(layer, "hc_pre", lambda *args, **kwargs: pytest.fail("legacy hc_pre should not be called"))

    actual = layer.hc_post_pre(x, residual, post, comb, fn, scale, base, norm_weight, norm_eps)

    assert all(actual_value is expected_value for actual_value, expected_value in zip(actual, expected))
    args = captured["args"]
    assert all(actual is expected for actual, expected in zip(args[:7], (x, residual, post, comb, fn, scale, base)))
    assert args[7].dtype == torch.float32
    torch.testing.assert_close(args[7], norm_weight.float())
    assert args[8:] == (4, 20, norm_eps, layer.rms_norm_eps, layer.hc_eps, layer.mhc_post_mult_value)


@pytest.mark.parametrize(
    ("mega_enabled", "with_norm", "x_dtype"),
    [
        (False, True, torch.bfloat16),
        (True, False, torch.bfloat16),
        (True, True, torch.float16),
        (True, True, torch.float32),
    ],
)
def test_mega_mhc_post_pre_preserves_legacy_fallback(monkeypatch, mega_enabled, with_norm, x_dtype):
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    nn.Module.__init__(layer)
    layer._use_mega_mhc = mega_enabled

    x = torch.randn(3, 8, dtype=x_dtype)
    residual = torch.randn(3, 4, 8, dtype=x_dtype)
    post = torch.randn(3, 4, 1)
    comb = torch.randn(3, 4, 4)
    fn = torch.randn(24, 32)
    scale = torch.tensor([0.5, 0.6, 0.7])
    base = torch.randn(24)
    norm_weight = torch.randn(8) if with_norm else None
    expected = (residual.clone(), post.clone(), comb.clone(), x.clone())
    calls = []

    def legacy_post(*args):
        calls.append("post")
        return expected[0]

    def legacy_pre(*args, **kwargs):
        calls.append("pre")
        assert kwargs["norm_weight"] is norm_weight
        return expected[1:]

    monkeypatch.setattr(layer, "hc_post", legacy_post)
    monkeypatch.setattr(layer, "hc_pre", legacy_pre)
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_mega_mhc",
        lambda *args: pytest.fail("MegaMhc should not be called on the fallback path"),
        raising=False,
    )

    actual = layer.hc_post_pre(x, residual, post, comb, fn, scale, base, norm_weight, 1e-5)

    assert all(actual_value is expected_value for actual_value, expected_value in zip(actual, expected))
    assert calls == ["post", "pre"]
