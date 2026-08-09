# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend._310p.ops.hyper_connection import (
    hc_post_310p,
    hc_pre_310p,
    hc_split_sinkhorn_310p,
)


def _reference_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
):
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = mixes[..., 2 * hc_mult :].reshape(*mixes.shape[:-1], hc_mult, hc_mult)
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def test_hc_split_sinkhorn_matches_reference() -> None:
    torch.manual_seed(7)
    hc_mult = 4
    mixes = torch.randn(3, (2 + hc_mult) * hc_mult)
    scale = torch.randn(3)
    base = torch.randn((2 + hc_mult) * hc_mult)
    actual = hc_split_sinkhorn_310p(mixes, scale, base, hc_mult, 5, 1e-6)
    expected = _reference_sinkhorn(mixes, scale, base, hc_mult, 5, 1e-6)
    for lhs, rhs in zip(actual, expected, strict=True):
        torch.testing.assert_close(lhs, rhs)


def test_hc_pre_and_post_match_reference_shapes_and_values() -> None:
    torch.manual_seed(11)
    tokens, hc_mult, hidden = 3, 4, 8
    x = torch.randn(tokens, hc_mult, hidden, dtype=torch.float16)
    hc_fn = torch.randn((2 + hc_mult) * hc_mult, hc_mult * hidden)
    hc_scale = torch.randn(3)
    hc_base = torch.randn((2 + hc_mult) * hc_mult)

    y, post, comb = hc_pre_310p(x, hc_fn, hc_scale, hc_base, hc_mult, 4, 1e-6, 1e-6)
    x_float = x.float()
    flat = x_float.flatten(1)
    mixes = torch.nn.functional.linear(flat, hc_fn)
    mixes *= torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
    pre_ref, post_ref, comb_ref = _reference_sinkhorn(mixes, hc_scale, hc_base, hc_mult, 4, 1e-6)
    y_ref = (pre_ref.unsqueeze(-1) * x_float).sum(-2).to(x.dtype)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(post, post_ref)
    torch.testing.assert_close(comb, comb_ref)

    branch = torch.randn(tokens, hidden, dtype=x.dtype)
    out = hc_post_310p(branch, x, post, comb)
    out_ref = post.unsqueeze(-1) * branch.unsqueeze(-2)
    out_ref += (comb.unsqueeze(-1) * x.unsqueeze(-2)).sum(-3)
    torch.testing.assert_close(out, out_ref.to(out.dtype))
    assert out.shape == x.shape
