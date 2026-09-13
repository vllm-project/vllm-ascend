# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from tests.deepseek_v41_reference import hc_post_reference
from vllm_ascend.models.deepseek_v41.model import DeepseekV41DecoderLayer

HC_MULT = 4
HIDDEN_SIZE = 5120
SINKHORN_ITERS = 3
NORM_EPS = 1e-6
HC_EPS = 1e-6


def _layer() -> DeepseekV41DecoderLayer:
    layer = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
    layer.hc_mult = HC_MULT
    layer.hc_sinkhorn_iters = SINKHORN_ITERS
    layer.norm_eps = NORM_EPS
    layer.hc_eps = HC_EPS
    return layer


def _reference(x, hc_fn, hc_scale, hc_base, pre_mix):
    x_float = x.float()
    x_flat = x_float.flatten(-2)
    mixes = F.linear(x_flat, hc_fn) * torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)
    pre, post, comb = mixes.split([HC_MULT, HC_MULT, HC_MULT * HC_MULT], dim=-1)
    comb = comb.unflatten(-1, (HC_MULT, HC_MULT))
    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[HC_MULT : 2 * HC_MULT])
    comb = comb * hc_scale[2] + hc_base[2 * HC_MULT :].view(HC_MULT, HC_MULT)
    comb = comb.softmax(-1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    y = (pre_mix.unsqueeze(-1) * x_float).sum(dim=-2).to(x.dtype)
    return y, post, comb, pre


def test_v41_hc_pre_handoff_5120_on_npu():
    torch.manual_seed(19)
    x = torch.randn(2, HC_MULT, HIDDEN_SIZE, dtype=torch.bfloat16)
    hc_fn = torch.randn(24, HC_MULT * HIDDEN_SIZE, dtype=torch.float32) / HIDDEN_SIZE
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(24, dtype=torch.float32)
    pre_mix = torch.rand(2, HC_MULT, dtype=torch.float32)
    expected = _reference(x, hc_fn, hc_scale, hc_base, pre_mix)

    actual = _layer().hc_pre(x.npu(), hc_fn.npu(), hc_scale.npu(), hc_base.npu(), pre_mix.npu())

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor.cpu().float(),
            expected_tensor.float(),
            atol=5e-3,
            rtol=5e-3,
        )

    expected_post = hc_post_reference(expected[0], x, expected[1], expected[2])
    actual_post = _layer().hc_post(actual[0], x.npu(), actual[1], actual[2])
    torch.testing.assert_close(actual_post.cpu().float(), expected_post.float(), atol=2e-2, rtol=2e-2)
