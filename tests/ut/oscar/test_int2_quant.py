# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the OSCAR INT2 encoder (production quantize path).

This is the path the frozen RC53 serve chain actually runs (pure Torch,
per-token-head affine INT2 with quantile clipping). The C++ single-call
variant in the launcher extension is value-identical (at::kthvalue(k) ==
sorted[k-1]); the threshold-equality test below documents that contract.
"""

import pytest
import torch

from vllm_ascend.oscar.int2_quant import (
    DEFAULT_K_CLIP_RATIO,
    DEFAULT_V_CLIP_RATIO,
    oscar_dequantize_int2,
    oscar_quantize_int2,
)


def _rand_kv(t=10, h=8, d=128, seed=42):
    torch.manual_seed(seed)
    return torch.randn(t, h, d, dtype=torch.float32)


class TestShapesAndDtypes:
    def test_output_shapes(self):
        x = _rand_kv()
        packed, scales, zeros = oscar_quantize_int2(x)
        assert packed.shape == (10, 8, 32)
        assert packed.dtype == torch.uint8
        assert scales.shape == zeros.shape == (10, 8)
        assert scales.dtype == torch.float32 and zeros.dtype == torch.float32

    def test_pack_layout_exact(self):
        # Layout contract (mirrors the C++ at:: variant byte-for-byte):
        # head_dim is split into 4 CONTIGUOUS quarters; byte r encodes
        # q0[r] | q1[r]<<2 | q2[r]<<4 | q3[r]<<6 where lane j is the j-th
        # quarter. x=[0,0, 1,1, 2,2, 3,3] with range [0,3] (scale=1, zero=0)
        # must pack every byte to 0 | 1<<2 | 2<<4 | 3<<6 = 228.
        x = torch.zeros(1, 1, 8)
        x[0, 0, :] = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        packed, scales, zeros = oscar_quantize_int2(x)
        assert torch.all(packed[0, 0] == 228)
        # and the unpacker inverts it lane-faithfully
        xd = oscar_dequantize_int2(packed, scales, zeros, torch.float32)
        assert torch.equal(xd[0, 0], x[0, 0])

    def test_head_dim_must_divide_4(self):
        with pytest.raises(AssertionError):
            oscar_quantize_int2(torch.randn(1, 1, 6))


class TestNumerics:
    def test_roundtrip_accuracy(self):
        x = _rand_kv()
        packed, scales, zeros = oscar_quantize_int2(x)
        xd = oscar_dequantize_int2(packed, scales, zeros, torch.float32)
        cos = torch.nn.functional.cosine_similarity(x.view(-1), xd.view(-1), dim=0)
        # 4-level affine per token-head on fp32 randn: cos ~0.93-0.96;
        # serve-level needle tests are the quality acceptance, this bounds
        # the encoder against gross defects (sign flips, lane swaps).
        assert cos.item() > 0.90
        # reconstruction error is bounded by the affine half-step
        assert (x - xd).abs().max().item() < 2.0 * scales.max().item()

    def test_codes_in_range(self):
        x = _rand_kv()
        packed, _, _ = oscar_quantize_int2(x)
        # every 2-bit lane is in [0,3] by construction: byte < 256 suffices
        assert packed.max() <= 255 and packed.min() >= 0

    def test_constant_row_degenerate_scale(self):
        # constant row: x_max - x_min = 0 -> range clamps to 1e-8 BEFORE the
        # /3 affine divisor, so scale = 1e-8/3; finite everywhere, no NaN
        x = torch.full((1, 1, 8), 2.5)
        packed, scales, zeros = oscar_quantize_int2(x)
        assert torch.isfinite(scales).all() and torch.isfinite(zeros).all()
        assert torch.allclose(scales, torch.full_like(scales, 1e-8 / 3.0))

    def test_outlier_clipped(self):
        # one extreme outlier must not blow the affine range: with clipping
        # at 0.96 quantile the range is set by the clipped body, and the
        # outlier dequantizes back to ~+-threshold
        x = _rand_kv(t=1, h=1, d=128, seed=7)
        x[0, 0, 0] = 1000.0
        packed, scales, zeros = oscar_quantize_int2(x, clip_ratio=0.96)
        xd = oscar_dequantize_int2(packed, scales, zeros, torch.float32)
        assert xd[0, 0, 0].abs().item() < 100.0  # not representable blow-up
        assert torch.isfinite(xd).all()

    def test_threshold_kthvalue_equals_sorted(self):
        # documents the C++-variant value-identity contract: at::kthvalue(k)
        # returns the same threshold value as ascending sort index k-1.
        x = _rand_kv()
        clip_idx = int(0.96 * x.shape[-1])
        abs_x = x.abs()
        thr_kth = torch.kthvalue(abs_x, clip_idx, dim=-1, keepdim=True)[0]
        thr_sorted = abs_x.sort(dim=-1)[0][:, :, clip_idx - 1].unsqueeze(-1)
        assert torch.equal(thr_kth, thr_sorted)


class TestDefaults:
    def test_oscar_clip_constants(self):
        # frozen-tuned clip ratios (K tighter than V) must not drift
        assert DEFAULT_K_CLIP_RATIO == 0.96
        assert DEFAULT_V_CLIP_RATIO == 0.92
