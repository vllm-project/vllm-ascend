# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Regression tests for the npu_hc_pre Sinkhorn column epsilon alignment bug (issue #19357).

The Sinkhorn column normalization of ``npu_hc_pre_v3``/``npu_hc_pre_v2`` added ``hc_eps``
to the column denominators over ``hc_mult`` elements while the column sums live in an
``hc_mult_align``-strided aligned layout, so token rows in the tail of a tile missed the
epsilon contribution. The visible symptom: within a single kernel call, identical inputs at
different token positions produced different ``comb`` values — a deterministic ~5e-7
difference that the usual pass-rate checks with a 1e-4 threshold cannot catch.

These tests therefore check row-position invariance: with the GEMM contribution zeroed
out (``hc_fn`` and ``hc_scale`` all zeros) and every token row of ``x``/``pre_mix``
identical, every output row must be bitwise identical to row 0 (``torch.equal``, not a
tolerance-based comparison). The token-count sweep crosses the ``SMALL_BS_THRESHOLD``
tiling boundary and tile edges, and the iteration sweep covers 1 to 20 Sinkhorn
iterations. One test additionally checks the fixed operator against the CPU reference
implementation with the usual pass-rate tolerances.

Reference (historical) tests:
``tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_hc_pre.py``.
"""

from __future__ import annotations

import gc

import pytest
import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401

    NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
except ImportError:
    NPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="npu_hc_pre regression tests require an Ascend NPU device (issue #19357).",
)

HC_MULT = 4
MIX_HC = 24
HC_SINKHORN_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6
DSV4_FLASH_HIDDEN_SIZE = 4096
EXTENDED_HIDDEN_SIZE = 5120
# Crosses SMALL_BS_THRESHOLD (bs <= 32 hits the ProcessOriginal copy) and tile edges.
TOKEN_COUNT_SWEEP = (1, 3, 8, 16, 32, 33, 65, 128, 129)
SINKHORN_ITERS_SWEEP = (1, 2, 4, 20)
HC_BASE_KINDS = ("zeros", "normal", "small")
# Representative points for the hc_base sweep; the 5120 case covers a second hidden size.
HC_BASE_SWEEP_CASES = (
    (1, 1, DSV4_FLASH_HIDDEN_SIZE),
    (32, 2, DSV4_FLASH_HIDDEN_SIZE),
    (33, HC_SINKHORN_ITERS, DSV4_FLASH_HIDDEN_SIZE),
    (33, HC_SINKHORN_ITERS, EXTENDED_HIDDEN_SIZE),
    (129, 4, DSV4_FLASH_HIDDEN_SIZE),
)
CPU_REFERENCE_NUM_TOKENS = 33
HF32_MANTISSA_BITS = 10
FP32_MANTISSA_BITS = 23
Y_DIFF_THRESHOLD = 4e-3
Y_REQUIRED_PASS_RATE = 0.98
AUX_DIFF_THRESHOLD = 1e-4
AUX_REQUIRED_PASS_RATE = 0.995


def _enable_hc_pre_custom_op() -> None:
    """Enable the vllm-ascend custom-op environment, like the nightly hc_pre tests."""
    import torch_npu

    torch_npu.npu.config.allow_internal_format = True

    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()


def _make_hc_base(kind: str) -> torch.Tensor:
    """Build one of the three hc_base regimes: all zeros, normal, or small random."""
    torch.manual_seed(1024)
    if kind == "zeros":
        return torch.zeros(MIX_HC, dtype=torch.float32)
    if kind == "small":
        return 0.001 * torch.randn(MIX_HC, dtype=torch.float32)
    assert kind == "normal"
    return torch.randn(MIX_HC, dtype=torch.float32)


def _make_identical_row_inputs(
    num_tokens: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inputs whose every token row is identical and whose GEMM contribution is zero.

    ``hc_fn`` and ``hc_scale`` are all zeros so ``comb`` depends only on ``hc_base`` and
    the Sinkhorn normalization, while every token row of ``x``/``pre_mix`` is a broadcast
    of the same row. The kernel then has no legitimate reason to produce
    row-position-dependent outputs.
    """
    torch.manual_seed(1024)
    x_row = torch.randn(HC_MULT, hidden_size, dtype=torch.float32).to(torch.bfloat16)
    x = x_row.unsqueeze(0).expand(num_tokens, HC_MULT, hidden_size).contiguous()
    hc_fn = torch.zeros(MIX_HC, HC_MULT * hidden_size, dtype=torch.float32)
    hc_scale = torch.zeros(3, dtype=torch.float32)
    pre_mix_row = torch.rand(HC_MULT, dtype=torch.float32)
    pre_mix = pre_mix_row.unsqueeze(0).expand(num_tokens, HC_MULT).contiguous()
    return x, hc_fn, hc_scale, pre_mix


def _assert_rows_bitwise_identical(name: str, tensor: torch.Tensor) -> None:
    """Assert every row of ``tensor`` is bitwise equal to its first row."""
    first_row = tensor[0:1].expand_as(tensor)
    if torch.equal(tensor, first_row):
        return
    mismatched_elements = (tensor != first_row).sum().item()
    max_abs_diff = (tensor.float() - first_row.float()).abs().max().item()
    raise AssertionError(
        f"{name} differs across token positions within one npu_hc_pre call "
        f"(row-position invariance violated, issue #19357): "
        f"{mismatched_elements} mismatched elements, max abs diff {max_abs_diff:.3e}, "
        f"shape {tuple(tensor.shape)}"
    )


def _check_v3_row_position_invariance(
    num_tokens: int,
    hidden_size: int,
    hc_sinkhorn_iters: int,
    hc_base: torch.Tensor,
) -> None:
    x, hc_fn, hc_scale, pre_mix = _make_identical_row_inputs(num_tokens, hidden_size)
    y, post, comb, pre = torch.ops._C_ascend.npu_hc_pre_v3(
        x.npu(),
        hc_fn.npu(),
        hc_scale.npu(),
        hc_base.npu(),
        pre_mix.npu(),
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
    )
    assert y.shape == (num_tokens, hidden_size)
    assert post.shape == (num_tokens, HC_MULT)
    assert comb.shape == (num_tokens, HC_MULT, HC_MULT)
    assert pre.shape == (num_tokens, HC_MULT)
    _assert_rows_bitwise_identical("comb", comb)
    _assert_rows_bitwise_identical("post", post)
    _assert_rows_bitwise_identical("pre", pre)


def _check_v2_row_position_invariance(
    num_tokens: int,
    hidden_size: int,
    hc_sinkhorn_iters: int,
    hc_base: torch.Tensor,
) -> None:
    x, hc_fn, hc_scale, _ = _make_identical_row_inputs(num_tokens, hidden_size)
    y, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
        x.npu(),
        hc_fn.npu(),
        hc_scale.npu(),
        hc_base.npu(),
        HC_MULT,
        hc_sinkhorn_iters,
        NORM_EPS,
        HC_EPS,
    )
    assert y.shape == (num_tokens, hidden_size)
    assert post.shape == (num_tokens, HC_MULT)
    assert comb.shape == (num_tokens, HC_MULT, HC_MULT)
    _assert_rows_bitwise_identical("comb", comb)
    _assert_rows_bitwise_identical("post", post)


# ---------------------------------------------------------------------------
# Row-position invariance (issue #19357 regression)
# ---------------------------------------------------------------------------


@torch.inference_mode()
@pytest.mark.parametrize("hc_sinkhorn_iters", SINKHORN_ITERS_SWEEP)
@pytest.mark.parametrize("num_tokens", TOKEN_COUNT_SWEEP)
def test_npu_hc_pre_v3_row_position_invariance(num_tokens: int, hc_sinkhorn_iters: int) -> None:
    """Identical token rows must give bitwise identical outputs, for any row position."""
    _enable_hc_pre_custom_op()
    _check_v3_row_position_invariance(
        num_tokens,
        DSV4_FLASH_HIDDEN_SIZE,
        hc_sinkhorn_iters,
        _make_hc_base("normal"),
    )
    gc.collect()
    torch.npu.empty_cache()


@torch.inference_mode()
@pytest.mark.parametrize(
    ("num_tokens", "hc_sinkhorn_iters", "hidden_size"),
    HC_BASE_SWEEP_CASES,
)
@pytest.mark.parametrize("hc_base_kind", HC_BASE_KINDS)
def test_npu_hc_pre_v3_row_position_invariance_hc_base_sweep(
    num_tokens: int,
    hc_sinkhorn_iters: int,
    hidden_size: int,
    hc_base_kind: str,
) -> None:
    """Row-position invariance must hold for zeros, normal, and small hc_base regimes."""
    _enable_hc_pre_custom_op()
    _check_v3_row_position_invariance(
        num_tokens,
        hidden_size,
        hc_sinkhorn_iters,
        _make_hc_base(hc_base_kind),
    )
    gc.collect()
    torch.npu.empty_cache()


@torch.inference_mode()
@pytest.mark.parametrize("hc_sinkhorn_iters", SINKHORN_ITERS_SWEEP)
@pytest.mark.parametrize("num_tokens", TOKEN_COUNT_SWEEP)
def test_npu_hc_pre_v2_row_position_invariance(num_tokens: int, hc_sinkhorn_iters: int) -> None:
    """v2 (no pre_mix) must also be row-position invariant.

    num_tokens <= 32 hits the ProcessOriginal copy; larger token counts hit the tiled
    copies. Both must treat identical rows identically.
    """
    _enable_hc_pre_custom_op()
    _check_v2_row_position_invariance(
        num_tokens,
        DSV4_FLASH_HIDDEN_SIZE,
        hc_sinkhorn_iters,
        _make_hc_base("normal"),
    )
    gc.collect()
    torch.npu.empty_cache()


# ---------------------------------------------------------------------------
# CPU reference comparison (pass-rate tolerances)
# ---------------------------------------------------------------------------


def _to_hf32(tensor: torch.Tensor) -> torch.Tensor:
    dropped_mantissa_bits = FP32_MANTISSA_BITS - HF32_MANTISSA_BITS
    mantissa_mask = ~((1 << dropped_mantissa_bits) - 1)
    bits = tensor.contiguous().view(torch.int32)
    return (bits & mantissa_mask).view(torch.float32)


def _hc_pre_cpu(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre_mix: torch.Tensor | None = None,
    hc_sinkhorn_iters: int = HC_SINKHORN_ITERS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_float = x.float()
    x_flat = x_float.flatten(-2)
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)

    # HcPre performs this matmul in HF32 round-toward-zero mode.
    mixes = F.linear(_to_hf32(x_flat), _to_hf32(hc_fn)) * inv_rms
    pre, post, comb_frag = mixes.split(
        [HC_MULT, HC_MULT, HC_MULT * HC_MULT],
        dim=-1,
    )
    comb_frag = comb_frag.unflatten(-1, (HC_MULT, HC_MULT))

    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[HC_MULT : 2 * HC_MULT])
    comb_frag = comb_frag * hc_scale[2] + hc_base[2 * HC_MULT :].view(HC_MULT, HC_MULT)

    comb_frag = comb_frag.softmax(-1) + HC_EPS
    comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(hc_sinkhorn_iters - 1):
        comb_frag = comb_frag / (comb_frag.sum(-1, keepdim=True) + HC_EPS)
        comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + HC_EPS)

    mix_for_y = pre_mix.float() if pre_mix is not None else pre
    y = (mix_for_y.unsqueeze(-1) * x_float).sum(dim=-2).to(x.dtype)
    return y, post, comb_frag, pre


def _assert_close_with_pass_rate(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    diff_threshold: float,
    required_pass_rate: float,
):
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    abs_diff = (actual - expected).abs()
    magnitude = torch.maximum(actual.abs(), expected.abs())
    close = (abs_diff <= diff_threshold) | (
        abs_diff / magnitude.clamp_min(torch.finfo(torch.float32).tiny) <= diff_threshold
    )
    pass_rate = close.float().mean().item()
    max_abs_diff = abs_diff.max().item()
    assert pass_rate >= required_pass_rate, (
        f"pass rate {pass_rate:.2%} is below {required_pass_rate:.2%}; max absolute difference: {max_abs_diff}"
    )


@torch.inference_mode()
def test_npu_hc_pre_v3_matches_cpu_reference() -> None:
    """The fixed operator must still match the CPU reference at a tail shape.

    Token count 33 with 20 Sinkhorn iterations sits just above the SMALL_BS_THRESHOLD
    tiling boundary, exercising the tiled copy that the epsilon fix touched.
    """
    _enable_hc_pre_custom_op()
    torch.manual_seed(1024)
    hidden_size = DSV4_FLASH_HIDDEN_SIZE
    fan_in = HC_MULT * hidden_size
    x = (torch.rand(CPU_REFERENCE_NUM_TOKENS, HC_MULT, hidden_size) * 2).to(torch.bfloat16)
    hc_fn = torch.rand(MIX_HC, fan_in, dtype=torch.float32) / fan_in
    hc_scale = torch.rand(3, dtype=torch.float32) * 2
    hc_base = torch.rand(MIX_HC, dtype=torch.float32) * 2
    pre_mix = torch.rand(CPU_REFERENCE_NUM_TOKENS, HC_MULT, dtype=torch.float32)

    expected_y, expected_post, expected_comb, expected_pre = _hc_pre_cpu(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        pre_mix,
    )
    y, post, comb, pre = torch.ops._C_ascend.npu_hc_pre_v3(
        x.npu(),
        hc_fn.npu(),
        hc_scale.npu(),
        hc_base.npu(),
        pre_mix.npu(),
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=HC_SINKHORN_ITERS,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
    )

    assert y.shape == (CPU_REFERENCE_NUM_TOKENS, hidden_size)
    assert post.shape == (CPU_REFERENCE_NUM_TOKENS, HC_MULT)
    assert comb.shape == (CPU_REFERENCE_NUM_TOKENS, HC_MULT, HC_MULT)
    assert pre.shape == (CPU_REFERENCE_NUM_TOKENS, HC_MULT)
    assert y.dtype == torch.bfloat16
    assert post.dtype == torch.float32
    assert comb.dtype == torch.float32
    assert pre.dtype == torch.float32
    for actual, expected, diff_threshold, required_pass_rate in (
        (y, expected_y, Y_DIFF_THRESHOLD, Y_REQUIRED_PASS_RATE),
        (post, expected_post, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
        (comb, expected_comb, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
        (pre, expected_pre, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
    ):
        _assert_close_with_pass_rate(
            actual,
            expected,
            diff_threshold=diff_threshold,
            required_pass_rate=required_pass_rate,
        )
    gc.collect()
    torch.npu.empty_cache()
