# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Recurrent state I/O checks for the GDN Triton chunk delta-rule kernel.

``chunk_gated_delta_rule_fwd_h`` writes three copies of the recurrent state: the running
state of every chunk (``h``), the state read back from memory at the start of a prefill
(``initial_state``), and the final state (``final_state``). The kernel keeps the state in
hard-coded ``[128, 64]`` registers and relies on masked/``boundary_check`` accesses when
``K`` or ``V`` is smaller, so a regression there silently corrupts the state while the
attention output can still look plausible.

The tests below do not need an independent reference: for a sequence of two chunks, the
state stored after chunk 0 must be identical to the final state of the same sequence run
on its own, and chunking a sequence must not change its final state.
"""

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h

CHUNK_SIZE = 64
NUM_TOKENS = 2 * CHUNK_SIZE
COSINE_THRESHOLD = 0.999
# `h` is written in the kernel dtype (bf16), the final state in fp32.
BF16_REL_TOLERANCE = 2.0**-6
FP32_REL_TOLERANCE = 1e-3

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)


def _make_inputs(num_heads, key_dim, value_dim, num_tokens=NUM_TOKENS):
    torch.manual_seed(20261114 + num_heads + key_dim + value_dim + num_tokens)
    shape_k = (1, num_tokens, num_heads, key_dim)
    k = torch.nn.functional.normalize(torch.randn(shape_k), dim=-1, eps=1e-6).to(torch.bfloat16).npu()
    w = (torch.randn(shape_k) * 0.1).to(torch.bfloat16).npu()
    u = (torch.randn(1, num_tokens, num_heads, value_dim)).to(torch.bfloat16).npu()
    g = (-torch.rand(1, num_tokens, num_heads, dtype=torch.float32) * 0.15).npu()
    return k, w, u, g


def _run_fwd_h(k, w, u, g, initial_state=None):
    final_h, _, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=CHUNK_SIZE,
        save_new_value=False,
        cu_seqlens=None,
    )
    if final_state is None:
        raise AssertionError("output_final_state=True must produce a final state")
    return final_h, final_state


def _cosine(actual, expected):
    actual = actual.float().flatten()
    expected = expected.float().flatten()
    actual_norm = actual.norm()
    expected_norm = expected.norm()
    # Two exactly zero states are a perfect match, not a zero cosine: ``h[0]`` is compared
    # against the zero initial state and normalising 0/0 would otherwise report 0.0. Matching
    # only when *both* are zero keeps "one side is zero" a genuine mismatch.
    if actual_norm < 1e-12 and expected_norm < 1e-12:
        return 1.0
    actual = actual / actual_norm.clamp_min(1e-12)
    expected = expected / expected_norm.clamp_min(1e-12)
    return float(torch.dot(actual, expected))


def _assert_state_matches(name, actual, expected, rel_tolerance):
    assert actual.shape == expected.shape, f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert torch.isfinite(actual).all().item(), f"{name}: non-finite values"
    cosine = _cosine(actual, expected)
    assert cosine >= COSINE_THRESHOLD, f"{name}: cosine={cosine:.6f}, expected >= {COSINE_THRESHOLD}"
    # The per-chunk state buffer (``h``) is stored in the kernel's dtype while the final
    # state is fp32, so compare in the dtype of the tensor under test.
    expected = expected.to(actual.dtype)
    tolerance = rel_tolerance * max(1.0, expected.float().abs().max().item())
    max_abs_diff = (actual.float() - expected.float()).abs().max().item()
    assert max_abs_diff <= tolerance, f"{name}: max_abs_diff={max_abs_diff:.6f}, expected <= {tolerance:.6f}"


@pytest.mark.parametrize(("key_dim", "value_dim"), [(128, 128)])
def test_state_io_is_consistent_across_chunk_boundaries(key_dim, value_dim):
    """h store / ht store / initial_state load must agree with a chunked replay."""
    num_heads = 8
    k, w, u, g = _make_inputs(num_heads, key_dim, value_dim)

    h_full, final_state_full = _run_fwd_h(k, w, u, g)
    assert h_full.shape == (1, NUM_TOKENS // CHUNK_SIZE, num_heads, key_dim, value_dim)

    # Replay the same sequence as two independent single-chunk calls.
    h_first, final_state_first = _run_fwd_h(
        k[:, :CHUNK_SIZE],
        w[:, :CHUNK_SIZE],
        u[:, :CHUNK_SIZE],
        g[:, :CHUNK_SIZE],
    )
    assert h_first.shape == (1, 1, num_heads, key_dim, value_dim)
    _, final_state_second = _run_fwd_h(
        k[:, CHUNK_SIZE:],
        w[:, CHUNK_SIZE:],
        u[:, CHUNK_SIZE:],
        g[:, CHUNK_SIZE:],
        initial_state=final_state_first,
    )

    # h[0] is the state entering chunk 0, h[1] the state entering chunk 1.
    _assert_state_matches(
        "h[0] vs zero initial state", h_full[:, 0], torch.zeros_like(h_full[:, 0]), BF16_REL_TOLERANCE
    )
    _assert_state_matches("h[1] vs final state of chunk 0", h_full[:, 1], final_state_first, BF16_REL_TOLERANCE)
    _assert_state_matches("final_state vs chunked replay", final_state_full, final_state_second, FP32_REL_TOLERANCE)


@pytest.mark.xfail(
    strict=False,
    reason=(
        "known defect: with K,V < 128 the fwd_h state I/O (initial_state load, per-chunk h store "
        "and final_state store) reads/writes the wrong rows on triton-ascend, while the attention "
        "output stays plausible; see the PR description for the measured cosines"
    ),
)
@pytest.mark.parametrize(("key_dim", "value_dim"), [(32, 32)])
def test_state_io_with_small_head_dim(key_dim, value_dim):
    """Same consistency property for head dims below the kernel's 128x64 register tile."""
    num_heads = 8
    k, w, u, g = _make_inputs(num_heads, key_dim, value_dim)

    h_full, final_state_full = _run_fwd_h(k, w, u, g)

    _, final_state_first = _run_fwd_h(
        k[:, :CHUNK_SIZE],
        w[:, :CHUNK_SIZE],
        u[:, :CHUNK_SIZE],
        g[:, :CHUNK_SIZE],
    )
    _, final_state_second = _run_fwd_h(
        k[:, CHUNK_SIZE:],
        w[:, CHUNK_SIZE:],
        u[:, CHUNK_SIZE:],
        g[:, CHUNK_SIZE:],
        initial_state=final_state_first,
    )

    _assert_state_matches(
        "h[0] vs zero initial state", h_full[:, 0], torch.zeros_like(h_full[:, 0]), BF16_REL_TOLERANCE
    )
    _assert_state_matches("h[1] vs final state of chunk 0", h_full[:, 1], final_state_first, BF16_REL_TOLERANCE)
    _assert_state_matches("final_state vs chunked replay", final_state_full, final_state_second, FP32_REL_TOLERANCE)
