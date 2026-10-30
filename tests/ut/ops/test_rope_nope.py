# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Regression tests for RoPE on NoPE layers (``rope_dim == 0``).

NoPE layers (e.g. ``qk_rope_head_dim == 0``) hand a zero-width rotary axis to
the Triton kernel. The host wrapper pads it with
``triton.next_power_of_2(0) == 0`` and the kernel then evaluates
``tl.arange(0, 0)``, which raises
``ValueError: arange's end argument must be greater than the start argument``.
``rope_forward_triton`` must therefore return its inputs untouched instead of
launching the kernel.

These tests do not require NPU hardware.
"""

from unittest.mock import patch

import torch

from vllm_ascend.ops.triton.rope import rope_forward_triton


def test_rope_dim_zero_is_noop():
    q = torch.randn(4, 8, 128)
    k = torch.randn(4, 2, 128)
    cos = torch.randn(4, 0)
    sin = torch.randn(4, 0)

    with patch("vllm_ascend.ops.triton.rope._triton_rope") as kernel:
        q_out, k_out = rope_forward_triton(q, k, cos=cos, sin=sin, rope_dim=0)

    kernel.assert_not_called()
    assert q_out is q
    assert k_out is k
