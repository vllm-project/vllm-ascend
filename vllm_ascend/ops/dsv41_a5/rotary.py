# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 partial-RoPE adapter."""

from __future__ import annotations

import torch

from vllm_ascend.ops.project_ops import get_project_op


def apply_partial_rotary_inplace(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    start: int,
    end: int,
    inverse: bool = False,
) -> torch.Tensor:
    work = x
    if work.ndim == 2:
        work = work.unsqueeze(-2)
    if work.ndim == 3:
        work = work.unsqueeze(1)
    get_project_op("inplace_partial_rotary_mul")(
        work,
        cos,
        -sin if inverse else sin,
        "interleave",
        [start, end],
    )
    return x
