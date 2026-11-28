# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Torch reference decomposition of shared delayed-mHC seam semantics."""

from __future__ import annotations

import torch


def stable_sigmoid(x: torch.Tensor) -> torch.Tensor:
    z = torch.exp(-x.abs())
    return torch.where(x >= 0, 1 / (1 + z), z / (1 + z))


def coefficients(
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm_eps: float,
    hc_eps: float,
    iterations: int,
    post_mult: float,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute post, comb, and pre from the rounded residual."""
    lanes = residual.shape[1]
    rows = residual.to(dtype).flatten(1)
    projected = (rows @ hc_fn.to(dtype).T) * torch.rsqrt(rows.square().mean(-1, keepdim=True) + norm_eps)
    sections = (lanes, lanes, lanes * lanes)
    pre, post, logits = (
        values * hc_scale[i].to(dtype) + bias
        for i, (values, bias) in enumerate(zip(projected.split(sections, -1), hc_base.to(dtype).split(sections)))
    )
    comb = logits.unflatten(-1, (lanes, lanes)).softmax(dim=-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(1, iterations):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return stable_sigmoid(post) * post_mult, comb, stable_sigmoid(pre) + hc_eps
