# SPDX-License-Identifier: Apache-2.0
"""Composed Hyper-Connection operators for DeepSeek V4 on Ascend 310P.

The fused ``npu_hc_pre_v2`` and ``npu_hc_post`` custom operators are not
shipped in the 310P package. These functions reproduce the formulas from the
DeepSeek V4 reference implementation using ordinary PyTorch operators, which
torch-npu lowers to supported ACLNN kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hc_split_sinkhorn_310p(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split HC logits and apply the reference Sinkhorn normalization."""
    expected_width = (2 + hc_mult) * hc_mult
    if mixes.shape[-1] != expected_width:
        raise ValueError(f"Expected HC mix width {expected_width}, got {mixes.shape[-1]}.")
    if hc_scale.numel() != 3:
        raise ValueError(f"Expected three HC scales, got {hc_scale.numel()}.")
    if hc_base.numel() != expected_width:
        raise ValueError(f"Expected HC base width {expected_width}, got {hc_base.numel()}.")
    if sinkhorn_iters < 1:
        raise ValueError("sinkhorn_iters must be at least one.")

    pre_logits = mixes[..., :hc_mult]
    post_logits = mixes[..., hc_mult : 2 * hc_mult]
    comb_logits = mixes[..., 2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult))

    pre = torch.sigmoid(pre_logits * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(post_logits * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = comb_logits * hc_scale[2] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult)

    # Reference order: row softmax + eps, column normalization, then
    # alternating row/column normalizations for the remaining iterations.
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


def hc_pre_310p(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce HC copies to one hidden state and return post-mix metadata."""
    if x.ndim not in (3, 4):
        raise ValueError(f"HC pre expects a 3D or 4D tensor, got {x.ndim}D.")
    if x.shape[-2] != hc_mult:
        raise ValueError(f"HC dimension must equal hc_mult={hc_mult}, got {x.shape[-2]}.")

    original_dtype = x.dtype
    x_float = x.float()
    x_flat = x_float.flatten(start_dim=x.ndim - 2)
    rsqrt = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn_310p(
        mixes,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_iters,
        hc_eps,
    )
    y = torch.sum(pre.unsqueeze(-1) * x_float, dim=-2)
    return y.to(original_dtype), post, comb


def hc_post_310p(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Expand one hidden state back to HC copies using learned mixing."""
    if residual.shape[-2] != post.shape[-1]:
        raise ValueError(f"Residual HC dimension and post width differ: {residual.shape[-2]} vs {post.shape[-1]}.")
    if comb.shape[-2:] != (post.shape[-1], post.shape[-1]):
        raise ValueError(f"Expected square HC comb matrix, got {tuple(comb.shape[-2:])}.")

    output = post.unsqueeze(-1) * x.unsqueeze(-2)
    output = output + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2),
        dim=-3,
    )
    return output.to(x.dtype)
