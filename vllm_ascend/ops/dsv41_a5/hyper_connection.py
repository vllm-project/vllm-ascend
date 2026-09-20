# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualified A5 mHC operator adapters."""

from __future__ import annotations

import torch

from vllm_ascend.ops.project_ops import get_project_op


def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre_mix: torch.Tensor | None,
    *,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
):
    import custom_ops  # noqa: F401, PLC0415  # Registers torch.ops.custom.*.

    return torch.ops.custom.npu_hc_pre_v2(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        pre_mix,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
    )


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return get_project_op("npu_hc_post")(
        x.unsqueeze(0),
        residual.unsqueeze(0),
        post.unsqueeze(0),
        comb.unsqueeze(0),
    ).squeeze(0)
