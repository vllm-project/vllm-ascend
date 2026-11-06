# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""A3 shared/routed Add + mHC Post, with the original BF16 rounding point."""

import torch
from vllm.triton_utils import tl, triton

MAX_HC_POST_FUSION_TOKENS = 128
MIN_HC_POST_BLOCK = 256
MAX_HC_POST_BLOCK = 2048


def can_fuse_hc_post_add(
    shared: torch.Tensor | None,
    routed: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> bool:
    """Metadata-only guard; communication eligibility is checked by the runner."""
    return (
        shared is not None
        and routed.ndim == 2
        and routed.shape[0] <= MAX_HC_POST_FUSION_TOKENS
        and residual.ndim == 3
        and residual.shape == (routed.shape[0], 4, routed.shape[1])
        and shared.shape == routed.shape
        and post.shape == (routed.shape[0], 4)
        and comb.shape == (routed.shape[0], 4, 4)
        and routed.shape[1] > 0
        and all(t.dtype == torch.bfloat16 for t in (shared, routed, residual))
        and post.dtype == comb.dtype == torch.float32
        and all(t.device == routed.device for t in (shared, residual, post, comb))
        and all(t.stride(-1) == 1 for t in (shared, routed, residual, post, comb))
    )


@triton.jit
def _hc_post_add_kernel(
    shared,
    routed,
    residual,
    post,
    comb,
    output,
    shared_stride,
    routed_stride,
    residual_stride_t,
    residual_stride_h,
    post_stride,
    comb_stride_t,
    comb_stride_h,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    branch = tl.arange(0, 4)
    valid = col < DIM
    s = tl.load(shared + token * shared_stride + col, valid, 0).to(tl.float32)
    r = tl.load(routed + token * routed_stride + col, valid, 0).to(tl.float32)
    # ForeachAdd writes BF16 before HcPost reads it. Removing this cast changes
    # the model even though the intermediate tensor is no longer materialized.
    x = (s + r).to(tl.bfloat16).to(tl.float32)
    p = tl.load(post + token * post_stride + branch)
    value = p[:, None] * x[None, :]
    # Match hc_post_d_split.h: x*post first, then sources 0,1,2,3, with
    # independent FP32 Mul/Add (not a contracted FMA), then BF16 RINT.
    for source in tl.static_range(4):
        h = tl.load(residual + token * residual_stride_t + source * residual_stride_h + col, valid, 0).to(tl.float32)
        c = tl.load(comb + token * comb_stride_t + source * comb_stride_h + branch)
        value = value + c[:, None] * h[None, :]
    tl.store(output + (token * 4 + branch[:, None]) * DIM + col[None, :], value, valid[None, :])


def fused_hc_post_add(
    shared: torch.Tensor,
    routed: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Fuse two already-finalized expert outputs; never perform a collective."""
    if not can_fuse_hc_post_add(shared, routed, residual, post, comb):
        raise ValueError("HcPost fusion requires at most 128 tokens, BF16 [tokens, 4, dim], and FP32 coefficients")
    tokens, _, dim = residual.shape
    output = torch.empty(residual.shape, device=residual.device, dtype=residual.dtype)
    if tokens:
        # A 256-column tile launches 20 programs per token at hidden size 5120.
        # Wider tiles reduce repeated coefficient loads and scheduling waves;
        # keep the same ordered FP32 operations and intermediate BF16 rounding.
        block = min(MAX_HC_POST_BLOCK, max(MIN_HC_POST_BLOCK, triton.next_power_of_2(dim)))
        _hc_post_add_kernel[(tokens, triton.cdiv(dim, block))](
            shared,
            routed,
            residual,
            post,
            comb,
            output,
            shared.stride(0),
            routed.stride(0),
            residual.stride(0),
            residual.stride(1),
            post.stride(0),
            comb.stride(0),
            comb.stride(1),
            DIM=dim,
            BLOCK=block,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
