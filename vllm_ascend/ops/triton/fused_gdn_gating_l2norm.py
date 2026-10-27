#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Fuse the GDN decode gating with the Q/K l2norm into a single launch.

The decode step of the GDN (Gated DeltaNet) linear attention issues three
small Triton kernels back to back: ``fused_gdn_gating_kernel`` (computes the
log-decay gate ``g`` and the interpolation coefficient ``beta``) followed by
two ``l2norm_fwd_kernel2`` calls (one for Q, one for K). The three have no
data dependency on each other, so they are merged here into one kernel.

Bit-exactness: the gating arithmetic is elementwise and is replicated in the
same fp32 order as ``fused_gdn_gating_kernel``. The l2norm part keeps the
``MBLOCK``/``BD`` block shape of ``l2norm_fwd_kernel2`` so the per-row
reduction tree over the feature axis stays identical.

This mirrors one specific ``l2norm_fwd`` backend. When ``USE_DEFAULT_FLA_NORM``
is set, ``l2norm_fwd`` dispatches to ``l2norm_fwd_kernel``/``l2norm_fwd_kernel1``
instead, whose different row tiling reorders the reduction and diverges by up to
one ULP. ``fusion_matches_l2norm_backend()`` reports that case so callers keep
the unfused path rather than silently changing results.
"""

import torch
from vllm.third_party.flash_linear_attention.ops.l2norm import USE_DEFAULT_FLA_NORM
from vllm.triton_utils import tl, triton

# Matches l2norm_fwd_kernel2's row block in
# vllm/third_party/flash_linear_attention/ops/l2norm.py.
L2NORM_MBLOCK = 32
# Token rows handled per program by the gating section.
GATE_BLK_T = 32


def fusion_matches_l2norm_backend() -> bool:
    """Whether l2norm_fwd currently uses the backend this kernel mirrors."""
    return not USE_DEFAULT_FLA_NORM


@triton.jit
def _l2norm_rows(X, Y, eps, M, N: tl.constexpr, BD: tl.constexpr, MBLOCK: tl.constexpr, row_base):
    """Row-wise l2 normalization, identical math to l2norm_fwd_kernel2."""
    row_idx = row_base + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    rindex = tl.arange(0, BD)[None, :]
    cmask = rindex < N
    mask = xmask & cmask
    xs = tl.load(X + (rindex + N * row_idx), mask, other=0.0).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, BD])
    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (rindex + N * row_idx), xs * rsqrt, mask)


@triton.jit(do_not_specialize=["num_tokens", "beta", "threshold", "n_norm_blocks", "n_gate_blocks"])
def fused_gdn_gating_l2norm_kernel(
    g,
    beta_output,
    q_out,
    k_out,
    A_log,
    a,
    b,
    dt_bias,
    q_in,
    k_in,
    eps,
    num_tokens,
    norm_rows,
    beta,
    threshold,
    n_norm_blocks,
    n_gate_blocks,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BD: tl.constexpr,
    MBLOCK: tl.constexpr,
    BLK_T: tl.constexpr,
):
    pid = tl.program_id(0)

    # --- Q / K l2norm: same block shape as l2norm_fwd_kernel2 ---
    if pid < n_norm_blocks:
        row_base = pid * MBLOCK
        _l2norm_rows(q_in, q_out, eps, norm_rows, HEAD_DIM, BD, MBLOCK, row_base)
        _l2norm_rows(k_in, k_out, eps, norm_rows, HEAD_DIM, BD, MBLOCK, row_base)

    # --- GDN gating: elementwise, same fp32 order as fused_gdn_gating_kernel ---
    if pid < n_gate_blocks:
        batch_off = pid * BLK_T + tl.arange(0, BLK_T)
        head_off = tl.arange(0, NUM_HEADS)
        off = batch_off[:, None] * NUM_HEADS + head_off[None, :]
        mask = batch_off[:, None] < num_tokens

        blk_A_log = tl.load(A_log + head_off)
        blk_a = tl.load(a + off, mask=mask)
        blk_b = tl.load(b + off, mask=mask)
        blk_bias = tl.load(dt_bias + head_off)

        x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
        softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)

        blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
        tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)

        blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
        tl.store(beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask)


def fused_gdn_gating_l2norm(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    eps: float = 1e-6,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-launch replacement for ``fused_gdn_gating`` + two ``l2norm_fwd``.

    Returns ``(g, beta_output, q_normed, k_normed)``. ``g``/``beta_output``
    match ``fused_gdn_gating_patch``; ``q_normed``/``k_normed`` match
    ``l2norm_fwd`` and keep the caller's shape.
    """
    if not fusion_matches_l2norm_backend():
        raise RuntimeError(
            "fused_gdn_gating_l2norm mirrors l2norm_fwd_kernel2, but "
            "USE_DEFAULT_FLA_NORM selects a different l2norm backend. "
            "Use the unfused path instead."
        )
    num_tokens, num_heads = a.shape
    q_shape, k_shape = q.shape, k.shape
    q_flat = q.view(-1, q_shape[-1])
    k_flat = k.view(-1, k_shape[-1])
    norm_rows, head_dim = q_flat.shape

    # Same BD derivation as l2norm_fwd.
    max_fused_size = 65536 // q.element_size()
    bd = min(max_fused_size, triton.next_power_of_2(head_dim))
    if head_dim > bd:
        raise RuntimeError(f"fused_gdn_gating_l2norm: feature dim >= 64KB, got {head_dim}.")

    g = torch.empty(1, num_tokens, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, num_tokens, num_heads, dtype=b.dtype, device=b.device)
    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)

    n_norm_blocks = triton.cdiv(norm_rows, L2NORM_MBLOCK)
    n_gate_blocks = triton.cdiv(num_tokens, GATE_BLK_T)
    grid = (max(n_norm_blocks, n_gate_blocks),)

    fused_gdn_gating_l2norm_kernel[grid](
        g,
        beta_output,
        q_out,
        k_out,
        A_log,
        a,
        b,
        dt_bias,
        q_flat,
        k_flat,
        eps,
        num_tokens,
        norm_rows,
        beta,
        threshold,
        n_norm_blocks,
        n_gate_blocks,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BD=bd,
        MBLOCK=L2NORM_MBLOCK,
        BLK_T=GATE_BLK_T,
    )
    return g, beta_output, q_out.view(q_shape), k_out.view(k_shape)

