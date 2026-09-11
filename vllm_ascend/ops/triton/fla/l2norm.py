# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fla/ops/l2norm.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["eps", "M", "NUM_CHUNKS"])
def l2norm_fwd_kernel2_loop(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, NUM_CHUNKS):
    base_row = tl.program_id(0) * (NUM_CHUNKS * MBLOCK)
    rindex = tl.arange(0, N)[None, :]

    for chunk in range(NUM_CHUNKS):
        row_idx = base_row + chunk * MBLOCK + tl.arange(0, MBLOCK)[:, None]
        xmask = row_idx < M

        xs = tl.load(X + (rindex + N * row_idx), mask=xmask, other=0.0).to(tl.float32)
        square = xs * xs
        square_sum = tl.sum(square, 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)

        tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask)


@triton.jit(do_not_specialize=["eps", "M", "NUM_CHUNKS"])
def l2norm_qk_fwd_kernel(XQ, XK, YQ, YK, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, NUM_CHUNKS):
    # q and k share the same (M, N) shape, so each program normalizes the same
    # row block of both tensors.
    base_row = tl.program_id(0) * (NUM_CHUNKS * MBLOCK)
    rindex = tl.arange(0, N)[None, :]

    for chunk in range(NUM_CHUNKS):
        row_idx = base_row + chunk * MBLOCK + tl.arange(0, MBLOCK)[:, None]
        xmask = row_idx < M

        qs = tl.load(XQ + (rindex + N * row_idx), mask=xmask, other=0.0).to(tl.float32)
        q_square = qs * qs
        q_square_sum = tl.sum(q_square, 1)[:, None]
        q_rsqrt = tl.rsqrt(q_square_sum + eps)
        tl.store(YQ + (rindex + N * row_idx), qs * q_rsqrt, xmask)

        ks = tl.load(XK + (rindex + N * row_idx), mask=xmask, other=0.0).to(tl.float32)
        k_square = ks * ks
        k_square_sum = tl.sum(k_square, 1)[:, None]
        k_rsqrt = tl.rsqrt(k_square_sum + eps)
        tl.store(YK + (rindex + N * row_idx), ks * k_rsqrt, xmask)


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError(f"l2norm_fwd: This layer doesn't support feature dim >= 64KB, got {D}.")

    MBLOCK = 69
    # M, N = x.shape
    num_core = get_vectorcore_num()
    main_bs = triton.cdiv(T, num_core)
    num_sub_blocks = triton.cdiv(main_bs, MBLOCK)
    grid = (num_core,)
    l2norm_fwd_kernel2_loop[grid](
        X=x,
        Y=y,
        eps=eps,
        M=T,
        N=D,
        MBLOCK=MBLOCK,
        NUM_CHUNKS=num_sub_blocks,
    )

    return y.view(x_shape_og)


def l2norm_qk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalize ``q`` and ``k`` over the last dim in a single kernel launch.

    Fused replacement for the ``q, k = l2norm_fwd(q), l2norm_fwd(k)`` pattern
    used by GDN-style attention, where q and k always share the same shape: one
    persistent kernel normalizes the same row block of both tensors per
    program. When the flattened shapes differ, the pair degrades to two
    single-tensor ``l2norm_fwd`` calls.
    """
    q_shape_og, k_shape_og = q.shape, k.shape
    q2d = q.reshape(-1, q.shape[-1])
    k2d = k.reshape(-1, k.shape[-1])

    if q2d.shape != k2d.shape:
        return l2norm_fwd(q, eps, output_dtype), l2norm_fwd(k, eps, output_dtype)

    if output_dtype is None:
        yq = torch.empty_like(q2d)
        yk = torch.empty_like(k2d)
    else:
        yq = torch.empty_like(q2d, dtype=output_dtype)
        yk = torch.empty_like(k2d, dtype=output_dtype)
    assert yq.stride(-1) == 1 and yk.stride(-1) == 1
    T, D = q2d.shape[0], q2d.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // max(q2d.element_size(), k2d.element_size())
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError(f"l2norm_qk_fwd: This layer doesn't support feature dim >= 64KB, got {D}.")

    MBLOCK = 69
    num_core = get_vectorcore_num()
    main_bs = triton.cdiv(T, num_core)
    num_sub_blocks = triton.cdiv(main_bs, MBLOCK)
    grid = (num_core,)
    l2norm_qk_fwd_kernel[grid](
        XQ=q2d,
        XK=k2d,
        YQ=yq,
        YK=yk,
        eps=eps,
        M=T,
        N=D,
        MBLOCK=MBLOCK,
        NUM_CHUNKS=num_sub_blocks,
    )

    return yq.view(q_shape_og), yk.view(k_shape_og)
