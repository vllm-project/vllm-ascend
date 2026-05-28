# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fla/ops/l2norm.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

BT_LIST = [8, 16, 32, 64, 128]
USE_DEFAULT_FLA_NORM = int(os.getenv("USE_DEFAULT_FLA_NORM", "0"))


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


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["D"],
)
@triton.jit
def l2norm_fwd_kernel_fla_large(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in BT_LIST
    ],
    key=["D"],
)
@triton.jit(do_not_specialize=["NB"])
def l2norm_fwd_kernel_fla(
    x,
    y,
    eps,
    NB,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def l2norm_fwd_kernel_fla_simple(
    X, Y, eps, M, N: tl.constexpr, BD: tl.constexpr, MBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    rindex = tl.arange(0, BD)[None, :]
    cmask = rindex < N
    mask = xmask & cmask
    xs = tl.load(X + (rindex + N * row_idx), mask, other=0.0).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, BD])
    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (rindex + N * row_idx), xs * rsqrt, mask)


def _l2norm_fwd_ascend(x: torch.Tensor, y: torch.Tensor, eps: float) -> None:
    T, D = x.shape[0], x.shape[-1]
    MBLOCK = 69
    num_core = get_vectorcore_num()
    main_bs = triton.cdiv(T, num_core)
    num_sub_blocks = triton.cdiv(main_bs, MBLOCK)
    l2norm_fwd_kernel2_loop[(num_core,)](
        X=x,
        Y=y,
        eps=eps,
        M=T,
        N=D,
        MBLOCK=MBLOCK,
        NUM_CHUNKS=num_sub_blocks,
    )


def _l2norm_fwd_fla(x: torch.Tensor, y: torch.Tensor, eps: float, D: int, BD: int) -> None:
    T = x.shape[0]
    if not USE_DEFAULT_FLA_NORM:
        MBLOCK = 32
        l2norm_fwd_kernel_fla_simple[(triton.cdiv(T, MBLOCK),)](
            x,
            y,
            eps,
            T,
            D,
            BD,
            MBLOCK,
        )
    else:
        if D <= 512:
            NB = triton.cdiv(T, 2048)

            def grid(meta):
                return (triton.cdiv(T, meta["BT"]),)

            l2norm_fwd_kernel_fla[grid](
                x,
                y,
                eps,
                NB=NB,
                T=T,
                D=D,
                BD=BD,
            )
        else:
            l2norm_fwd_kernel_fla_large[(T,)](
                x,
                y,
                eps=eps,
                D=D,
                BD=BD,
            )


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
    backend: str = "ascend",
):
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
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if backend == "ascend":
        _l2norm_fwd_ascend(x, y, eps)
    elif backend == "fla":
        _l2norm_fwd_fla(x, y, eps, D, BD)
    else:
        raise ValueError(f"Unsupported l2norm backend: {backend}")

    return y.view(x_shape_og)
