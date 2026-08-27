# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fla/ops/l2norm.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# mypy: ignore-errors

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["eps", "M", "NUM_CHUNKS"])
def l2norm_fwd_persistent_kernel(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, NUM_CHUNKS):
    # One program per vector core; each program loops over NUM_CHUNKS blocks
    # of MBLOCK rows so the grid stays resident on the device.
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


@triton.jit
def l2norm_fwd_tiled_kernel(X, Y, eps, M, N: tl.constexpr, BD: tl.constexpr, MBLOCK: tl.constexpr):
    # One program per MBLOCK-row tile; columns are padded to BD and masked.
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


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
    use_tiled_kernel: bool = False,
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

    if use_tiled_kernel:
        MBLOCK = 32
        l2norm_fwd_tiled_kernel[(triton.cdiv(T, MBLOCK),)](
            x,
            y,
            eps,
            T,
            D,
            BD,
            MBLOCK,
        )
    else:
        MBLOCK = 69
        num_core = get_vectorcore_num()
        main_bs = triton.cdiv(T, num_core)
        num_sub_blocks = triton.cdiv(main_bs, MBLOCK)
        l2norm_fwd_persistent_kernel[(num_core,)](
            X=x,
            Y=y,
            eps=eps,
            M=T,
            N=D,
            MBLOCK=MBLOCK,
            NUM_CHUNKS=num_sub_blocks,
        )

    return y.view(x_shape_og)
