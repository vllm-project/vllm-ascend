# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# mypy: ignore-errors

import torch
from vllm.triton_utils import tl, triton


@triton.heuristics(
    {"HAS_SCALE": lambda args: args["scale"] is not None, "IS_VARLEN": lambda args: args["cu_seqlens"] is not None}
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    H: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    CHUNK_SIZE: tl.constexpr = 64,
):
    i_prog = tl.program_id(0)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_prog).to(tl.int32)
        eos = tl.load(cu_seqlens + i_prog + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_prog * T

    if HEAD_FIRST:
        for i_chunk in range(N_CHUNKS):
            t_offset = i_chunk * CHUNK_SIZE
            ptr_s = tl.make_block_ptr(s + bos * H, (H, T), (T, 1), (0, t_offset), (H, CHUNK_SIZE), (1, 0))
            ptr_o = tl.make_block_ptr(o + bos * H, (H, T), (T, 1), (0, t_offset), (H, CHUNK_SIZE), (1, 0))
            b_s = tl.load(ptr_s, boundary_check=(1,)).to(tl.float32)
            b_o = tl.cumsum(b_s, axis=1, reverse=REVERSE)
            if HAS_SCALE:
                b_o *= scale
            tl.store(ptr_o, b_o.to(s.dtype.element_ty), boundary_check=(1,))
    else:
        for i_chunk in range(N_CHUNKS):
            t_offset = i_chunk * CHUNK_SIZE
            ptr_s = tl.make_block_ptr(s + bos * H, (T, H), (H, 1), (t_offset, 0), (CHUNK_SIZE, H), (1, 0))
            ptr_o = tl.make_block_ptr(o + bos * H, (T, H), (H, 1), (t_offset, 0), (CHUNK_SIZE, H), (1, 0))
            b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
            b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
            if HAS_SCALE:
                b_o *= scale
            tl.store(ptr_o, b_o.to(s.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum_scalar(
    g,
    chunk_size,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.Tensor | None = torch.float,
):
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"

    N_CHUNKS = triton.cdiv(T, chunk_size)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    if cu_seqlens is not None:
        num_seqs = cu_seqlens.shape[0] - 1
        grid = (num_seqs,)
    else:
        grid = (B,)

    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        N_CHUNKS=N_CHUNKS,
        CHUNK_SIZE=chunk_size,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        num_warps=1,
        num_stages=1,
    )
    return g


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise"
        )