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
from typing import Optional

import torch
from vllm.triton_utils import tl, triton

from .utils import prepare_chunk_offsets, safe_exp


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o_update(
    h,
    h_update,
    updated_h_state,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H  # 按某个req的某个头划分网格

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(
            tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # offset calculation
    updated_h_state += (i_n * H + i_h).to(tl.int64) * K * V

    for i_t in range(NT):
        i_tg = boh + i_t
        h_base = h + (i_tg * H + i_h).to(tl.int64) * K * V
        hupd_base = h_update + ((i_tg + i_n) * H + i_h).to(tl.int64) * K * K

        for i_k in range(tl.cdiv(K, BK)):
            p_h = tl.make_block_ptr(h_base, (K, V), (V, 1),
                                    (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            p_hupd = tl.make_block_ptr(hupd_base, (K, V), (V, 1),
                                    (i_k * BK, i_v * BV), (BK, BK), (1, 0))
            p_updated_h_state = tl.make_block_ptr(updated_h_state, (K, V), (V, 1),
                                    (i_k * BK, i_v * BV), (BK, BV), (1, 0))

            # [BK, BV]
            b_h = tl.load(p_h, boundary_check=(0, 1))
            # [BK, BK]
            b_hupd = tl.load(p_hupd, boundary_check=(0, 1))
            # [BK, BV]
            b_updated_h_state = tl.load(p_updated_h_state, boundary_check=(0, 1))

            b_h += tl.dot(b_hupd.to(tl.bfloat16), b_updated_h_state.to(tl.bfloat16))
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o_update(
    q: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    h_update: torch.Tensor,
    updated_h_state: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = chunk_size

    o = torch.empty_like(v)
    if cu_seqlens is None:
        N, chunk_offsets = B, None
    else:
        N, chunk_offsets = (
            len(cu_seqlens) - 1,
            prepare_chunk_offsets(cu_seqlens, BT),
        )

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), N * H)

    chunk_fwd_kernel_o_update[grid](
        h=h,
        h_update=h_update,
        updated_h_state=updated_h_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=128,
        BV=128,
        num_warps=4,
        num_stages=2,
    )
    return h
