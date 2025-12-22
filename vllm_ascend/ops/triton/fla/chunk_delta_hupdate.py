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

from .utils import prepare_chunk_indices, prepare_chunk_offsets, safe_exp, prepare_update_chunk_offsets
from vllm_ascend.utils import prefill_context_parallel_enable

# isort: off
if prefill_context_parallel_enable():
    from vllm.distributed import get_pcp_group, get_tp_group

_CONDITIONS = ("seq7168", )
        
@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_hupdate_blockdim64(
    k,
    w,
    g,
    cu_seqlens,
    chunk_offsets,
    h_update,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = 1 * T
    bos, eos = (
        tl.load(cu_seqlens + i_n).to(tl.int32),
        tl.load(cu_seqlens + i_n + 1).to(tl.int32),
    )
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    stride_k = Hg * K
    stride_w = H * K

    # create b_hupd_bv1 and b_hupd_bv2
    off_hupd_1_top = tl.arange(0, 64)[:, None]
    off_hupd_2_top = tl.arange(0, 64)[None, :]

    # main recurrence
    for i_t in range(NT):
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + bos + i_h * T_max + last_idx)

        offs_t = i_t * BT + tl.arange(0, BT)
        mask_t = offs_t < T
        g_ptr = g + bos + i_h * T_max
        b_g = tl.load(g_ptr + offs_t, mask=mask_t, other=0.0)

        b_g = safe_exp(b_g_last - b_g)
        b_g_last = tl.exp(b_g_last)

        offs_t_wv = (i_t * BT + tl.arange(0, BT))[:, None]
        w_base = w + bos * H * K + i_h * K
        # get column-sliced w [BT, 64]
        offs_w_upd1 = tl.arange(0, 64)[None, :]
        mask_w_upd1 = (offs_t_wv < T) & (offs_w_upd1 < K)
        ptr_w_upd1 = w_base + offs_t_wv * stride_w + offs_w_upd1 * 1
        b_w_upd1 = tl.load(ptr_w_upd1, mask=mask_w_upd1, other=0.0).to(tl.float32)

        offs_w_upd2 = 64 + tl.arange(0, 64)[None, :]
        mask_w_upd2 = (offs_t_wv < T) & (offs_w_upd2 < K)
        ptr_w_upd2 = w_base + offs_t_wv * stride_w + offs_w_upd2 * 1
        b_w_upd2 = tl.load(ptr_w_upd2, mask=mask_w_upd2, other=0.0).to(tl.float32)

        k_base = k + bos * Hg * K + (i_h // (H // Hg)) * K
        # get row-sliced k [64, T]
        p_k_upd1 = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, i_t * BT),
                                (64, BT), (0, 1))
        b_k_upd1 = tl.load(p_k_upd1, boundary_check=(0, 1))
        p_k_upd2 = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (64, i_t * BT),
                                (64, BT), (0, 1))
        b_k_upd2 = tl.load(p_k_upd2, boundary_check=(0, 1))

        if USE_G:
            b_w_upd1 = b_w_upd1 * b_g[:, None]
            b_w_upd2 = b_w_upd2 * b_g[:, None]

        # compute [64, BT] @ [BT, 64]
        b_hupd_local_11 = (off_hupd_1_top == off_hupd_2_top).to(tl.float32)
        b_hupd_local_22 = (off_hupd_1_top == off_hupd_2_top).to(tl.float32)

        # fp32
        if USE_G:
            b_hupd_local_11 = b_hupd_local_11 * b_g_last
            b_hupd_local_22 = b_hupd_local_22 * b_g_last

        b_hupd_local_11 -= tl.dot(b_k_upd1, b_w_upd1.to(b_k_upd1.dtype))
        b_hupd_local_22 -= tl.dot(b_k_upd2, b_w_upd2.to(b_k_upd2.dtype))
        b_hupd_local_12 = -tl.dot(b_k_upd1, b_w_upd2.to(b_k_upd1.dtype)).to(tl.float32)
        b_hupd_local_21 = -tl.dot(b_k_upd2, b_w_upd1.to(b_k_upd2.dtype)).to(tl.float32)

        hupd_base = h_update + (boh + i_t + i_n) * H * K * K + i_h * K * K
        p_hupd_11 = tl.make_block_ptr(hupd_base, (K, K), (K, 1), (0, 0),
                                    (64, 64), (1, 0))
        b_hupd_11 = tl.load(p_hupd_11, boundary_check=(1, 0))
        p_hupd_21 = tl.make_block_ptr(hupd_base, (K, K), (K, 1), (64, 0),
                                    (64, 64), (1, 0))
        b_hupd_21 = tl.load(p_hupd_21, boundary_check=(1, 0))
        p_hupd_12 = tl.make_block_ptr(hupd_base, (K, K), (K, 1), (0, 64),
                                    (64, 64), (1, 0))
        b_hupd_12 = tl.load(p_hupd_12, boundary_check=(1, 0))
        p_hupd_22 = tl.make_block_ptr(hupd_base, (K, K), (K, 1), (64, 64),
                                    (64, 64), (1, 0))
        b_hupd_22 = tl.load(p_hupd_22, boundary_check=(1, 0))

        b_hupd11_new = tl.dot(b_hupd_local_11.to(b_hupd_11.dtype), b_hupd_11).to(tl.float32)
        b_hupd11_new += tl.dot(b_hupd_local_12.to(b_hupd_21.dtype), b_hupd_21)

        b_hupd21_new = tl.dot(b_hupd_local_21.to(b_hupd_11.dtype), b_hupd_11).to(tl.float32)
        b_hupd21_new += tl.dot(b_hupd_local_22.to(b_hupd_21.dtype), b_hupd_21)

        b_hupd12_new = tl.dot(b_hupd_local_11.to(b_hupd_12.dtype), b_hupd_12).to(tl.float32)
        b_hupd12_new += tl.dot(b_hupd_local_12.to(b_hupd_22.dtype), b_hupd_22)

        b_hupd22_new = tl.dot(b_hupd_local_21.to(b_hupd_12.dtype), b_hupd_12).to(tl.float32)
        b_hupd22_new += tl.dot(b_hupd_local_22.to(b_hupd_22.dtype), b_hupd_22)

        hupd_next = h_update + (boh + i_t + i_n + 1) * H * K * K + i_h * K * K
        p_hupd_11 = tl.make_block_ptr(hupd_next, (K, K), (K, 1), (0, 0),
                                    (64, 64), (1, 0))
        tl.store(p_hupd_11,
                 b_hupd11_new.to(p_hupd_11.dtype.element_ty),
                 boundary_check=(0, 1))
        
        p_hupd_21 = tl.make_block_ptr(hupd_next, (K, K), (K, 1), (64, 0),
                                    (64, 64), (1, 0))
        tl.store(p_hupd_21,
                 b_hupd21_new.to(p_hupd_21.dtype.element_ty),
                 boundary_check=(0, 1))
        
        p_hupd_12 = tl.make_block_ptr(hupd_next, (K, K), (K, 1), (0, 64),
                                    (64, 64), (1, 0))
        tl.store(p_hupd_12,
                 b_hupd12_new.to(p_hupd_12.dtype.element_ty),
                 boundary_check=(0, 1))
        
        p_hupd_22 = tl.make_block_ptr(hupd_next, (K, K), (K, 1), (64, 64),
                                    (64, 64), (1, 0))
        tl.store(p_hupd_22,
                 b_hupd22_new.to(p_hupd_22.dtype.element_ty),
                 boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_hupdate(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    cu_seqlens: Optional[torch.LongTensor] = None,
    num_decodes: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    chunk_indices = (prepare_chunk_indices(cu_seqlens, chunk_size)
                     if cu_seqlens is not None else None)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h_update = k.new_empty(B, NT + N, H, K, K, dtype=torch.float32)
    update_indices = prepare_update_chunk_offsets(cu_seqlens, BT)[:-1]
    h_update[:, update_indices, :, :, :] = torch.eye(K, dtype=h_update.dtype, device=h_update.device)

    g = g.transpose(1, 2).contiguous()

    def grid(meta):
        return (1, N * H)

    chunk_gated_delta_rule_fwd_kernel_hupdate_blockdim64[grid](
        k=k,
        w=w,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        h_update=h_update,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        num_warps=4,
        num_stages=2,
    )
    # if get_pcp_group().rank_in_group == 1 and get_tp_group().rank_in_group == 1:
    #     print(">>>>>>>>", h_update.dtype, h_update.shape)
    h_update[:, :num_decodes * 2, :, :, :] = torch.zeros((K, K), dtype=h_update.dtype, device=h_update.device)
    return h_update
