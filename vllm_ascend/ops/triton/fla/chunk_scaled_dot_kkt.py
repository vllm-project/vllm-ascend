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

from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args["g_cumsum"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "B"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,
    g_cumsum,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    HPG: tl.constexpr = H // Hg

    bt_stride = B * T
    i_t_i = tl.program_id(0)
    o_t = tl.arange(0, BT)
    o_t_fp32 = o_t.to(tl.float32)

    lower_tri_float = (o_t_fp32[:, None] > o_t_fp32[None, :]).to(tl.float32)

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t_i * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t_i * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos

    for i_b in range(B):
        if not IS_VARLEN:
            bos = i_b * T
            i_t = i_t_i

        if Hg == 2:
            p_k0 = tl.make_block_ptr(k + (bos * Hg + 0) * K, (T, K), (Hg * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
            b_k0 = tl.load(p_k0, boundary_check=(0, 1))
            p_k1 = tl.make_block_ptr(k + (bos * Hg + 1) * K, (T, K), (Hg * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            b_base_A0 = tl.dot(b_k0, tl.trans(b_k0))
            b_base_A1 = tl.dot(b_k1, tl.trans(b_k1))
            b_lower0 = b_base_A0 * lower_tri_float
            for i_h_local in range(HPG):
                i_h = 0 * HPG + i_h_local
                p_beta = tl.make_block_ptr(beta + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))
                b_beta = tl.load(p_beta, boundary_check=(0,))
                if USE_G:
                    p_g = tl.make_block_ptr(g_cumsum + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))
                    b_g = tl.load(p_g, boundary_check=(0,))
                    exp_pos = tl.exp(b_g.to(tl.float32))
                    exp_neg = 1.0 / exp_pos
                    b_A = b_lower0 * ((b_beta.to(tl.float32) * exp_pos)[:, None] * exp_neg[None, :])
                else:
                    b_A = b_lower0 * b_beta[:, None]
                p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
                tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))
            b_lower1 = b_base_A1 * lower_tri_float
            for i_h_local in range(HPG):
                i_h = 1 * HPG + i_h_local
                p_beta = tl.make_block_ptr(beta + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))
                b_beta = tl.load(p_beta, boundary_check=(0,))
                if USE_G:
                    p_g = tl.make_block_ptr(g_cumsum + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))
                    b_g = tl.load(p_g, boundary_check=(0,))
                    exp_pos = tl.exp(b_g.to(tl.float32))
                    exp_neg = 1.0 / exp_pos
                    b_A = b_lower1 * ((b_beta.to(tl.float32) * exp_pos)[:, None] * exp_neg[None, :])
                else:
                    b_A = b_lower1 * b_beta[:, None]
                p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
                tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))
        else:
            for i_kg in range(Hg):
                p_k = tl.make_block_ptr(k + (bos * Hg + i_kg) * K, (T, K), (Hg * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_base_A = tl.dot(b_k, tl.trans(b_k))
                b_base_A_lower = b_base_A * lower_tri_float
                for i_h_local in range(HPG):
                    i_h = i_kg * HPG + i_h_local
                    p_beta = tl.make_block_ptr(beta + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))
                    b_beta = tl.load(p_beta, boundary_check=(0,))
                    if USE_G:
                        p_g = tl.make_block_ptr(g_cumsum + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))
                        b_g = tl.load(p_g, boundary_check=(0,))
                        exp_pos = tl.exp(b_g.to(tl.float32))
                        exp_neg = 1.0 / exp_pos
                        b_A = b_base_A_lower * ((b_beta.to(tl.float32) * exp_pos)[:, None] * exp_neg[None, :])
                    else:
                        b_A = b_base_A_lower * b_beta[:, None]
                    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
                    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, Hg, K = k.shape

    H = beta.shape[-1]
    BT = chunk_size
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.cpu()
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
        chunk_indices = chunk_indices.npu()
        cu_seqlens = cu_seqlens.npu()
    else:
        chunk_indices = None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)

    chunk_scaled_dot_kkt_fwd_kernel[(NT, 1)](
        k=k,
        beta=torch.permute(beta, (2, 0, 1)).contiguous(),
        g_cumsum=torch.permute(g_cumsum, (2, 0, 1)).contiguous(),
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BK=128,
        num_warps=8,
        num_stages=3,
        multibuffer=True,
    )
    return A
