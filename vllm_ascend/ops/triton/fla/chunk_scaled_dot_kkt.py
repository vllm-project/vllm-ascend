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

from vllm_ascend.ops.triton.triton_utils import get_aicore_num

from .utils import prepare_chunk_indices, safe_exp


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args["g_cumsum"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "B", "bh_step", "task_num", "num_core"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,  # [B, T, H]
    g_cumsum,  # [B, T, H]
    A,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    bh_step,
    task_num,
    num_core,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    # Keep the original physical sequence stride for BHTD K addressing. In
    # varlen mode T below becomes the current sequence length.
    T_max = T
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t_i = task_id // bh_step
        i_bh = task_id % bh_step
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = (
                tl.load(chunk_indices + i_t_i * 2).to(tl.int32),
                tl.load(chunk_indices + i_t_i * 2 + 1).to(tl.int32),
            )
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
            k_offset = (i_h // (H // Hg) * T_max + bos) * K
        else:
            bos, eos = i_b * T, i_b * T + T
            i_t = i_t_i
            k_offset = (i_b * Hg + i_h // (H // Hg)) * T_max * K
        o_t = tl.arange(0, BT)
        o_t_fp32 = o_t.to(tl.float32)
        token_offsets = i_t * BT + o_t
        token_mask = token_offsets < T

        # A3's block-pointer lowering does not support this strided 1D BTH
        # access. Use explicit loads to avoid materializing an HBT copy.
        b_beta = tl.load(beta + (bos + token_offsets) * H + i_h, mask=token_mask, other=0.0)

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(k + k_offset, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_A += tl.dot(b_k, tl.trans(b_k))

        if USE_G:
            b_g = tl.load(g_cumsum + (bos + token_offsets) * H + i_h, mask=token_mask, other=0.0)
            b_g_diff = b_g[:, None] - b_g[None, :]
            b_A *= safe_exp(b_g_diff)

        b_A *= b_beta[:, None]
        b_A = tl.where(o_t_fp32[:, None] > o_t_fp32[None, :], b_A, 0)
        p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, H, T, K]`.
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
    B, Hg, T, K = k.shape

    H = beta.shape[-1]
    BT = chunk_size
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)

    num_core = get_aicore_num()
    bh_step = B * H
    task_num = NT * bh_step

    from vllm_ascend.device.device_op import DeviceOperator

    A = DeviceOperator.chunk_scaled_dot_kkt_fwd(
        num_core=num_core,
        bh_step=bh_step,
        task_num=task_num,
        k=k,
        beta=beta,
        g_cumsum=g_cumsum,
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
    )
    return A
