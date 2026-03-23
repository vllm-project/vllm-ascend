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


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
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
    """
    Optimized kernel with half-precision A matrix:
    1. BV=BK=128 to process full K/V dimensions without inner loops
    2. Keep A in float16 for dot product to reduce memory bandwidth
    3. Fused beta-g computation
    """
    i_t_o = tl.program_id(0)
    i_b = tl.program_id(1)
    
    # Compute sequence boundaries
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t_o * 2).to(tl.int32),
            tl.load(chunk_indices + i_t_o * 2 + 1).to(tl.int32),
        )
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, (i_b + 1) * T
        i_t = i_t_o

    # Pre-compute offsets
    offs_t = tl.arange(0, BT)
    global_offs_t = i_t * BT + offs_t
    mask_t = global_offs_t < T
    mask_t_2d = mask_t[:, None]
    offs_t_2d = global_offs_t[:, None]
    offs_bt = tl.arange(0, BT)[None, :]
    
    # Pre-compute V and K offsets
    offs_v = tl.arange(0, BV)[None, :]
    offs_k = tl.arange(0, BK)[None, :]
    mask_v = mask_t_2d & (offs_v < V)
    mask_k = mask_t_2d & (offs_k < K)

    # Process each head
    for i_h in range(H):
        # Load A matrix (BT×BT) - keep in float16 to reduce memory bandwidth
        ptr_A = A + (bos * H + i_h) * BT + offs_t_2d * (H * BT) + offs_bt
        b_A = tl.load(ptr_A, mask=mask_t_2d, other=0.0).to(tl.float32)  # Keep in float16

        # Load g and beta, compute exp(g) and fused scale factors
        ptr_g = g + bos + i_h * T + global_offs_t
        ptr_beta = beta + bos + i_h * T + global_offs_t
        b_g = tl.exp(tl.load(ptr_g, mask=mask_t, other=0.0).to(tl.float32))
        b_beta = tl.load(ptr_beta, mask=mask_t, other=0.0).to(tl.float32)
        
        # Pre-compute fused scaling factors in float32
        b_beta_2d = b_beta[:, None]
        b_beta_g_2d = b_beta[:, None] * b_g[:, None]

        # V computation: u = A @ (v * beta)
        # Keep v * beta in float32, but convert to float16 for dot product
        ptr_v = v + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v
        b_v = tl.load(ptr_v, mask=mask_v, other=0.0)
        b_v_scaled = (b_v.to(tl.float32) * b_beta_2d)  # Scale and convert to fp16
        b_u = tl.dot(b_A, b_v_scaled)  # fp16 @ fp16
        ptr_u = u + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v
        tl.store(ptr_u, b_u.to(ptr_u.dtype.element_ty), mask=mask_v)
        
        # K computation: w = A @ (k * beta * g)
        ptr_k = k + (bos * Hg + i_h // (H // Hg)) * K + offs_t_2d * (Hg * K) + offs_k
        b_k = tl.load(ptr_k, mask=mask_k, other=0.0)
        b_k_scaled = (b_k.to(tl.float32) * b_beta_g_2d)  # Scale and convert to fp16
        b_w = tl.dot(b_A, b_k_scaled)  # fp16 @ fp16
        ptr_w = w + (bos * H + i_h) * K + offs_t_2d * (H * K) + offs_k
        tl.store(ptr_w, b_w.to(ptr_w.dtype.element_ty), mask=mask_k)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Use BK=BV=128 to process all dimensions at once (K=V=128)
    BK = 128
    BV = 128

    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    beta = beta.transpose(1, 2).contiguous()
    g_cumsum = g_cumsum.transpose(1, 2).contiguous()
    
    # Launch one kernel per chunk, inner loop processes all heads
    recompute_w_u_fwd_kernel[(NT, B)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=2,
    )
    return w, u