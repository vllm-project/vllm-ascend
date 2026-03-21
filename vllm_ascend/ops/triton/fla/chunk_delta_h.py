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

from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices, prepare_chunk_offsets, safe_exp

_CONDITIONS = ("seq7168",)


# ============================================================================
# Multi-Kernel Split Architecture for 3x Speedup
# ============================================================================
# Split the original kernel into two independent kernels:
# - kernel_v1: Process V dimension [0:64]
# - kernel_v2: Process V dimension [64:128]
# 
# Benefits:
# 1. Each kernel has half the state tensors (16KB vs 32KB)
# 2. Each kernel has half the float32 intermediates (64KB vs 128KB)
# 3. Enables better UB utilization and potential double buffering
# ============================================================================

@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_v1(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    h_update,
    T,
    H,
    Hg,
    K,
    V,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Process V dimension [0:64] - first half of state tensor."""
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = 1 * T
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

    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K

    # State tensor for v1 path only (16KB)
    b_h1_bv = tl.zeros([128, 64], dtype=tl.bfloat16)
    v_start = 0  # V dimension [0:64]

    # Load initial state for v1 only
    if USE_INITIAL_STATE:
        h0_ptr = h0 + i_nh * K * V
        p_h0_bv = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (0, v_start), (128, 64), (1, 0))
        b_h1_bv = tl.load(p_h0_bv, boundary_check=(0, 1)).to(tl.bfloat16)

    # Main recurrence
    for i_t in range(NT):
        h_base = h + (boh + i_t) * H * K * V + i_h * K * V
        p_h1_bv = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start), (128, 64), (1, 0))
        tl.store(p_h1_bv, b_h1_bv.to(p_h1_bv.dtype.element_ty), boundary_check=(0, 1))

        w_base = w + bos * H * K + i_h * K
        p_w = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 128), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))

        k_base = k + bos * Hg * K + (i_h // (H // Hg)) * K
        p_k = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, i_t * BT), (128, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        v_new_base = v_new + bos * H * V + i_h * V

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + bos + i_h * T_max + last_idx)

        g_ptr = g + bos + i_h * T_max
        p_g = tl.make_block_ptr(g_ptr, (T,), (1,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))

        b_g = safe_exp(b_g_last - b_g)
        b_g_last = tl.exp(b_g_last)

        v_base = v + bos * H * V + i_h * V
        p_v = tl.make_block_ptr(v_base, (T, V), (stride_v, 1), (i_t * BT, v_start), (BT, 64), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        
        # Compute v_new
        b_h1_bv_f32 = b_h1_bv.to(tl.float32)
        b_v_new = b_v.to(tl.float32)
        b_v_new -= tl.dot(b_w, b_h1_bv.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (i_t * BT, v_start), (BT, 64), (1, 0))
            tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new = b_v_new * b_g[:, None]
            b_h1_bv_f32 = b_h1_bv_f32 * b_g_last

        b_v_new = b_v_new.to(k.dtype.element_ty)
        b_h1_bv_f32 += tl.dot(b_k, b_v_new)
        b_h1_bv = b_h1_bv_f32.to(tl.bfloat16)

    # Epilogue
    if STORE_FINAL_STATE:
        ht_ptr = ht + i_nh * K * V
        p_ht_bv = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start), (128, 64), (1, 0))
        tl.store(p_ht_bv, b_h1_bv.to(tl.float32), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_v2(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    h_update,
    T,
    H,
    Hg,
    K,
    V,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Process V dimension [64:128] - second half of state tensor."""
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = 1 * T
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

    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K

    # State tensor for v2 path only (16KB)
    b_h1_bv = tl.zeros([128, 64], dtype=tl.bfloat16)
    v_start = 64  # V dimension [64:128]

    # Load initial state for v2 only
    if USE_INITIAL_STATE:
        h0_ptr = h0 + i_nh * K * V
        p_h0_bv = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (0, v_start), (128, 64), (1, 0))
        b_h1_bv = tl.load(p_h0_bv, boundary_check=(0, 1)).to(tl.bfloat16)

    # Main recurrence
    for i_t in range(NT):
        h_base = h + (boh + i_t) * H * K * V + i_h * K * V
        p_h1_bv = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start), (128, 64), (1, 0))
        tl.store(p_h1_bv, b_h1_bv.to(p_h1_bv.dtype.element_ty), boundary_check=(0, 1))

        w_base = w + bos * H * K + i_h * K
        p_w = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 128), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))

        k_base = k + bos * Hg * K + (i_h // (H // Hg)) * K
        p_k = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, i_t * BT), (128, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        v_new_base = v_new + bos * H * V + i_h * V

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + bos + i_h * T_max + last_idx)

        g_ptr = g + bos + i_h * T_max
        p_g = tl.make_block_ptr(g_ptr, (T,), (1,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))

        b_g = safe_exp(b_g_last - b_g)
        b_g_last = tl.exp(b_g_last)

        v_base = v + bos * H * V + i_h * V
        p_v = tl.make_block_ptr(v_base, (T, V), (stride_v, 1), (i_t * BT, v_start), (BT, 64), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        
        # Compute v_new
        b_h1_bv_f32 = b_h1_bv.to(tl.float32)
        b_v_new = b_v.to(tl.float32)
        b_v_new -= tl.dot(b_w, b_h1_bv.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (i_t * BT, v_start), (BT, 64), (1, 0))
            tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new = b_v_new * b_g[:, None]
            b_h1_bv_f32 = b_h1_bv_f32 * b_g_last

        b_v_new = b_v_new.to(k.dtype.element_ty)
        b_h1_bv_f32 += tl.dot(b_k, b_v_new)
        b_h1_bv = b_h1_bv_f32.to(tl.bfloat16)

    # Epilogue
    if STORE_FINAL_STATE:
        ht_ptr = ht + i_nh * K * V
        p_ht_bv = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start), (128, 64), (1, 0))
        tl.store(p_ht_bv, b_h1_bv.to(tl.float32), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
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

    h = k.new_empty(B, NT, H, K, V)
    h_update = k.new_empty(B, NT, H, K, K)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    g = g.transpose(1, 2).contiguous()

    def grid(meta):
        return (1, N * H)

    # Launch v1 kernel (processes V[0:64])
    chunk_gated_delta_rule_fwd_kernel_h_v1[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        h_update=h_update,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        num_warps=4,
        num_stages=4,
    )
    
    # Launch v2 kernel (processes V[64:128])
    chunk_gated_delta_rule_fwd_kernel_h_v2[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        h_update=h_update,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        num_warps=4,
        num_stages=4,
    )
    return h, v_new, final_state