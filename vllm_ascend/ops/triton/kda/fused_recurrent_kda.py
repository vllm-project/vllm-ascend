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
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    a_log=None,
    g_bias=None,
    COMPUTE_GATE: tl.constexpr = False,
    SIGMOID_BETA: tl.constexpr = False,
    LOWER_BOUND: tl.constexpr = -5.0,
    Q_TOKEN_STRIDE: tl.constexpr = 0,
    K_TOKEN_STRIDE: tl.constexpr = 0,
    V_TOKEN_STRIDE: tl.constexpr = 0,
    BETA_TOKEN_STRIDE: tl.constexpr = 0,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    q_stride = Q_TOKEN_STRIDE if Q_TOKEN_STRIDE else H * K
    k_stride = K_TOKEN_STRIDE if K_TOKEN_STRIDE else H * K
    v_stride = V_TOKEN_STRIDE if V_TOKEN_STRIDE else HV * V
    beta_stride = BETA_TOKEN_STRIDE if BETA_TOKEN_STRIDE else HV * (V if IS_BETA_HEADWISE else 1)
    p_q = q + bos * q_stride + i_h * K + o_k
    p_k = k + bos * k_stride + i_h * K + o_k
    p_v = v + bos * v_stride + i_hv * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * beta_stride + i_hv * V + o_v
    else:
        p_beta = beta + bos * beta_stride + i_hv

    if not IS_KDA:
        p_g = g + bos * HV + i_hv
    else:
        p_gk = g + (bos * HV + i_hv) * K + o_k

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    if COMPUTE_GATE:
        gate_scale = tl.exp(tl.load(a_log + i_hv).to(tl.float32))
        gate_bias = tl.load(g_bias + i_hv * K + o_k, mask_k, other=0).to(tl.float32)

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64)
            if state_idx <= 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= tl.exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            if COMPUTE_GATE:
                b_gk = LOWER_BOUND * tl.sigmoid(gate_scale * (b_gk + gate_bias))
            b_h *= tl.exp(b_gk[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        if SIGMOID_BETA:
            b_beta = tl.sigmoid(b_beta)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if INPLACE_FINAL_STATE:
            final_state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64)
            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += q_stride
        p_k += k_stride
        p_o += HV * V
        p_v += v_stride
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += beta_stride


def fused_recurrent_kda_varlen(q, k, v, raw_gate, raw_beta, state, starts, indices, accepted, a_log, bias, lower_bound):
    """Fused GLM verification with fixed state rows and device query lengths.

    Keep Q/K/V's token strides, and fuse the bounded gate and beta sigmoid.
    The generic KDA callers retain their preprocessed-gate arithmetic.
    """
    heads, dim = q.shape[-2:]
    if (
        q.shape[0] != 1
        or dim != 128
        or k.shape != q.shape
        or v.shape != q.shape
        or any(x.stride(-1) != 1 or x.stride(-2) != dim for x in (q, k, v))
        or indices.ndim != 2
        or indices.stride(1) != 1
        or indices.shape[0] != starts.numel() - 1
        or raw_gate.shape != q.shape
        or raw_beta.shape != q.shape[:-1]
        or raw_beta.stride(-1) != 1
        or state.shape[1:] != (heads, dim, dim)
        or state.stride()[1:] != (dim * dim, dim, 1)
        or a_log.numel() != heads
        or bias.numel() != heads * dim
        or accepted is None
        or accepted.numel() != indices.shape[0]
    ):
        raise ValueError("GLM adaptive KDA requires token-strided Q/K/V and fixed 2D request state rows")
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    requests = starts.numel() - 1
    if requests == 0:
        return output
    fused_recurrent_gated_delta_rule_fwd_kernel[(1, triton.cdiv(dim, 8), requests * heads)](
        q,
        k,
        v,
        raw_gate.contiguous(),
        raw_beta,
        output,
        state,
        state,
        starts,
        indices,
        accepted,
        dim**-0.5,
        requests,
        q.shape[1],
        1,
        heads,
        heads,
        dim,
        dim,
        dim,
        8,
        state.stride(0),
        state.stride(0),
        indices.stride(0),
        indices.stride(1),
        INPLACE_FINAL_STATE=True,
        IS_BETA_HEADWISE=False,
        USE_QK_L2NORM_IN_KERNEL=True,
        IS_KDA=True,
        a_log=a_log,
        g_bias=bias,
        COMPUTE_GATE=True,
        SIGMOID_BETA=True,
        LOWER_BOUND=lower_bound,
        Q_TOKEN_STRIDE=q.stride(1),
        K_TOKEN_STRIDE=k.stride(1),
        V_TOKEN_STRIDE=v.stride(1),
        BETA_TOKEN_STRIDE=raw_beta.stride(1),
        num_warps=1,
        num_stages=3,
    )
    return output
