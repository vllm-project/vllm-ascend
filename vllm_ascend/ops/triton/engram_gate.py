# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused post-wkv gate with separate scoring and residual bases."""

import torch
from vllm.triton_utils import tl, triton

# Bound per-program UB usage independently of the hidden dimension.
_GATE_CHANNEL_TILE = 1024


@triton.jit
def _engram_gate_kernel(
    hidden,
    gate_hidden,
    kv,
    q,
    k,
    mask,
    output,
    hidden_stride_t,
    hidden_stride_h,
    gate_stride_t,
    gate_stride_h,
    kv_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_h,
    k_stride_d,
    mask_stride,
    HAS_GATE_HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    branch = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    hidden_sq = tl.full((), 0, tl.float32)
    key_sq = tl.full((), 0, tl.float32)
    weighted_dot = tl.full((), 0, tl.float32)
    # Bound live FP32 vectors: a padded 8192-channel row overflows A5 UB.
    for start in range(tl.cdiv(DIM, BLOCK)):
        col = start * BLOCK + offsets
        valid = col < DIM
        if HAS_GATE_HIDDEN:
            score_h = tl.load(gate_hidden + token * gate_stride_t + branch * gate_stride_h + col, valid, 0).to(
                tl.float32
            )
        else:
            score_h = tl.load(hidden + token * hidden_stride_t + branch * hidden_stride_h + col, valid, 0).to(
                tl.float32
            )
        key = tl.load(kv + token * kv_stride_t + branch * DIM + col, valid, 0).to(tl.float32)
        qw = tl.load(q + branch * q_stride_h + col * q_stride_d, valid, 0).to(tl.float32)
        kw = tl.load(k + branch * k_stride_h + col * k_stride_d, valid, 0).to(tl.float32)
        hidden_sq += tl.sum(score_h * score_h, 0)
        key_sq += tl.sum(key * key, 0)
        weighted_dot += tl.sum((score_h * (qw * kw)) * key, 0)
    rstd = tl.rsqrt(hidden_sq / DIM + EPS)
    rstd *= tl.rsqrt(key_sq / DIM + EPS)
    dot = weighted_dot * rstd * (DIM**-0.5)
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
    negative = (dot.to(tl.int32, bitcast=True) & -2147483648) != 0
    gate = tl.sigmoid(tl.where(negative, -magnitude, magnitude))
    active = tl.load(mask + token * mask_stride) != 0
    gate = tl.where(active, gate, 0.0)
    # Reuse the bounded tile for the residual; do not contract its FMA.
    for start in range(tl.cdiv(DIM, BLOCK)):
        col = start * BLOCK + offsets
        valid = col < DIM
        value = tl.load(kv + token * kv_stride_t + HC * DIM + col, valid, 0).to(tl.float32)
        h = tl.load(hidden + token * hidden_stride_t + branch * hidden_stride_h + col, valid, 0).to(tl.float32)
        tl.store(output + (token * HC + branch) * DIM + col, h + gate * value, valid)


def fused_engram_gate(hidden, kv, q_weight, k_weight, token_mask, eps, *, gate_hidden=None):
    """Fuse RMS/dot/gate/residual, keeping wkv projection outside this kernel."""
    # gate_hidden is the restored FP32 basis; hidden remains the residual basis.
    if gate_hidden is not None:
        if gate_hidden.shape != hidden.shape or gate_hidden.stride(-1) != 1:
            raise ValueError("gate_hidden must match hidden with contiguous channels")
        if gate_hidden.dtype != torch.float32 or gate_hidden.device != hidden.device:
            raise ValueError("gate_hidden must be FP32 on the hidden device")
    score_input = hidden if gate_hidden is None else gate_hidden
    tokens, branches, dim = hidden.shape
    if hidden.stride(-1) != 1 or kv.stride(-1) != 1:
        raise ValueError("fused Engram gate requires contiguous hidden/key channels")
    output = torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    if tokens:
        # Strided channel gathers can exceed A3 UB capacity during lowering.
        q_weight = q_weight.contiguous()
        k_weight = k_weight.contiguous()
        _engram_gate_kernel[(tokens, branches)](
            hidden,
            score_input,
            kv,
            q_weight,
            k_weight,
            token_mask,
            output,
            hidden.stride(0),
            hidden.stride(1),
            score_input.stride(0),
            score_input.stride(1),
            kv.stride(0),
            *q_weight.stride(),
            *k_weight.stride(),
            token_mask.stride(0),
            HAS_GATE_HIDDEN=gate_hidden is not None,
            EPS=eps,
            DIM=dim,
            HC=branches,
            BLOCK=min(triton.next_power_of_2(dim), _GATE_CHANNEL_TILE),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
