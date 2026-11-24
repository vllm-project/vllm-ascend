# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused post-wkv gate for native (unrotated) A5 checkpoints."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _engram_gate_kernel(
    hidden,
    kv,
    q,
    k,
    mask,
    output,
    hidden_stride_t,
    hidden_stride_h,
    kv_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_h,
    k_stride_d,
    mask_stride,
    EPS: tl.constexpr,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    branch = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    valid = col < DIM
    h = tl.load(hidden + token * hidden_stride_t + branch * hidden_stride_h + col, valid, 0).to(tl.float32)
    key = tl.load(kv + token * kv_stride_t + branch * DIM + col, valid, 0).to(tl.float32)
    qw = tl.load(q + branch * q_stride_h + col * q_stride_d, valid, 0).to(tl.float32)
    kw = tl.load(k + branch * k_stride_h + col * k_stride_d, valid, 0).to(tl.float32)
    # Match the unfused FP32 operation order; do not contract the residual FMA.
    rstd = tl.rsqrt(tl.sum(h * h, 0) / DIM + EPS)
    rstd *= tl.rsqrt(tl.sum(key * key, 0) / DIM + EPS)
    dot = tl.sum((h * (qw * kw)) * key, 0) * rstd * (DIM**-0.5)
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
    negative = (dot.to(tl.int32, bitcast=True) & -2147483648) != 0
    gate = tl.sigmoid(tl.where(negative, -magnitude, magnitude))
    active = tl.load(mask + token * mask_stride) != 0
    gate = tl.where(active, gate, 0.0)
    value = tl.load(kv + token * kv_stride_t + HC * DIM + col, valid, 0).to(tl.float32)
    tl.store(output + (token * HC + branch) * DIM + col, h + gate * value, valid)


def fused_engram_gate(hidden, kv, q_weight, k_weight, token_mask, eps):
    """Fuse RMS/dot/gate/residual, keeping wkv projection outside this kernel."""
    tokens, branches, dim = hidden.shape
    if hidden.stride(-1) != 1 or kv.stride(-1) != 1:
        raise ValueError("fused Engram gate requires contiguous hidden/key channels")
    output = torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    if tokens:
        _engram_gate_kernel[(tokens, branches)](
            hidden,
            kv,
            q_weight,
            k_weight,
            token_mask,
            output,
            hidden.stride(0),
            hidden.stride(1),
            kv.stride(0),
            *q_weight.stride(),
            *k_weight.stride(),
            token_mask.stride(0),
            EPS=eps,
            DIM=dim,
            HC=branches,
            BLOCK=triton.next_power_of_2(dim),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
