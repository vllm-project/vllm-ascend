# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused post-wkv gate. A5 keeps the original unrotated kernel; A3 adds rotated scoring."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


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


@triton.jit
def _rotated_engram_gate_kernel(
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
    EPS: tl.constexpr,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    branch = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    valid = col < DIM
    score_h = tl.load(gate_hidden + token * gate_stride_t + branch * gate_stride_h + col, valid, 0).to(tl.float32)
    key = tl.load(kv + token * kv_stride_t + branch * DIM + col, valid, 0).to(tl.float32)
    qw = tl.load(q + branch * q_stride_h + col * q_stride_d, valid, 0).to(tl.float32)
    kw = tl.load(k + branch * k_stride_h + col * k_stride_d, valid, 0).to(tl.float32)
    # Match the unfused FP32 operation order; do not contract the residual FMA.
    rstd = tl.rsqrt(tl.sum(score_h * score_h, 0) / DIM + EPS)
    rstd *= tl.rsqrt(tl.sum(key * key, 0) / DIM + EPS)
    dot = tl.sum((score_h * (qw * kw)) * key, 0) * rstd * (DIM**-0.5)
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
    negative = (dot.to(tl.int32, bitcast=True) & -2147483648) != 0
    gate = tl.sigmoid(tl.where(negative, -magnitude, magnitude))
    active = tl.load(mask + token * mask_stride) != 0
    gate = tl.where(active, gate, 0.0)
    value = tl.load(kv + token * kv_stride_t + HC * DIM + col, valid, 0).to(tl.float32)
    # Keep the residual out of UB until scoring reductions have finished.
    h = tl.load(hidden + token * hidden_stride_t + branch * hidden_stride_h + col, valid, 0).to(tl.float32)
    tl.store(output + (token * HC + branch) * DIM + col, h + gate * value, valid)


def fused_engram_gate(hidden, kv, q_weight, k_weight, token_mask, eps, *, gate_hidden=None):
    """Fuse RMS/dot/gate/residual, keeping wkv projection outside this kernel."""
    tokens, branches, dim = hidden.shape
    if hidden.stride(-1) != 1 or kv.stride(-1) != 1:
        raise ValueError("fused Engram gate requires contiguous hidden/key channels")
    output = torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    if gate_hidden is not None:
        return _launch_rotated_engram_gate(
            hidden, gate_hidden, kv, q_weight, k_weight, token_mask, eps, output, tokens, branches, dim
        )
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


def _launch_rotated_engram_gate(
    hidden, gate_hidden, kv, q_weight, k_weight, token_mask, eps, output, tokens, branches, dim
):
    """Launch the A3 kernel that scores restored FP32 hidden and residuals separately."""
    if not get_current_hardware_profile().supports(HardwareCapability.ENGRAM_ROTATED_GATE):
        raise ValueError("Rotated Engram gate fusion requires ENGRAM_ROTATED_GATE")
    if gate_hidden.shape != hidden.shape or gate_hidden.stride(-1) != 1:
        raise ValueError("gate_hidden must match hidden with contiguous channels")
    if gate_hidden.dtype != torch.float32 or gate_hidden.device != hidden.device:
        raise ValueError("gate_hidden must be FP32 on the hidden device")
    if not tokens:
        return output
    # Strided channel gathers can exceed A3 UB capacity during lowering.
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    _rotated_engram_gate_kernel[(tokens, branches)](
        hidden,
        gate_hidden,
        kv,
        q_weight,
        k_weight,
        token_mask,
        output,
        hidden.stride(0),
        hidden.stride(1),
        gate_hidden.stride(0),
        gate_hidden.stride(1),
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
