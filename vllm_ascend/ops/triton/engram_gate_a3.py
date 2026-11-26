# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[Role] Fuse Engram gate and residual injection for rotated A3 checkpoints.
[Invariants] Gate reads inverse-rotated FP32 hidden; residual keeps its basis.
[Timeouts & Locks] Current-stream asynchronous work; no host waits or locks.
[Side-Effect Policy] Return a new hidden tensor without mutating inputs.

Keep block32 inverse rotation as a separate MatMul. Chunked row reductions
follow Triton-Ascend's LayerNorm pattern and bound A3's live UB working set:
https://github.com/triton-lang/triton-ascend/blob/main/docs/en/examples/03_layer_norm_example.md
"""

import torch
from vllm.triton_utils import tl, triton

GATE_BLOCK_SIZE = 1024


@triton.jit
def _engram_gate_a3_kernel(
    hidden,
    original,
    kv,
    q,
    k,
    mask,
    output,
    hidden_stride_t,
    hidden_stride_h,
    original_stride_t,
    original_stride_h,
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
    h2 = tl.zeros((BLOCK,), tl.float32)
    k2 = tl.zeros((BLOCK,), tl.float32)
    products = tl.zeros((BLOCK,), tl.float32)
    for offset in range(0, DIM, BLOCK):
        col = offset + tl.arange(0, BLOCK)
        valid = col < DIM
        gate_hidden = tl.load(original + token * original_stride_t + branch * original_stride_h + col, valid, 0).to(
            tl.float32
        )
        key = tl.load(kv + token * kv_stride_t + branch * DIM + col, valid, 0).to(tl.float32)
        qw = tl.load(q + branch * q_stride_h + col * q_stride_d, valid, 0).to(tl.float32)
        kw = tl.load(k + branch * k_stride_h + col * k_stride_d, valid, 0).to(tl.float32)
        h2 += gate_hidden * gate_hidden
        k2 += key * key
        products += (gate_hidden * (qw * kw)) * key
    rstd = tl.rsqrt(tl.sum(h2, 0) / DIM + EPS)
    rstd *= tl.rsqrt(tl.sum(k2, 0) / DIM + EPS)
    dot = tl.sum(products, 0) * rstd * (DIM**-0.5)
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
    negative = (dot.to(tl.int32, bitcast=True) & -2147483648) != 0
    gate = tl.sigmoid(tl.where(negative, -magnitude, magnitude))
    gate = tl.where(tl.load(mask + token * mask_stride) != 0, gate, 0.0)
    for offset in range(0, DIM, BLOCK):
        col = offset + tl.arange(0, BLOCK)
        valid = col < DIM
        h = tl.load(hidden + token * hidden_stride_t + branch * hidden_stride_h + col, valid, 0).to(tl.float32)
        value = tl.load(kv + token * kv_stride_t + HC * DIM + col, valid, 0).to(tl.float32)
        tl.store(output + (token * HC + branch) * DIM + col, h + gate * value, valid)


def fused_engram_gate_a3(hidden, kv, q_weight, k_weight, rotation, token_mask, eps):
    """Inverse-rotate only the gate input, then fuse the remaining vector work.

    ``kv`` has HC keys plus one value per row; retain its full row stride.
    The value projection already includes the checkpoint's forward rotation.
    """
    tokens, branches, dim = hidden.shape
    original = (hidden.float().unflatten(-1, (-1, rotation.shape[0])) @ rotation.float().T).flatten(-2)
    output = torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    if tokens:
        _engram_gate_a3_kernel[(tokens, branches)](
            hidden,
            original,
            kv,
            q_weight,
            k_weight,
            token_mask,
            output,
            hidden.stride(0),
            hidden.stride(1),
            original.stride(0),
            original.stride(1),
            kv.stride(0),
            *q_weight.stride(),
            *k_weight.stride(),
            token_mask.stride(0),
            EPS=eps,
            DIM=dim,
            HC=branches,
            BLOCK=min(GATE_BLOCK_SIZE, triton.next_power_of_2(dim)),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
