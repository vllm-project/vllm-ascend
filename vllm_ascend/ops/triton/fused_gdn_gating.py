# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def fused_gdn_gating_kernel(
    g,
    gate_beta,
    a_log,
    a,
    b,
    dt_bias,
    num_tokens,
    num_heads: tl.constexpr,
    a_row_stride: tl.constexpr,
    b_row_stride: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = (tokens[:, None] < num_tokens) & (heads[None, :] < num_heads)
    log_weight = tl.load(a_log + heads, heads < num_heads, 0).to(tl.float32)
    bias = tl.load(dt_bias + heads, heads < num_heads, 0).to(tl.float32)
    x = tl.load(a + tokens[:, None] * a_row_stride + heads[None, :], mask, 0).to(tl.float32)
    y = tl.load(b + tokens[:, None] * b_row_stride + heads[None, :], mask, 0).to(tl.float32)
    x = x + bias[None, :]
    softplus = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    offsets = tokens[:, None] * num_heads + heads[None, :]
    tl.store(g + offsets, -tl.exp(log_weight)[None, :] * softplus, mask)
    tl.store(gate_beta + offsets, tl.sigmoid(y), mask)


def fused_gdn_gating_patch(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the GDN gates directly to the strided linear projection slices."""
    num_tokens, num_heads = a.shape
    g = a.new_empty((1, num_tokens, num_heads), dtype=torch.float32)
    gate_beta = b.new_empty((1, num_tokens, num_heads))
    if num_tokens and num_heads:
        # BA projection slices have adjacent heads and a gap between tokens.
        # Pack only an interleaved head axis to preserve vector loads on large batches.
        if a.stride(1) != 1:
            a = a.contiguous()
        if b.stride(1) != 1:
            b = b.contiguous()
        block_h = min(64, triton.next_power_of_2(num_heads))
        block_t = min(16, 256 // block_h)
        grid = (triton.cdiv(num_tokens, block_t), triton.cdiv(num_heads, block_h))
        fused_gdn_gating_kernel[grid](
            g,
            gate_beta,
            A_log,
            a,
            b,
            dt_bias,
            num_tokens,
            num_heads,
            a.stride(0),
            b.stride(0),
            beta,
            threshold,
            BLOCK_T=block_t,
            BLOCK_H=block_h,
        )
    return g, gate_beta
