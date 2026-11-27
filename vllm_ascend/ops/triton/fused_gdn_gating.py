# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

UNIFIED_BUFFER_SIZE = 1572864


@triton.jit(do_not_specialize=["seq_len", "NUM_HEADS", "NUM_BATCHES", "beta", "threshold", "ROW_ITER"])
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS,
    NUM_BATCHES,
    beta,
    threshold,
    OUTPUT_HEAD_MAJOR: tl.constexpr,
    OUTPUT_RAW_BETA: tl.constexpr,
    BLK_HEADS: tl.constexpr,
    BLK_BATCHES: tl.constexpr,
    ROW_ITER,
):
    i_b, i_s = tl.program_id(0), tl.program_id(1)
    COL_ITER = tl.cdiv(NUM_HEADS, BLK_HEADS)

    for row_idx in range(0, ROW_ITER):
        batch_off = i_b * ROW_ITER * BLK_BATCHES + row_idx * BLK_BATCHES + tl.arange(0, BLK_BATCHES)

        for col_idx in range(0, COL_ITER):
            head_off = col_idx * BLK_HEADS + tl.arange(0, BLK_HEADS)

            input_off = batch_off[:, None] * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off[None, :]
            if OUTPUT_HEAD_MAJOR:
                output_off = head_off[None, :] * NUM_BATCHES * seq_len + batch_off[:, None] * seq_len + i_s
            else:
                output_off = input_off
            head_mask = head_off < NUM_HEADS
            mask = head_mask[None, :] & (batch_off[:, None] < NUM_BATCHES)

            blk_A_log = tl.load(A_log + head_off, mask=head_mask)
            blk_a = tl.load(a + input_off, mask=mask)
            blk_b = tl.load(b + input_off, mask=mask)
            blk_bias = tl.load(dt_bias + head_off, mask=head_mask)

            x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
            softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)

            blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
            tl.store(g + output_off, blk_g.to(g.dtype.element_ty), mask=mask)

            if OUTPUT_RAW_BETA:
                blk_beta_output = blk_b
            else:
                blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
            tl.store(beta_output + output_off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask)


def fused_gdn_gating_patch(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    output_head_major: bool = False,
    output_raw_beta: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads = a.shape
    seq_len = 1

    num_cores = get_vectorcore_num()

    BLK_HEADS = 8

    progs = num_cores
    row_per_core = triton.cdiv(batch, progs)
    BLK_BATCHES = 64
    ROW_ITER = triton.cdiv(row_per_core, BLK_BATCHES)

    output_shape = (1, num_heads, batch) if output_head_major else (1, batch, num_heads)
    g = torch.empty(output_shape, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(output_shape, dtype=b.dtype, device=b.device)

    grid = (progs, seq_len)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        batch,
        beta,
        threshold,
        OUTPUT_HEAD_MAJOR=output_head_major,
        OUTPUT_RAW_BETA=output_raw_beta,
        BLK_HEADS=BLK_HEADS,
        BLK_BATCHES=BLK_BATCHES,
        ROW_ITER=ROW_ITER,
    )
    return g, beta_output
