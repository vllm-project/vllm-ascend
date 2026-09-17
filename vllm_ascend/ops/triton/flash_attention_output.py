# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit(do_not_specialize=["num_tokens"])
def _flash_attention_gate_kernel(
    projected,
    gate,
    token_live,
    num_tokens,
    projected_stride,
    gate_stride,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    column_blocks = tl.cdiv(HIDDEN, BLOCK)
    for block in range(tl.program_id(0), num_tokens * column_blocks, tl.num_programs(0)):
        row = block // column_blocks
        columns = block % column_blocks * BLOCK + tl.arange(0, BLOCK)
        live = tl.load(token_live + row) != 0
        values = tl.load(projected + row * projected_stride + columns, (columns < HIDDEN) & live, other=0)
        gates = tl.load(gate + row * gate_stride + columns, (columns < HIDDEN) & live, other=0).to(tl.float32)
        # Match the materialized sigmoid's rounding before multiplication.
        scale = tl.sigmoid(gates).to(values.dtype)
        tl.store(projected + row * projected_stride + columns, values * scale, columns < HIDDEN)


def flash_attention_gate(
    projected: torch.Tensor,
    gate: torch.Tensor,
    token_live: torch.Tensor,
) -> torch.Tensor:
    """Fuse output gating and NaN-safe masking before a bias-free O projection."""
    init_device_properties_triton()
    block = (
        2048
        if projected.dtype == torch.bfloat16 and projected.shape[0] in (32, 64) and projected.shape[1] == 1536
        else 1024
    )
    tasks = projected.shape[0] * triton.cdiv(projected.shape[1], block)
    _flash_attention_gate_kernel[(min(tasks, get_vectorcore_num()),)](
        projected,
        gate,
        token_live,
        projected.shape[0],
        projected.stride(0),
        gate.stride(0),
        HIDDEN=projected.shape[1],
        BLOCK=block,
        multibuffer=False,
    )
    return projected


@triton.jit(do_not_specialize=["num_tokens", "output_tokens"])
def _flash_attention_output_kernel(
    result,
    token_live,
    output,
    num_tokens,
    output_tokens,
    result_stride,
    output_stride,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    column_blocks = tl.cdiv(HIDDEN, BLOCK)
    for block in range(tl.program_id(0), output_tokens * column_blocks, tl.num_programs(0)):
        row = block // column_blocks
        columns = block % column_blocks * BLOCK + tl.arange(0, BLOCK)
        live = tl.load(token_live + row, mask=row < num_tokens, other=0) != 0
        # A masked load avoids propagating NaN/Inf from inactive rows.
        values = tl.load(
            result + row * result_stride + columns,
            mask=(columns < HIDDEN) & live,
            other=0,
        )
        tl.store(output + row * output_stride + columns, values, mask=columns < HIDDEN)


def flash_attention_output(
    result: torch.Tensor,
    token_live: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write contiguous hidden rows, zeroing inactive tokens and graph padding."""
    if output.numel() == 0:
        return output
    # TP ranks beyond the last live row still own a padded output shard.
    # Avoid passing an empty mask storage to the NPU scalar load lowering.
    if token_live.numel() == 0 or result.shape[0] == 0:
        return output.zero_()
    init_device_properties_triton()
    # A fixed 1-D tile bounds UB use independently of hidden size or batch.
    block = 1024
    tasks = output.shape[0] * triton.cdiv(result.shape[1], block)
    grid = (min(tasks, get_vectorcore_num()),)
    _flash_attention_output_kernel[grid](
        result,
        token_live,
        output,
        min(result.shape[0], token_live.shape[0]),
        output.shape[0],
        result.stride(0),
        output.stride(0),
        HIDDEN=result.shape[1],
        BLOCK=block,
        multibuffer=False,
    )
    return output
