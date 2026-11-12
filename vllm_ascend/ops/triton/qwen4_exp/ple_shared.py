# SPDX-License-Identifier: Apache-2.0
"""Read BF16 Qwen PLE rows from independently mapped host-memory slabs."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_ple_rows(
    ptrs,
    ids,
    output,
    rows,
    vocab_start,
    vocab_end,
    ROWS_PER_MAPPING: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row < rows:
        index = tl.load(ids + row).to(tl.int64)
        valid = (index >= vocab_start) & (index < vocab_end)
        index = tl.where(valid, index, 0)
        mapping = index // ROWS_PER_MAPPING
        within = index % ROWS_PER_MAPPING
        base = tl.load(ptrs + mapping).to(tl.pointer_type(tl.bfloat16))
        col = tl.arange(0, BLOCK)
        value = tl.load(base + within * WIDTH + col, mask=valid & (col < WIDTH), other=0)
        tl.store(output + row * WIDTH + col, value, mask=col < WIDTH)


def gather_ple_rows(
    ptrs: torch.Tensor,
    ids: torch.Tensor,
    output: torch.Tensor,
    *,
    vocab_start: int,
    vocab_end: int,
    rows_per_mapping: int,
) -> None:
    """Fill output for global IDs; IDs outside this ETP vocabulary range are zero."""
    if ids.dtype not in (torch.int32, torch.int64) or not ids.is_contiguous():
        raise ValueError("PLE IDs must be contiguous int32 or int64")
    if output.dtype != torch.bfloat16 or not output.is_contiguous() or output.shape[:-1] != ids.shape:
        raise ValueError("PLE output must be contiguous BF16 with one row per ID")
    if ptrs.dtype != torch.int64 or ptrs.ndim != 1 or not ptrs.is_contiguous():
        raise ValueError("PLE mapping pointers must be a contiguous int64 vector")
    if ids.device != output.device or ids.device != ptrs.device:
        raise ValueError("PLE IDs, pointers and output must share a device")
    width = output.shape[-1]
    if rows_per_mapping <= 0 or width <= 0 or not 0 <= vocab_start <= vocab_end:
        raise ValueError("Invalid PLE lookup bounds")
    if vocab_end > ptrs.numel() * rows_per_mapping:
        raise ValueError("PLE vocabulary range exceeds mapped rows")
    if ids.numel() and not ptrs.numel():
        raise ValueError("PLE lookup requires at least one mapping")
    if ids.numel():
        _gather_ple_rows[(ids.numel(),)](
            ptrs,
            ids,
            output,
            ids.numel(),
            vocab_start,
            vocab_end,
            ROWS_PER_MAPPING=rows_per_mapping,
            WIDTH=width,
            BLOCK=triton.next_power_of_2(width),
        )
