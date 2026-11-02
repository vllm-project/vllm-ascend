# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Fuse query-row range, right-search and request clamp without AI_CPU ops."""

import torch
import triton
import triton.language as tl


@triton.jit
def _request_rows(
    boundaries,
    output,
    ROWS: tl.constexpr,
    ENDS: tl.constexpr,
    MAX_REQUEST: tl.constexpr,
    ITERATIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    low = tl.full((BLOCK,), 0, tl.int32)
    high = tl.full((BLOCK,), ENDS, tl.int32)
    for _ in tl.static_range(ITERATIONS):
        active = (row < ROWS) & (low < high)
        middle = low + (high - low) // 2
        end = tl.load(boundaries + 1 + middle, mask=active, other=0)
        advance = active & (end <= row)
        high = tl.where(active & ~advance, middle, high)
        low = tl.where(advance, middle + 1, low)
    tl.store(output + row, tl.minimum(low, MAX_REQUEST), mask=row < ROWS)


def try_build_request_rows(query_start_loc, output, num_reqs: int) -> bool:
    """Read current boundaries and write the existing graph-owned output view."""
    if not (
        query_start_loc.device.type == "npu"
        and query_start_loc.ndim == output.ndim == 1
        and query_start_loc.dtype in (torch.int32, torch.int64)
        and output.dtype == torch.int32
        and output.device == query_start_loc.device
        and query_start_loc.is_contiguous()
        and output.is_contiguous()
        and query_start_loc.numel() >= 1
    ):
        return False
    rows = output.numel()
    if rows:
        ends = query_start_loc.numel() - 1
        _request_rows[(triton.cdiv(rows, 256),)](
            query_start_loc,
            output,
            rows,
            ends,
            max(num_reqs - 1, 0),
            max(1, ends.bit_length()),
            256,
        )
    return True
