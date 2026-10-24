# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Plan private eager Draft reads from current device lengths."""

import torch
from vllm.triton_utils import tl, triton

PLAN_BLOCK_SIZE = 256


@triton.jit
def _plan_draft_window(
    table,
    rows,
    exact,
    query_ends,
    frozen,
    slots,
    starts,
    visible,
    mask,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COLUMN_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    EXACT_STRIDE: tl.constexpr,
    QUERY_END_STRIDE: tl.constexpr,
    FROZEN_STRIDE: tl.constexpr,
    BLOCK_SPAN: tl.constexpr,
    ANCHOR_SPAN: tl.constexpr,
    WINDOW: tl.constexpr,
    CAPACITY: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_FROZEN: tl.constexpr,
    HAS_QUERY_ENDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    index = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = index < QUERY_WIDTH * CAPACITY
    query = index // CAPACITY
    column = index % CAPACITY
    end = tl.load(exact + req * EXACT_STRIDE).to(tl.int64)
    if HAS_FROZEN:
        start = tl.load(frozen + req * FROZEN_STRIDE).to(tl.int64)
    else:
        start = tl.maximum(end - WINDOW, 0) // ANCHOR_SPAN * ANCHOR_SPAN
    length = end - start
    if tl.program_id(1) == 0:
        tl.store(starts + req, start)
        tl.store(visible + req, length)
    # Only the first query writes the shared token read set. Capacity padding
    # repeats a valid token, never an unwritten or NaN Full-cache tail.
    first = valid & (query == 0)
    row = tl.load(rows + req * ROW_STRIDE)
    absolute = start + tl.minimum(column, length - 1)
    block_column = absolute // BLOCK_SPAN
    physical = tl.load(table + row * TABLE_ROW_STRIDE + block_column * TABLE_COLUMN_STRIDE, mask=first, other=0).to(
        tl.int64
    )
    tl.store(slots + req * CAPACITY + column, physical * BLOCK_SPAN + absolute % BLOCK_SPAN, mask=first)
    excluded = column >= length
    if CAUSAL:
        query_end = tl.load(query_ends + req * QUERY_END_STRIDE).to(tl.int64) if HAS_QUERY_ENDS else end
        excluded = excluded | (column > query_end - start - QUERY_WIDTH + query)
    tl.store(mask + req * QUERY_WIDTH * CAPACITY + index, excluded, mask=valid)


def plan_draft_window(
    table: torch.Tensor,
    rows: torch.Tensor,
    exact: torch.Tensor,
    frozen: torch.Tensor | None,
    *,
    block_span: int,
    anchor_span: int,
    window: int,
    capacity: int,
    query_width: int,
    causal: bool,
    query_ends: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return owned slots/S/V/mask; Host values are allocation geometry only."""
    n = exact.numel()
    slots = torch.empty((n, capacity), dtype=torch.int64, device=exact.device)
    starts, visible = torch.empty_like(exact), torch.empty_like(exact)
    mask = torch.empty((n, 1, query_width, capacity), dtype=torch.bool, device=exact.device)
    _plan_draft_window[(n, triton.cdiv(query_width * capacity, PLAN_BLOCK_SIZE))](
        table,
        rows,
        exact,
        exact if query_ends is None else query_ends,
        exact if frozen is None else frozen,
        slots,
        starts,
        visible,
        mask,
        table.stride(0),
        table.stride(1),
        rows.stride(0),
        exact.stride(0),
        0 if query_ends is None else query_ends.stride(0),
        0 if frozen is None else frozen.stride(0),
        block_span,
        anchor_span,
        window,
        capacity,
        query_width,
        causal,
        frozen is not None,
        query_ends is not None and query_ends is not exact,
        PLAN_BLOCK_SIZE,
    )
    return slots, starts, visible, mask
