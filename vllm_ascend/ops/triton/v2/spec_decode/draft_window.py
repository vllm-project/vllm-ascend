# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Plan private eager Draft reads from current device lengths."""

from collections.abc import Callable

import torch
from vllm.triton_utils import tl, triton

PLAN_BLOCK_SIZE = 256
GATHER_TOKEN_BLOCK_SIZE = 8


@triton.jit
def _pack_draft_query_indices(offsets, packed, restored, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    begin = tl.load(offsets + row).to(tl.int64)
    end = tl.load(offsets + row + 1).to(tl.int64)
    width = end - begin
    padding = WIDTH - width
    tl.store(packed + row * WIDTH + local, begin + tl.maximum(local - padding, 0), mask=local < WIDTH)
    tl.store(restored + begin + local, row * WIDTH + padding + local, mask=local < width)


def pack_draft_query_indices(
    offsets: torch.Tensor, total: int, query_width: int, *, retain: Callable[..., None] | None = None
):
    """Right-align current queries without reading device boundaries on Host."""
    n = offsets.numel() - 1
    packed = torch.empty((n, query_width), dtype=torch.int64, device=offsets.device)
    restored = torch.empty(total, dtype=torch.int64, device=offsets.device)
    if retain is not None:
        retain(offsets, packed, restored)
    _pack_draft_query_indices[(n,)](offsets, packed, restored, query_width, triton.next_power_of_2(query_width))
    return packed, restored


@triton.jit
def _gather_draft_kv_fp32(
    slots,
    keys,
    values,
    out_keys,
    out_values,
    KEY_PAGE_STRIDE: tl.constexpr,
    VALUE_PAGE_STRIDE: tl.constexpr,
    BLOCK_SPAN: tl.constexpr,
    TOKEN_SIZE: tl.constexpr,
    TOKEN_PADDING: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
):
    # Infer allocation extent from the launch grid. Only the final tile's
    # padding specializes this kernel, not each Full-prefill capacity.
    total_tokens = tl.num_programs(0) * TOKEN_BLOCK - TOKEN_PADDING
    token = tl.program_id(0) * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    channel = tl.arange(0, CHANNEL_BLOCK)
    slot = tl.load(slots + token, mask=token < total_tokens, other=0).to(tl.int64)
    page, offset = slot // BLOCK_SPAN, slot % BLOCK_SPAN
    # Compute each physical token address once, then broadcast its channels.
    # The flat version repeated the slot load and page arithmetic per channel.
    valid = (token[:, None] < total_tokens) & (channel[None, :] < TOKEN_SIZE)
    output_offset = token[:, None] * TOKEN_SIZE + channel[None, :]
    key_offset = (page * KEY_PAGE_STRIDE + offset * TOKEN_SIZE)[:, None] + channel[None, :]
    key = tl.load(keys + key_offset, mask=valid, other=0)
    tl.store(out_keys + output_offset, key.to(tl.float32), mask=valid)
    value_offset = (page * VALUE_PAGE_STRIDE + offset * TOKEN_SIZE)[:, None] + channel[None, :]
    value = tl.load(values + value_offset, mask=valid, other=0)
    tl.store(out_values + output_offset, value.to(tl.float32), mask=valid)


def gather_draft_kv_fp32(
    slots: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    retain: Callable[..., None] | None = None,
    output_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather validated ND token rows directly into private native FP32 operands.

    Tensor pointers preserve their own storage offsets; K/V page strides can
    differ. The caller validates contiguous token rows and owns both outputs
    until reader completion. Slots already exclude every uninitialized tail.
    """
    if not slots.is_contiguous():
        raise ValueError("Fused Draft gather requires contiguous private slots")
    shape = (*slots.shape, *keys.shape[2:])
    if output_buffers is None:
        gathered_keys = torch.empty(shape, dtype=torch.float32, device=keys.device)
        gathered_values = torch.empty_like(gathered_keys)
    else:
        import torch_npu

        if len(output_buffers) != 2 or any(
            buffer.shape != shape
            or buffer.dtype != torch.float32
            or buffer.device != keys.device
            or not buffer.is_contiguous()
            or torch_npu.get_npu_format(buffer) not in (0, 2)
            for buffer in output_buffers
        ):
            raise ValueError("Reusable Draft gather outputs must match contiguous ND FP32 operands")
        gathered_keys, gathered_values = output_buffers
        regions = []
        for tensor in (slots, keys, values, gathered_keys, gathered_values):
            storage = tensor.untyped_storage()
            regions.append((storage.data_ptr(), storage.nbytes()))
        for index in (3, 4):
            start, size = regions[index]
            if any(start < other + count and other < start + size for other, count in regions[:index]):
                raise ValueError("Reusable Draft outputs must not alias Full KV, slots or one another")
    if retain is not None:
        retain(slots, keys, values, gathered_keys, gathered_values)
    token_size = keys.shape[2] * keys.shape[3]
    _gather_draft_kv_fp32[(triton.cdiv(slots.numel(), GATHER_TOKEN_BLOCK_SIZE),)](
        slots,
        keys,
        values,
        gathered_keys,
        gathered_values,
        keys.stride(0),
        values.stride(0),
        keys.shape[1],
        token_size,
        -slots.numel() % GATHER_TOKEN_BLOCK_SIZE,
        GATHER_TOKEN_BLOCK_SIZE,
        triton.next_power_of_2(token_size),
    )
    return gathered_keys, gathered_values


@triton.jit
def _plan_draft_window(
    table,
    rows,
    exact,
    query_ends,
    frozen,
    full_rows,
    query_offsets,
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
    CAPACITY_PADDING: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_FROZEN: tl.constexpr,
    HAS_QUERY_ENDS: tl.constexpr,
    HAS_FULL_ROWS: tl.constexpr,
    HAS_QUERY_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    capacity = tl.num_programs(1) * BLOCK - CAPACITY_PADDING
    req = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    query = tl.program_id(2)
    valid = column < capacity
    end = tl.load(exact + req * EXACT_STRIDE).to(tl.int64)
    if HAS_FROZEN:
        start = tl.load(frozen + req * FROZEN_STRIDE).to(tl.int64)
    else:
        start = tl.maximum(end - WINDOW, 0) // ANCHOR_SPAN * ANCHOR_SPAN
    if HAS_FULL_ROWS:
        start = tl.where(tl.load(full_rows + req) != 0, 0, start)
    length = end - start
    if tl.program_id(1) == 0 and query == 0:
        tl.store(starts + req, start)
        tl.store(visible + req, length)
    # Only the first query writes the shared token read set. Capacity padding
    # repeats a valid token, never an unwritten or NaN Full-cache tail.
    if query == 0:
        # Keep stores linear within each tile. Flattening query and column
        # then taking modulo CAPACITY makes the NPU compiler lower this to
        # scalar scatter stores protected by a global cross-block lock.
        row = tl.load(rows + req * ROW_STRIDE)
        absolute = start + tl.minimum(column, length - 1)
        block_column = absolute // BLOCK_SPAN
        physical = tl.load(table + row * TABLE_ROW_STRIDE + block_column * TABLE_COLUMN_STRIDE, mask=valid, other=0).to(
            tl.int64
        )
        tl.store(slots + req * capacity + column, physical * BLOCK_SPAN + absolute % BLOCK_SPAN, mask=valid)
    excluded = column >= length
    if CAUSAL:
        query_end = tl.load(query_ends + req * QUERY_END_STRIDE).to(tl.int64) if HAS_QUERY_ENDS else end
        excluded = excluded | (column > query_end - start - QUERY_WIDTH + query)
    if HAS_QUERY_OFFSETS:
        query_length = tl.load(query_offsets + req + 1) - tl.load(query_offsets + req)
        excluded = excluded | (query < QUERY_WIDTH - query_length)
    tl.store(mask + (req * QUERY_WIDTH + query) * capacity + column, excluded, mask=valid)


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
    full_rows: torch.Tensor | None = None,
    query_offsets: torch.Tensor | None = None,
    retain: Callable[..., None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return owned slots/S/V/mask; Host values are allocation geometry only."""
    n = exact.numel()
    slots = torch.empty((n, capacity), dtype=torch.int64, device=exact.device)
    starts, visible = torch.empty_like(exact), torch.empty_like(exact)
    mask = torch.empty((n, 1, query_width, capacity), dtype=torch.bool, device=exact.device)
    if retain is not None:
        # A launcher can raise after submitting device work. The proposal
        # must already own every private output when that happens, including
        # when recording the exception-path completion subsequently fails.
        retain(table, rows, exact, frozen, query_ends, full_rows, query_offsets, slots, starts, visible, mask)
    _plan_draft_window[(n, triton.cdiv(capacity, PLAN_BLOCK_SIZE), query_width)](
        table,
        rows,
        exact,
        exact if query_ends is None else query_ends,
        exact if frozen is None else frozen,
        exact if full_rows is None else full_rows,
        exact if query_offsets is None else query_offsets,
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
        -capacity % PLAN_BLOCK_SIZE,
        query_width,
        causal,
        frozen is not None,
        query_ends is not None and query_ends is not exact,
        full_rows is not None,
        query_offsets is not None,
        PLAN_BLOCK_SIZE,
    )
    return slots, starts, visible, mask
