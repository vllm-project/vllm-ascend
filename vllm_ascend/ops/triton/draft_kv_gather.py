# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather bounded draft KV pages without reading rejected or uninitialized tails."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_draft_kv_kernel(
    key,
    value,
    table,
    lengths,
    out_key,
    out_value,
    out_table,
    key_page_stride: tl.constexpr,
    key_token_stride: tl.constexpr,
    key_hidden_stride: tl.constexpr,
    value_page_stride: tl.constexpr,
    value_token_stride: tl.constexpr,
    value_hidden_stride: tl.constexpr,
    table_row_stride: tl.constexpr,
    table_col_stride: tl.constexpr,
    length_stride: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGES_PER_REQUEST: tl.constexpr,
    COPY_ROWS: tl.constexpr,
    COPY_WIDTH: tl.constexpr,
):
    request = tl.program_id(2)
    logical_page = tl.program_id(1)
    page = request * PAGES_PER_REQUEST + logical_page
    length = tl.load(lengths + request * length_stride)
    valid_rows = tl.minimum(tl.maximum(length - logical_page * PAGE_SIZE, 0), PAGE_SIZE)
    source_page = tl.load(
        table + request * table_row_stride + logical_page * table_col_stride,
        mask=valid_rows > 0,
        other=0,
    ).to(tl.int64)
    row_start = tl.program_id(0) * COPY_ROWS
    key_block = tl.make_block_ptr(
        base=key + source_page * key_page_stride,
        shape=(valid_rows, WIDTH),
        strides=(key_token_stride, key_hidden_stride),
        offsets=(row_start, 0),
        block_shape=(COPY_ROWS, COPY_WIDTH),
        order=(1, 0),
    )
    value_block = tl.make_block_ptr(
        base=value + source_page * value_page_stride,
        shape=(valid_rows, WIDTH),
        strides=(value_token_stride, value_hidden_stride),
        offsets=(row_start, 0),
        block_shape=(COPY_ROWS, COPY_WIDTH),
        order=(1, 0),
    )
    keys = tl.load(key_block, boundary_check=(0, 1), padding_option="zero")
    values = tl.load(value_block, boundary_check=(0, 1), padding_option="zero")
    output_base = page.to(tl.int64) * PAGE_SIZE * WIDTH
    key_output = tl.make_block_ptr(
        base=out_key + output_base,
        shape=(PAGE_SIZE, WIDTH),
        strides=(WIDTH, 1),
        offsets=(row_start, 0),
        block_shape=(COPY_ROWS, COPY_WIDTH),
        order=(1, 0),
    )
    value_output = tl.make_block_ptr(
        base=out_value + output_base,
        shape=(PAGE_SIZE, WIDTH),
        strides=(WIDTH, 1),
        offsets=(row_start, 0),
        block_shape=(COPY_ROWS, COPY_WIDTH),
        order=(1, 0),
    )
    tl.store(key_output, keys, boundary_check=(0, 1))
    tl.store(value_output, values, boundary_check=(0, 1))
    if tl.program_id(0) == 0:
        tl.store(out_table + page, page)


def gather_draft_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_requests: int,
    blocks_per_request: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy only valid KV into private dense pages and emit their block table.

    Exact lengths and page IDs stay on the device. All three cache strides are
    honored, and masked loads prevent NaN/Inf in rejected tails from escaping.
    The caller bounds temporary storage and validates the paged NHD contract.
    """
    page_size, width = key.shape[1:]
    shape = (num_requests * blocks_per_request, page_size, width)
    out_key = torch.empty(shape, dtype=key.dtype, device=key.device)
    out_value = torch.empty(shape, dtype=value.dtype, device=value.device)
    out_table = torch.empty((num_requests, blocks_per_request), dtype=block_table.dtype, device=block_table.device)
    # Budget physical row spans as well as dtype size for strided DMA staging.
    copy_width = triton.next_power_of_2(width)
    physical_row_bytes = copy_width * max(key.stride(2), value.stride(2)) * key.element_size()
    copy_tile_bytes = 8192
    copy_rows = max(1, min(16, copy_tile_bytes // physical_row_bytes))
    _gather_draft_kv_kernel[(triton.cdiv(page_size, copy_rows), blocks_per_request, num_requests)](
        key,
        value,
        block_table,
        seq_lens,
        out_key,
        out_value,
        out_table,
        *key.stride(),
        *value.stride(),
        *block_table.stride(),
        seq_lens.stride(0),
        page_size,
        width,
        blocks_per_request,
        copy_rows,
        copy_width,
    )
    return out_key, out_value, out_table
