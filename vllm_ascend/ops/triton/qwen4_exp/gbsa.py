# SPDX-License-Identifier: Apache-2.0
"""Integer-only preparation for the Qwen4Exp GBSA input contract."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_qsa_gbsa_rows_kernel(
    indices,
    positions,
    lengths,
    requests,
    table,
    packed_indices,
    counts,
    visible_lengths,
    cu_query,
    row_table,
    indices_row_stride,
    indices_column_stride,
    table_row_stride,
    table_column_stride,
    position_stride,
    length_stride,
    request_stride,
    requests_count,
    rows,
    table_columns,
    GROUPS: tl.constexpr,
    OUTPUT_GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    KV_HEADS: tl.constexpr,
    TABLE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, GROUP_BLOCK)
    request = tl.load(requests + row * request_stride).to(tl.int64)
    # Match torch indexing for negative request IDs. Callers must provide in-range IDs.
    request = tl.where(request < 0, request + requests_count, request)
    position = tl.load(positions + row * position_stride).to(tl.int32)
    length = tl.load(lengths + request * length_stride).to(tl.int32)
    visible = tl.minimum(tl.maximum(position + 1, 0), length)
    group = tl.load(
        indices + row * indices_row_stride + columns * indices_column_stride, columns < GROUPS, other=-1
    ).to(tl.int32)
    valid = (columns < GROUPS) & (group >= 0) & (group < visible // 4)
    count = tl.sum(valid.to(tl.int32), 0)
    has_tail = visible % 4 != 0
    tail_slot = tl.minimum(count, OUTPUT_GROUPS - has_tail.to(tl.int32))
    if tl.sum(((columns < count) & (~valid)).to(tl.int32), 0) == 0:
        # LI-v2 packing normally already has a valid prefix. Preserve its
        # selected-group order directly, without sorting or a prefix scan.
        packed = tl.where(has_tail & (columns == tail_slot), visible // 4, group)
        for head in tl.static_range(KV_HEADS):
            tl.store(packed_indices + (head * rows + row) * OUTPUT_GROUPS + columns, packed, columns < GROUPS)
    else:
        # Use the native scan: gather-based scans can miscompile on Triton-Ascend 3.6.
        prefix = tl.cumsum(valid.to(tl.int32), 0)
        destination = tl.where(valid, prefix - 1, count + columns - prefix)
        packed = tl.where(has_tail & (destination == tail_slot), visible // 4, group)
        for head in tl.static_range(KV_HEADS):
            tl.store(packed_indices + (head * rows + row) * OUTPUT_GROUPS + destination, packed, columns < GROUPS)
    for head in tl.static_range(KV_HEADS):
        # Append the residual slot when capacity allows, otherwise replace the
        # final selection exactly as the PR adapter does at the 512-entry limit.
        if OUTPUT_GROUPS > GROUPS:
            extra = tl.where(has_tail & (tail_slot == GROUPS), visible // 4, -1)
            tl.store(packed_indices + (head * rows + row) * OUTPUT_GROUPS + GROUPS, extra)
        tl.store(counts + head * rows + row, tail_slot + has_tail.to(tl.int32))
    tl.store(visible_lengths + row, visible)
    tl.store(cu_query + row, row.to(tl.int64))
    if row == 0:
        tl.store(cu_query + rows, rows)
    page_columns = tl.arange(0, TABLE_BLOCK)
    stored_page = tl.load(
        table + request * table_row_stride + page_columns * table_column_stride,
        page_columns < table_columns,
        other=-1,
    )
    tl.store(row_table + row * table_columns + page_columns, stored_page, page_columns < table_columns)


def prepare_qsa_gbsa_inputs(
    compact_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce the same integer tensors as the original GBSA torch pipeline.

    The block table already addresses 128-token kernel pages. Residual blocks
    remain explicit (residual_block_mode=0), including the 512-entry limit.
    Every output element is written before use. No device-to-host reads or
    process-global tensor cache are introduced; inputs can change every call.
    """
    rows = compact_indices.shape[0]
    groups = compact_indices.shape[1] - 3
    columns = block_table.shape[1]
    output_groups = groups + (groups < 512)
    sparse_idx = torch.empty((num_kv_heads, rows, output_groups), dtype=torch.int32, device=compact_indices.device)
    sparse_count = torch.empty((num_kv_heads, rows), dtype=torch.int32, device=compact_indices.device)
    visible = torch.empty((rows,), dtype=torch.int32, device=compact_indices.device)
    cu_query = torch.empty((rows + 1,), dtype=torch.int64, device=compact_indices.device)
    row_table = torch.empty((rows, columns), dtype=block_table.dtype, device=block_table.device)
    if rows == 0:
        cu_query.zero_()
        return sparse_idx, sparse_count, visible, cu_query, row_table
    group_block = triton.next_power_of_2(groups)
    _prepare_qsa_gbsa_rows_kernel[(rows,)](
        compact_indices,
        logical_positions,
        seq_lens,
        token_to_req,
        block_table,
        sparse_idx,
        sparse_count,
        visible,
        cu_query,
        row_table,
        compact_indices.stride(0),
        compact_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        logical_positions.stride(0),
        seq_lens.stride(0),
        token_to_req.stride(0),
        block_table.shape[0],
        rows,
        columns,
        GROUPS=groups,
        OUTPUT_GROUPS=output_groups,
        GROUP_BLOCK=group_block,
        KV_HEADS=num_kv_heads,
        TABLE_BLOCK=triton.next_power_of_2(columns),
        num_warps=1,
    )
    return sparse_idx, sparse_count, visible, cu_query, row_table


@triton.jit
def _pack_resident_gbsa_tiles(
    resident,
    selected,
    counts,
    visible,
    resident_table,
    key_out,
    value_out,
    packed_indices,
    packed_lengths,
    packed_table,
    ROWS: tl.constexpr,
    GROUPS: tl.constexpr,
    SOURCE_PAGES: tl.constexpr,
    PAGES: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHANNELS: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    PAGE_BLOCK: tl.constexpr,
    TILE_TOKENS: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    token = tile * TILE_TOKENS + tl.arange(0, TILE_TOKENS)
    group_count = tl.load(counts + row)
    tail = tl.load(visible + row) % 4
    length = (group_count - (tail != 0).to(tl.int32)) * 4 + tail
    live = token < length
    group = tl.load(selected + row * GROUPS + token // 4, live, other=0)
    logical = group * 4 + token % 4
    page = tl.load(resident_table + row * SOURCE_PAGES + logical // 128, live, other=0)
    head_channel = tl.arange(0, CHANNEL_BLOCK)
    channel_valid = head_channel < CHANNELS
    source_channel = head_channel // HEAD_DIM * (2 * HEAD_DIM) + head_channel % HEAD_DIM
    source = page.to(tl.int64) * PAGE_STRIDE + (logical % 128) * TOKEN_STRIDE
    mask = live[:, None] & channel_valid[None, :]
    keys = tl.load(resident + source[:, None] + source_channel[None, :], mask, other=0)
    values = tl.load(resident + source[:, None] + source_channel[None, :] + HEAD_DIM, mask, other=0)
    dest = ((row * PAGES * 128 + token) * CHANNELS)[:, None] + head_channel[None, :]
    tl.store(key_out + dest, keys, channel_valid[None, :])
    tl.store(value_out + dest, values, channel_valid[None, :])

    if tile == 0:
        groups = tl.arange(0, GROUP_BLOCK)
        for head in tl.static_range(KV_HEADS):
            tl.store(packed_indices + (head * ROWS + row) * GROUPS + groups, groups, groups < GROUPS)
        pages = tl.arange(0, PAGE_BLOCK)
        tl.store(packed_table + row * PAGES + pages, row * PAGES + pages, pages < PAGES)
        tl.store(packed_lengths + row, length)


def pack_resident_gbsa_inputs(
    resident: torch.Tensor,
    selected: torch.Tensor,
    counts: torch.Tensor,
    visible: torch.Tensor,
    resident_table: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage selected groups from resident pages without resolving individual tokens."""
    heads, rows, groups = selected.shape
    pages = triton.cdiv(groups * 4, 128)
    keys = torch.empty((rows * pages, 128, heads, head_dim), device=resident.device, dtype=resident.dtype)
    values = torch.empty_like(keys)
    packed_indices = torch.empty_like(selected)
    packed_lengths = torch.empty_like(visible)
    packed_table = resident_table.new_empty((rows, pages))
    tile_tokens = 16
    _pack_resident_gbsa_tiles[(rows, pages * 128 // tile_tokens)](
        resident,
        selected,
        counts,
        visible,
        resident_table,
        keys,
        values,
        packed_indices,
        packed_lengths,
        packed_table,
        rows,
        groups,
        resident_table.shape[1],
        pages,
        resident.stride(0),
        resident.stride(1),
        heads,
        head_dim,
        heads * head_dim,
        triton.next_power_of_2(heads * head_dim),
        triton.next_power_of_2(groups),
        triton.next_power_of_2(pages),
        tile_tokens,
        num_warps=1,
    )
    return keys, values, packed_indices, packed_lengths, packed_table
