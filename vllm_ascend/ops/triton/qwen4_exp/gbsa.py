# SPDX-License-Identifier: Apache-2.0
"""Integer-only preparation for the Qwen4Exp GBSA input contract."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_qsa_lengths_kernel(
    lengths,
    compressed_lengths,
    residual_lengths,
    requests,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < requests
    values = tl.load(lengths + offsets, mask=mask, other=0).to(tl.int32)
    tl.store(compressed_lengths + offsets, values // COMPRESS_RATIO, mask=mask)
    tl.store(residual_lengths + offsets, values % COMPRESS_RATIO, mask=mask)


@triton.jit
def _prepare_qsa_gbsa_shared_kernel(
    positions,
    lengths,
    requests,
    table,
    visible_lengths,
    row_table,
    position_stride,
    length_stride,
    request_stride,
    table_row_stride,
    table_column_stride,
    requests_count,
    rows,
    table_columns,
    TABLE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    request = tl.load(requests + row * request_stride).to(tl.int64)
    request = tl.where(request < 0, request + requests_count, request)
    valid_request = (request >= 0) & (request < requests_count)
    safe_request = tl.minimum(tl.maximum(request, 0), requests_count - 1)
    position = tl.load(positions + row * position_stride).to(tl.int32)
    length = tl.load(
        lengths + safe_request * length_stride,
        mask=valid_request,
        other=0,
    ).to(tl.int32)
    visible = tl.minimum(tl.maximum(position + 1, 0), length)
    tl.store(visible_lengths + row, visible)

    columns = tl.arange(0, TABLE_BLOCK)
    stored_page = tl.load(
        table + safe_request * table_row_stride + columns * table_column_stride,
        mask=valid_request & (columns < table_columns),
        other=-1,
    )
    tl.store(
        row_table + row * table_columns + columns,
        stored_page,
        mask=columns < table_columns,
    )


@triton.jit
def _pack_qsa_gbsa_indices_kernel(
    indices,
    visible_lengths,
    packed_indices,
    counts,
    indices_row_stride,
    indices_column_stride,
    rows,
    GROUPS: tl.constexpr,
    OUTPUT_GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    KV_HEADS: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, GROUP_BLOCK)
    visible = tl.load(visible_lengths + row).to(tl.int32)
    group = tl.load(
        indices + row * indices_row_stride + columns * indices_column_stride,
        mask=columns < GROUPS,
        other=-1,
    ).to(tl.int32)
    valid = (columns < GROUPS) & (group >= 0) & (group < visible // 4)
    count = tl.sum(valid.to(tl.int32), 0)
    has_tail = visible % 4 != 0
    tail_slot = tl.minimum(count, OUTPUT_GROUPS - has_tail.to(tl.int32))
    if tl.sum(((columns < count) & (~valid)).to(tl.int32), 0) == 0:
        packed = tl.where(has_tail & (columns == tail_slot), visible // 4, group)
        for head in tl.static_range(KV_HEADS):
            tl.store(
                packed_indices + (head * rows + row) * OUTPUT_GROUPS + columns,
                packed,
                mask=columns < GROUPS,
            )
    else:
        prefix = tl.cumsum(valid.to(tl.int32), 0)
        destination = tl.where(valid, prefix - 1, count + columns - prefix)
        packed = tl.where(has_tail & (destination == tail_slot), visible // 4, group)
        for head in tl.static_range(KV_HEADS):
            tl.store(
                packed_indices + (head * rows + row) * OUTPUT_GROUPS + destination,
                packed,
                mask=columns < GROUPS,
            )
    for head in tl.static_range(KV_HEADS):
        if OUTPUT_GROUPS > GROUPS:
            extra = tl.where(has_tail & (tail_slot == GROUPS), visible // 4, -1)
            tl.store(packed_indices + (head * rows + row) * OUTPUT_GROUPS + GROUPS, extra)
        tl.store(counts + head * rows + row, tail_slot + has_tail.to(tl.int32))


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


def prepare_qsa_lengths(
    lengths: torch.Tensor,
    compressed_lengths: torch.Tensor,
    residual_lengths: torch.Tensor,
    requests: int,
    compress_ratio: int = 4,
) -> None:
    """Prepare LI-v2 quotient and remainder arrays in one launch."""
    if requests < 0 or requests > lengths.shape[0]:
        raise ValueError("QSA prepared request count exceeds length capacity")
    if compressed_lengths.shape[0] < requests or residual_lengths.shape[0] < requests:
        raise ValueError("QSA prepared length buffers are too small")
    if requests == 0:
        return
    block = 128
    _prepare_qsa_lengths_kernel[(triton.cdiv(requests, block),)](
        lengths,
        compressed_lengths,
        residual_lengths,
        requests,
        COMPRESS_RATIO=compress_ratio,
        BLOCK=block,
        num_warps=1,
    )


def prepare_qsa_gbsa_shared(
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    visible_lengths: torch.Tensor,
    row_block_table: torch.Tensor,
    rows: int,
) -> None:
    """Prepare row metadata shared by every QSA layer in the model step."""
    if rows < 0 or rows > logical_positions.shape[0] or rows > token_to_req.shape[0]:
        raise ValueError("QSA prepared row count exceeds input capacity")
    if visible_lengths.shape[0] < rows:
        raise ValueError("QSA visible-length buffer is too small")
    if row_block_table.shape[0] < rows or row_block_table.shape[1] != block_table.shape[1]:
        raise ValueError("QSA row block-table buffer has an invalid shape")
    if rows == 0:
        return
    columns = block_table.shape[1]
    _prepare_qsa_gbsa_shared_kernel[(rows,)](
        logical_positions,
        seq_lens,
        token_to_req,
        block_table,
        visible_lengths,
        row_block_table,
        logical_positions.stride(0),
        seq_lens.stride(0),
        token_to_req.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        block_table.shape[0],
        rows,
        columns,
        TABLE_BLOCK=triton.next_power_of_2(columns),
        num_warps=1,
    )


def pack_qsa_gbsa_indices(
    compact_indices: torch.Tensor,
    visible_lengths: torch.Tensor,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack only layer-dependent LI-v2 groups using prepared row metadata."""
    rows = compact_indices.shape[0]
    groups = compact_indices.shape[1] - 3
    output_groups = groups + (groups < 512)
    sparse_idx = torch.empty(
        (num_kv_heads, rows, output_groups),
        dtype=torch.int32,
        device=compact_indices.device,
    )
    sparse_count = torch.empty(
        (num_kv_heads, rows),
        dtype=torch.int32,
        device=compact_indices.device,
    )
    if rows == 0:
        return sparse_idx, sparse_count
    _pack_qsa_gbsa_indices_kernel[(rows,)](
        compact_indices,
        visible_lengths,
        sparse_idx,
        sparse_count,
        compact_indices.stride(0),
        compact_indices.stride(1),
        rows,
        GROUPS=groups,
        OUTPUT_GROUPS=output_groups,
        GROUP_BLOCK=triton.next_power_of_2(groups),
        KV_HEADS=num_kv_heads,
        num_warps=1,
    )
    return sparse_idx, sparse_count


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


__all__ = [
    "pack_qsa_gbsa_indices",
    "prepare_qsa_gbsa_inputs",
    "prepare_qsa_gbsa_shared",
    "prepare_qsa_lengths",
]
