# SPDX-License-Identifier: Apache-2.0

"""Build the LightningIndexer ABI from current device-side QSA metadata."""

import math

import torch
from vllm.triton_utils import tl, triton

_HEAD_DIM = 128
_COMPRESS_RATIO = 4
_SUPPORTED_HEAD_COUNTS = (8, 16, 24, 32, 64)


@triton.jit
def _prepare_lightning_kernel(
    query_ptr,
    positions_ptr,
    lengths_ptr,
    owners_ptr,
    table_ptr,
    padded_query_ptr,
    weights_ptr,
    visible_ptr,
    cu_query_ptr,
    row_table_ptr,
    stride_q_row,
    stride_q_head,
    stride_pos,
    stride_len,
    stride_owner,
    stride_table_row,
    stride_table_col,
    REQUESTS: tl.constexpr,
    HEADS: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    COLUMNS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_TABLE: tl.constexpr,
    SCORE_WEIGHT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PAD_QUERY: tl.constexpr,
):
    row = tl.program_id(0)
    owner = tl.load(owners_ptr + row * stride_owner).to(tl.int64)
    # Match advanced indexing for valid negative request indices as well.
    owner = tl.where(owner < 0, owner + REQUESTS, owner)
    valid = (owner >= 0) & (owner < REQUESTS)
    safe_owner = tl.minimum(tl.maximum(owner, 0), REQUESTS - 1)
    length = tl.load(lengths_ptr + safe_owner * stride_len, mask=valid, other=0).to(tl.int64)
    position_end = tl.load(positions_ptr + row * stride_pos).to(tl.int64) + 1
    # Triton's signed division truncates; torch.floor_divide rounds down.
    position_groups = position_end // COMPRESS_RATIO - ((position_end < 0) & (position_end % COMPRESS_RATIO != 0)).to(
        tl.int64
    )
    length_groups = length // COMPRESS_RATIO - ((length < 0) & (length % COMPRESS_RATIO != 0)).to(tl.int64)
    tl.store(visible_ptr + row, tl.where(valid, tl.minimum(position_groups, length_groups), 0))
    tl.store(cu_query_ptr + row, row + 1)
    cols = tl.arange(0, BLOCK_TABLE)
    blocks = tl.load(
        table_ptr + safe_owner * stride_table_row + cols * stride_table_col, mask=valid & (cols < COLUMNS), other=0
    )
    tl.store(row_table_ptr + row * COLUMNS + cols, blocks, mask=cols < COLUMNS)
    head = tl.arange(0, BLOCK_HEADS)
    tl.store(weights_ptr + row * PADDED_HEADS + head, tl.where(head < HEADS, SCORE_WEIGHT, 0), mask=head < PADDED_HEADS)
    if PAD_QUERY:
        offsets = tl.arange(0, BLOCK_Q)
        source = query_ptr + row * stride_q_row + offsets // HEAD_DIM * stride_q_head + offsets % HEAD_DIM
        values = tl.load(source, mask=offsets < HEADS * HEAD_DIM, other=0)
        tl.store(
            padded_query_ptr + row * PADDED_HEADS * HEAD_DIM + offsets, values, mask=offsets < PADDED_HEADS * HEAD_DIM
        )


def prepare_qsa_lightning_inputs(
    query: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return query, weights, visible groups, query ends and per-row blocks.

    Shapes are checked on the host; all metadata values are read on device on
    every replay. Valid owners follow torch advanced-index semantics. Invalid
    owners are masked to empty rows instead of forming out-of-bounds pointers.
    No scheduler metadata, request ownership or visible length is cached.
    """
    if (
        query.ndim != 3
        or query.dtype != torch.bfloat16
        or query.shape[2] != _HEAD_DIM
        or not 0 < query.shape[1] <= _SUPPORTED_HEAD_COUNTS[-1]
        or query.stride(-1) != 1
    ):
        raise ValueError("QSA Lightning preparation requires BF16 [rows, 1..64 heads, 128] with contiguous head rows")
    rows, heads, _ = query.shape
    if query_positions.shape != (rows,) or token_to_req.shape != (rows,):
        raise ValueError("QSA Lightning positions and owners must match query rows")
    if (
        block_table.ndim != 2
        or sequence_lengths.shape != (block_table.shape[0],)
        or (rows and not all(block_table.shape))
    ):
        raise ValueError("QSA Lightning block table must have a nonempty row per request")
    for tensor in (query_positions, sequence_lengths, token_to_req, block_table):
        if tensor.dtype not in (torch.int32, torch.int64) or tensor.device != query.device:
            raise ValueError("QSA Lightning metadata must be device-local int32 or int64")
    padded_heads = next(h for h in _SUPPORTED_HEAD_COUNTS if h >= heads)
    padded_query = (
        torch.empty((rows, padded_heads, _HEAD_DIM), dtype=query.dtype, device=query.device)
        if padded_heads != heads
        else query
    )
    weights = torch.empty((rows, padded_heads), dtype=query.dtype, device=query.device)
    visible = torch.empty(rows, dtype=torch.int32, device=query.device)
    cu_query = torch.empty_like(visible)
    row_table = torch.empty((rows, block_table.shape[1]), dtype=block_table.dtype, device=query.device)
    if rows:
        _prepare_lightning_kernel[(rows,)](
            query,
            query_positions,
            sequence_lengths,
            token_to_req,
            block_table,
            padded_query,
            weights,
            visible,
            cu_query,
            row_table,
            query.stride(0),
            query.stride(1),
            query_positions.stride(0),
            sequence_lengths.stride(0),
            token_to_req.stride(0),
            block_table.stride(0),
            block_table.stride(1),
            REQUESTS=sequence_lengths.shape[0],
            HEADS=heads,
            PADDED_HEADS=padded_heads,
            COLUMNS=block_table.shape[1],
            BLOCK_Q=triton.next_power_of_2(padded_heads * _HEAD_DIM),
            BLOCK_HEADS=triton.next_power_of_2(padded_heads),
            BLOCK_TABLE=triton.next_power_of_2(block_table.shape[1]),
            SCORE_WEIGHT=1.0 / math.sqrt(_HEAD_DIM),
            HEAD_DIM=_HEAD_DIM,
            COMPRESS_RATIO=_COMPRESS_RATIO,
            PAD_QUERY=padded_heads != heads,
            num_warps=4,
        )
    return padded_query, weights, visible, cu_query, row_table
