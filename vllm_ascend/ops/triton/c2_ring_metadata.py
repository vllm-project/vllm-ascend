# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish C2 ring controls and source RoPE rows in one NPU launch."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit(
    do_not_specialize=[
        "num_reqs",
        "num_tokens",
        "num_actual_reqs",
        "num_actual_tokens",
        "skip_update",
    ]
)
def _build_c2_ring_metadata_kernel(
    query_start_loc_ptr,
    seq_lens_ptr,
    positions_ptr,
    block_table_ptr,
    source_cos_ptr,
    source_sin_ptr,
    ring_metadata_ptr,
    complete_mask_ptr,
    source_positions_ptr,
    output_cos_ptr,
    output_sin_ptr,
    num_reqs,
    num_tokens,
    num_actual_reqs,
    num_actual_tokens,
    skip_update,
    query_start_stride,
    seq_lens_stride,
    position_stride,
    block_table_row_stride,
    block_table_col_stride,
    source_rope_row_stride,
    source_rope_dim_stride,
    ring_row_stride,
    ring_col_stride,
    complete_stride,
    source_position_stride,
    output_rope_row_stride,
    output_rope_dim_stride,
    REQ_BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    ROPE_BLOCK: tl.constexpr,
):
    program = tl.program_id(0)

    reqs = program * REQ_BLOCK + tl.arange(0, REQ_BLOCK)
    req_mask = reqs < num_reqs
    starts = tl.load(
        query_start_loc_ptr + reqs * query_start_stride,
        mask=req_mask,
        other=0,
    ).to(tl.int32)
    ends = tl.load(
        query_start_loc_ptr + (reqs + 1) * query_start_stride,
        mask=req_mask,
        other=0,
    ).to(tl.int32)
    seq_lens = tl.load(
        seq_lens_ptr + reqs * seq_lens_stride,
        mask=req_mask,
        other=0,
    ).to(tl.int32)
    query_lens = ends - starts
    used = tl.maximum(tl.minimum(ends, num_actual_tokens) - starts, 0)
    live = reqs < num_actual_reqs
    used = tl.where(live & (skip_update == 0), used, 0)
    first_blocks = tl.load(
        block_table_ptr
        + reqs * block_table_row_stride
        + 0 * block_table_col_stride,
        mask=req_mask,
        other=0,
    ).to(tl.int32)
    ring0 = tl.maximum(seq_lens - query_lens, 0)
    ring4 = tl.where(used > 0, first_blocks, 0)
    tl.store(
        ring_metadata_ptr + 0 * ring_row_stride + reqs * ring_col_stride,
        ring0,
        mask=req_mask,
    )
    tl.store(
        ring_metadata_ptr + 1 * ring_row_stride + reqs * ring_col_stride,
        used,
        mask=req_mask,
    )
    tl.store(
        ring_metadata_ptr + 2 * ring_row_stride + reqs * ring_col_stride,
        starts,
        mask=req_mask,
    )
    tl.store(
        ring_metadata_ptr + 3 * ring_row_stride + reqs * ring_col_stride,
        starts,
        mask=req_mask,
    )
    tl.store(
        ring_metadata_ptr + 4 * ring_row_stride + reqs * ring_col_stride,
        ring4,
        mask=req_mask,
    )

    tokens = program * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    token_mask = tokens < num_tokens
    positions = tl.load(
        positions_ptr + tokens * position_stride,
        mask=token_mask,
        other=0,
    ).to(tl.int64)
    valid_end = tl.load(
        query_start_loc_ptr + num_actual_reqs * query_start_stride
    ).to(tl.int32)
    valid_end = tl.minimum(valid_end, num_actual_tokens)
    complete = (
        token_mask
        & (tokens < valid_end)
        & ((positions % 2) == 1)
        & (skip_update == 0)
    )
    source_positions = tl.where(complete, positions - 1, 0)
    tl.store(
        complete_mask_ptr + tokens * complete_stride,
        complete,
        mask=token_mask,
    )
    tl.store(
        source_positions_ptr + tokens * source_position_stride,
        source_positions,
        mask=token_mask,
    )

    dims = tl.arange(0, ROPE_BLOCK)
    rope_mask = token_mask[:, None] & (dims[None, :] < ROPE_DIM)
    source_offsets = (
        source_positions[:, None] * source_rope_row_stride
        + dims[None, :] * source_rope_dim_stride
    )
    output_offsets = (
        tokens[:, None] * output_rope_row_stride
        + dims[None, :] * output_rope_dim_stride
    )
    cos = tl.load(source_cos_ptr + source_offsets, mask=rope_mask)
    sin = tl.load(source_sin_ptr + source_offsets, mask=rope_mask)
    tl.store(output_cos_ptr + output_offsets, cos, mask=rope_mask)
    tl.store(output_sin_ptr + output_offsets, sin, mask=rope_mask)


def build_c2_ring_metadata(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    source_cos: torch.Tensor,
    source_sin: torch.Tensor,
    num_reqs: int,
    num_tokens: int,
    num_actual_reqs: int,
    num_actual_tokens: int,
    *,
    skip_update: bool,
    ring_metadata_output: torch.Tensor,
    complete_mask_output: torch.Tensor,
    source_positions_output: torch.Tensor,
    cos_output: torch.Tensor,
    sin_output: torch.Tensor,
) -> None:
    if ring_metadata_output.shape != (5, num_reqs):
        raise ValueError("ring_metadata_output must have shape [5, num_reqs]")
    if complete_mask_output.shape != (num_tokens,):
        raise ValueError("complete_mask_output must have shape [num_tokens]")
    if source_positions_output.shape != (num_tokens,):
        raise ValueError("source_positions_output must have shape [num_tokens]")
    rope_dim = source_cos.shape[-1]
    if source_sin.shape[-1] != rope_dim:
        raise ValueError("source RoPE tables must have the same width")
    if cos_output.shape != (num_tokens, 1, 1, rope_dim):
        raise ValueError("cos_output has an invalid shape")
    if sin_output.shape != cos_output.shape:
        raise ValueError("sin_output must match cos_output")
    if not num_reqs and not num_tokens:
        return

    req_block = 64
    token_block = 16
    rope_block = triton.next_power_of_2(rope_dim)
    grid = (
        max(
            triton.cdiv(num_reqs, req_block),
            triton.cdiv(num_tokens, token_block),
        ),
    )
    _build_c2_ring_metadata_kernel[grid](
        query_start_loc,
        seq_lens,
        positions,
        block_table,
        source_cos,
        source_sin,
        ring_metadata_output,
        complete_mask_output,
        source_positions_output,
        cos_output,
        sin_output,
        num_reqs,
        num_tokens,
        num_actual_reqs,
        num_actual_tokens,
        int(skip_update),
        query_start_loc.stride(0),
        seq_lens.stride(0),
        positions.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        source_cos.stride(0),
        source_cos.stride(-1),
        ring_metadata_output.stride(0),
        ring_metadata_output.stride(1),
        complete_mask_output.stride(0),
        source_positions_output.stride(0),
        cos_output.stride(0),
        cos_output.stride(-1),
        REQ_BLOCK=req_block,
        TOKEN_BLOCK=token_block,
        ROPE_DIM=rope_dim,
        ROPE_BLOCK=rope_block,
    )
