# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare accepted-prefix commit metadata for RecoverSSM."""

import torch
from vllm.triton_utils import tl, triton


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "ALIGN_MODE": lambda args: args["block_table_ptr"] is not None,
    }
)
@triton.jit
def _prepare_commit_plan_kernel(
    num_accepted_ptr,
    request_indices_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    num_computed_ptr,
    commit_lens_ptr,
    state_commit_lens_ptr,
    boundary_commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    mamba_block_size,
    block_table_width,
    stride_num_accepted,
    stride_request_indices,
    stride_state_indices,
    stride_query_start_loc,
    stride_block_table_row,
    stride_block_table_col,
    stride_num_computed,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    spec_idx = tl.program_id(0)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices).to(tl.int64)
    request_idx = spec_idx
    if HAS_REQUEST_INDICES:
        request_idx = tl.load(request_indices_ptr + spec_idx * stride_request_indices).to(tl.int64)
    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted).to(tl.int32)
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc).to(tl.int64)
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)

    final_state_idx = source_state_idx
    boundary_state_idx = null_block_id
    boundary_recovery_len = 0
    if ALIGN_MODE:
        num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(tl.int32)
        final_num_computed = num_computed + commit_len
        # The checkpoint belongs to the page containing the last accepted token.
        final_state_col = tl.minimum(tl.maximum(final_num_computed - 1, 0) // mamba_block_size, block_table_width - 1)
        final_state_idx = tl.load(
            block_table_ptr + request_idx * stride_block_table_row + final_state_col * stride_block_table_col
        ).to(tl.int64)
        next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
        crosses_boundary = final_num_computed >= next_boundary
        boundary_recovery_len = next_boundary - num_computed
        boundary_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + (next_boundary // mamba_block_size - 1) * stride_block_table_col,
            mask=crosses_boundary,
            other=null_block_id,
        ).to(tl.int64)
    valid = (source_state_idx > null_block_id) & (commit_len > 0)
    tl.store(commit_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    has_boundary = (boundary_state_idx > null_block_id) & (boundary_recovery_len > 0)
    # Predicate once per request, rather than reading several metadata arrays in
    # every head/value program merely to discover that it has no work.
    tl.store(state_commit_lens_ptr + spec_idx, tl.where(valid & ~has_boundary, commit_len, 0))
    tl.store(boundary_commit_lens_ptr + spec_idx, tl.where(valid & has_boundary, commit_len, 0))
    tl.store(
        final_state_indices_ptr + spec_idx,
        tl.where(valid, final_state_idx, null_block_id),
    )
    tl.store(
        boundary_state_indices_ptr + spec_idx,
        tl.where(valid, boundary_state_idx, null_block_id),
    )
    tl.store(
        boundary_recovery_lens_ptr + spec_idx,
        tl.where(valid, boundary_recovery_len, 0),
    )


def prepare_commit_plan(
    num_accepted_tokens: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    commit_lens: torch.Tensor,
    state_commit_lens: torch.Tensor,
    boundary_commit_lens: torch.Tensor,
    final_state_indices: torch.Tensor,
    boundary_state_indices: torch.Tensor,
    boundary_recovery_lens: torch.Tensor,
    *,
    spec_query_len: int,
    null_block_id: int,
    request_indices: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    mamba_block_size: int | None = None,
) -> None:
    """Resolve accepted lengths and destinations into reusable plan buffers."""
    batch = state_indices.numel()
    block_table_stride = (0, 0) if block_table is None else block_table.stride()
    _prepare_commit_plan_kernel[(batch,)](
        num_accepted_tokens,
        request_indices,
        state_indices,
        query_start_loc,
        block_table,
        num_computed_tokens,
        commit_lens,
        state_commit_lens,
        boundary_commit_lens,
        final_state_indices,
        boundary_state_indices,
        boundary_recovery_lens,
        null_block_id,
        mamba_block_size or 1,
        block_table.shape[1] if block_table is not None else 1,
        num_accepted_tokens.stride(0),
        request_indices.stride(0) if request_indices is not None else 0,
        state_indices.stride(0),
        query_start_loc.stride(0),
        block_table_stride[0],
        block_table_stride[1],
        num_computed_tokens.stride(0) if num_computed_tokens is not None else 0,
        SPEC_QUERY_LEN=spec_query_len,
        num_warps=1,
    )
