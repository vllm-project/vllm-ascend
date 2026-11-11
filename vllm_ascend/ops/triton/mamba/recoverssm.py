# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared RecoverSSM commit planning and convolution compaction."""

from vllm.triton_utils import tl, triton


@triton.jit
def _copy_conv_state_tile(
    source_ptr,
    final_ptr,
    boundary_ptr,
    channel_offsets,
    history_offsets,
    token_stride,
    commit_len,
    boundary_recovery_len,
    mask,
    boundary_valid,
    ALIGN_MODE: tl.constexpr,
):
    final_values = tl.load(
        source_ptr + channel_offsets + (commit_len - 1 + history_offsets) * token_stride,
        mask=mask,
    )
    if ALIGN_MODE:
        boundary_values = tl.load(
            source_ptr + channel_offsets + (boundary_recovery_len - 1 + history_offsets) * token_stride,
            mask=mask & boundary_valid,
        )
        tl.store(
            boundary_ptr + channel_offsets + history_offsets * token_stride,
            boundary_values,
            mask=mask & boundary_valid,
        )
    tl.store(
        final_ptr + channel_offsets + history_offsets * token_stride,
        final_values,
        mask=mask,
    )


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "ALIGN_MODE": lambda args: args["block_table_ptr"] is not None,
    }
)
@triton.jit
def prepare_commit_plan_kernel(
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
    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted)
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc)
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)

    final_state_idx = source_state_idx
    boundary_state_idx = null_block_id
    boundary_recovery_len = 0
    if ALIGN_MODE:
        num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(tl.int32)
        final_num_computed = num_computed + commit_len
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


@triton.jit
def compact_conv_state_kernel(
    conv_state_ref_ptr,
    conv_state_base_addrs_ptr,
    conv_state_block_strides_ptr,
    conv_state_dim_strides_ptr,
    conv_state_token_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    conv_dim,
    conv_history_len,
    stride_state_indices,
    BLOCK_D: tl.constexpr,
    BLOCK_HISTORY: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    DIM_STRIDE: tl.constexpr = None,
    USE_1D_CHANNEL_GATHER: tl.constexpr = False,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_l = tl.program_id(2)
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if source_state_idx <= null_block_id:
        return

    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    base_addr = tl.load(conv_state_base_addrs_ptr + pid_l)
    block_stride = tl.load(conv_state_block_strides_ptr + pid_l)
    if DIM_STRIDE is None:
        dim_stride = tl.load(conv_state_dim_strides_ptr + pid_l)
    else:
        dim_stride = DIM_STRIDE
    token_stride = tl.load(conv_state_token_strides_ptr + pid_l)
    conv_state_ptr = base_addr.to(tl.pointer_type(conv_state_ref_ptr.dtype.element_ty))
    source_ptr = conv_state_ptr + source_state_idx * block_stride
    final_ptr = conv_state_ptr + final_state_idx * block_stride

    boundary_ptr = None
    if ALIGN_MODE:
        # Clamp the address only; access masks still key off the raw index.
        boundary_ptr = conv_state_ptr + tl.maximum(boundary_state_idx, 0) * block_stride

    if USE_1D_CHANNEL_GATHER and (DIM_STRIDE is not None and DIM_STRIDE != 1):
        # A3 Triton 3.2 requires one-dimensional gathers for strided channels.
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        for history in tl.static_range(BLOCK_HISTORY):
            mask = (offs_d < conv_dim) & (history < conv_history_len)
            _copy_conv_state_tile(
                source_ptr,
                final_ptr,
                boundary_ptr,
                offs_d * dim_stride,
                history,
                token_stride,
                commit_len,
                boundary_recovery_len,
                mask,
                boundary_state_idx > null_block_id,
                ALIGN_MODE,
            )
        return
    offs_d = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D))[None, :]
    offs_h = tl.arange(0, BLOCK_HISTORY)[:, None]
    mask = (offs_d < conv_dim) & (offs_h < conv_history_len)
    _copy_conv_state_tile(
        source_ptr,
        final_ptr,
        boundary_ptr,
        offs_d * dim_stride,
        offs_h,
        token_stride,
        commit_len,
        boundary_recovery_len,
        mask,
        boundary_state_idx > null_block_id,
        ALIGN_MODE,
    )
