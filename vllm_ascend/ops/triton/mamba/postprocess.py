# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/mamba_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def _copy_mamba_state_block(
    state_idx,
    bt_row_idx,
    src_col,
    dst_col,
    token_bias,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
):
    """Copy one Mamba state block using Ascend-compatible pointers."""
    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)

    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)
    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table_typed = group_base_addr.to(tl.pointer_type(tl.int32))
    block_table_base = block_table_typed + bt_row_idx * block_table_stride_req

    dest_block_id = tl.load(block_table_base + dst_col).to(tl.int64)
    dst_addr = state_base_addr + dest_block_id * state_block_stride
    is_conv_state = conv_width > 0

    if CONV_STATE_DIM_FIRST and is_conv_state:
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        dim_rows = tl.load(state_dim_row_count_ptr + state_idx)
        row_stride = tl.load(state_dim_row_stride_ptr + state_idx)
        per_row_bytes = (conv_width - token_bias).to(tl.int64) * state_elem_size
        bias_bytes = token_bias.to(tl.int64) * state_elem_size
        src_block_addr = state_base_addr + src_block_id * state_block_stride
        offsets = tl.arange(0, COPY_BLOCK_SIZE)
        for dim_idx in range(0, dim_rows):
            row_src = src_block_addr + dim_idx * row_stride + bias_bytes
            row_dst = dst_addr + dim_idx * row_stride
            # Hoist pointer casts out of the inner loop for triton-ascend's
            # pointer-axis analysis.
            row_src_ptr = row_src.to(tl.pointer_type(tl.uint8))
            row_dst_ptr = row_dst.to(tl.pointer_type(tl.uint8))
            for offset in range(0, per_row_bytes, COPY_BLOCK_SIZE):
                mask = (offset + offsets) < per_row_bytes
                data = tl.load(row_src_ptr + offset + offsets, mask=mask)
                tl.store(row_dst_ptr + offset + offsets, data, mask=mask)
        return

    if is_conv_state:
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        src_offset = (
            token_bias.to(tl.int64) * state_inner_size * state_elem_size
        )
        src_addr = (
            state_base_addr + src_block_id * state_block_stride + src_offset
        )
        copy_size = (
            (conv_width - token_bias).to(tl.int64)
            * state_inner_size
            * state_elem_size
        )
    else:
        actual_src_block_id = tl.load(
            block_table_base + src_col + token_bias
        ).to(tl.int64)
        src_addr = state_base_addr + actual_src_block_id * state_block_stride
        copy_size = state_inner_size * state_elem_size

    src_ptr = src_addr.to(tl.pointer_type(tl.uint8))
    dst_ptr = dst_addr.to(tl.pointer_type(tl.uint8))
    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    for offset in range(0, copy_size, COPY_BLOCK_SIZE):
        mask = (offset + offsets) < copy_size
        data = tl.load(src_ptr + offset + offsets, mask=mask)
        tl.store(dst_ptr + offset + offsets, data, mask=mask)


@triton.jit
def postprocess_mamba_fused_kernel(
    # Decision inputs (per-request)
    num_accepted_tokens_ptr,
    mamba_state_idx_ptr,
    num_scheduled_tokens_ptr,
    num_computed_tokens_ptr,
    num_draft_tokens_ptr,
    # Per-group block table base addresses: int64[num_groups]. Each entry is
    # the data_ptr of that group's persistent [max_reqs, max_blocks] int32
    # block table.
    block_table_ptrs_ptr,
    block_table_stride_req: tl.int64,  # stride between requests (in elements)
    # Mamba state metadata (per-layer, per-state-type)
    # These are 1D arrays indexed by (layer_idx * num_state_types + state_type_idx)
    state_base_addrs_ptr,  # base address of each state tensor
    state_block_strides_ptr,  # bytes per block for each state
    state_elem_sizes_ptr,  # element size for each state
    state_inner_sizes_ptr,  # number of elements in inner dimensions
    state_conv_widths_ptr,  # conv width for conv states (0 for temporal)
    state_group_indices_ptr,  # maps state_idx to group index in block table
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    # Output: num_accepted_tokens update (for src==dst case)
    num_accepted_tokens_out_ptr,
    idx_mapping_ptr,
    # Runtime parameter (varies per batch - NOT constexpr to avoid recompilation)
    num_reqs,
    # Compile-time constants (fixed after model initialization)
    # block_size: determined by model config, constant for all invocations
    block_size: tl.constexpr,
    # COPY_BLOCK_SIZE: fixed tuning parameter for memory copy loop
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
    HAS_IDX_MAPPING: tl.constexpr = False,
    PRECOMPUTED_NEW_COMPUTED: tl.constexpr = False,
):
    """Postprocess Mamba state for both v1 and v2 runner layouts."""
    batch_idx = tl.program_id(0)
    state_idx = tl.program_id(1)
    if batch_idx >= num_reqs:
        return

    if HAS_IDX_MAPPING:
        req_idx = tl.load(idx_mapping_ptr + batch_idx)
        if req_idx < 0:
            return
    else:
        req_idx = batch_idx

    num_accepted = tl.load(num_accepted_tokens_ptr + req_idx)
    src_block_idx = tl.load(mamba_state_idx_ptr + req_idx)
    if PRECOMPUTED_NEW_COMPUTED:
        new_num_computed = tl.load(num_computed_tokens_ptr + req_idx)
        num_tokens_running_state = new_num_computed - num_accepted + 1
    else:
        num_scheduled = tl.load(num_scheduled_tokens_ptr + req_idx)
        num_computed = tl.load(num_computed_tokens_ptr + req_idx)
        num_draft = tl.load(num_draft_tokens_ptr + req_idx)
        num_tokens_running_state = num_computed + num_scheduled - num_draft
        new_num_computed = num_tokens_running_state + num_accepted - 1

    aligned_new_computed = (new_num_computed // block_size) * block_size
    if aligned_new_computed < num_tokens_running_state:
        return

    token_bias = aligned_new_computed - num_tokens_running_state
    dest_block_idx = aligned_new_computed // block_size - 1
    if src_block_idx == dest_block_idx and state_idx == 0:
        if HAS_IDX_MAPPING:
            tl.store(num_accepted_tokens_ptr + req_idx, 1)
        else:
            tl.store(num_accepted_tokens_out_ptr + req_idx, 1)

    if src_block_idx == dest_block_idx and token_bias == 0:
        return

    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx
    _copy_mamba_state_block(
        state_idx,
        bt_row_idx,
        src_block_idx,
        dest_block_idx,
        token_bias,
        block_table_ptrs_ptr,
        block_table_stride_req,
        state_base_addrs_ptr,
        state_block_strides_ptr,
        state_elem_sizes_ptr,
        state_inner_sizes_ptr,
        state_conv_widths_ptr,
        state_group_indices_ptr,
        state_dim_row_count_ptr,
        state_dim_row_stride_ptr,
        COPY_BLOCK_SIZE,
        CONV_STATE_DIM_FIRST,
    )


@triton.jit
def precopy_mamba_align_fused_kernel(
    mamba_state_idx_ptr,
    src_col_ptr,
    token_bias_ptr,
    block_table_ptrs_ptr,
    block_table_stride_req: tl.int64,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    idx_mapping_ptr,
    num_reqs,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
):
    """Pre-copy an align-mode state when a request crosses a block."""
    batch_idx = tl.program_id(0)
    state_idx = tl.program_id(1)
    if batch_idx >= num_reqs:
        return
    req_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_idx < 0:
        return

    src_col = tl.load(src_col_ptr + req_idx)
    dst_col = tl.load(mamba_state_idx_ptr + req_idx)
    if src_col < 0 or src_col == dst_col:
        return

    token_bias = tl.load(token_bias_ptr + req_idx)
    _copy_mamba_state_block(
        state_idx,
        batch_idx,
        src_col,
        dst_col,
        token_bias,
        block_table_ptrs_ptr,
        block_table_stride_req,
        state_base_addrs_ptr,
        state_block_strides_ptr,
        state_elem_sizes_ptr,
        state_inner_sizes_ptr,
        state_conv_widths_ptr,
        state_group_indices_ptr,
        state_dim_row_count_ptr,
        state_dim_row_stride_ptr,
        COPY_BLOCK_SIZE,
        CONV_STATE_DIM_FIRST,
    )
