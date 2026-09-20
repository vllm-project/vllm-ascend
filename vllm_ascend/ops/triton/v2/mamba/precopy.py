# Adapted from vllm/v1/worker/mamba_utils.py.
# SPDX-License-Identifier: Apache-2.0

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
    tile_idx,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
):
    """Copy one Mamba state block without casting pointers in copy loops.

    triton-ascend's AxisInfo analysis can abort on
    ``(integer_address + loop_offset).to(pointer_type)``. Cast the base
    addresses once and use pointer arithmetic inside the loops instead.
    """
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
        if tile_idx > 0:
            return
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        dim_rows = tl.load(state_dim_row_count_ptr + state_idx)
        row_stride = tl.load(state_dim_row_stride_ptr + state_idx)
        per_row_bytes = (conv_width - token_bias).to(tl.int64) * state_elem_size
        bias_bytes = token_bias.to(tl.int64) * state_elem_size
        src_block_addr = state_base_addr + src_block_id * state_block_stride
        offsets = tl.arange(0, COPY_BLOCK_SIZE)

        for row in range(0, dim_rows):
            row_src = (src_block_addr + row * row_stride + bias_bytes).to(tl.pointer_type(tl.uint8))
            row_dst = (dst_addr + row * row_stride).to(tl.pointer_type(tl.uint8))
            for offset in range(0, per_row_bytes, COPY_BLOCK_SIZE):
                mask = offset + offsets < per_row_bytes
                data = tl.load(row_src + offset + offsets, mask=mask)
                tl.store(row_dst + offset + offsets, data, mask=mask)
        return

    if is_conv_state:
        if tile_idx > 0:
            return
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        src_offset = token_bias.to(tl.int64) * state_inner_size * state_elem_size
        src_addr = state_base_addr + src_block_id * state_block_stride + src_offset
        copy_size = (conv_width - token_bias).to(tl.int64) * state_inner_size * state_elem_size
        offsets = tl.arange(0, COPY_BLOCK_SIZE)
        src_ptr = src_addr.to(tl.pointer_type(tl.uint8))
        dst_ptr = dst_addr.to(tl.pointer_type(tl.uint8))
        for offset in range(0, copy_size, COPY_BLOCK_SIZE):
            mask = offset + offsets < copy_size
            data = tl.load(src_ptr + offset + offsets, mask=mask)
            tl.store(dst_ptr + offset + offsets, data, mask=mask)
        return

    actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)
    src_addr = state_base_addr + actual_src_block_id * state_block_stride
    copy_size = state_inner_size * state_elem_size
    copy_size_u64 = copy_size // 8
    src_u64 = src_addr.to(tl.pointer_type(tl.uint64))
    dst_u64 = dst_addr.to(tl.pointer_type(tl.uint64))

    work_per_tile = tl.cdiv(copy_size_u64, TEMPORAL_TILES)
    tile_start = tile_idx.to(tl.int64) * work_per_tile
    tile_end = tl.minimum(tile_start + work_per_tile, copy_size_u64)
    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    for offset in range(tile_start, tile_end, COPY_BLOCK_SIZE):
        mask = offset + offsets < tile_end
        data = tl.load(src_u64 + offset + offsets, mask=mask)
        tl.store(dst_u64 + offset + offsets, data, mask=mask)

    if tile_idx == 0:
        tail_start = copy_size_u64 * 8
        tail_bytes = copy_size - tail_start
        tail_offsets = tl.arange(0, 8)
        tail_src = (src_addr + tail_start).to(tl.pointer_type(tl.uint8))
        tail_dst = (dst_addr + tail_start).to(tl.pointer_type(tl.uint8))
        tail_mask = tail_offsets < tail_bytes
        tail_data = tl.load(tail_src + tail_offsets, mask=tail_mask)
        tl.store(tail_dst + tail_offsets, tail_data, mask=tail_mask)


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
    HAS_IDX_MAPPING: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr = 1,
):
    batch_idx = tl.program_id(0)
    state_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)

    if batch_idx >= num_reqs:
        return

    if HAS_IDX_MAPPING:
        req_idx = tl.load(idx_mapping_ptr + batch_idx)

        if req_idx < 0:
            return
    else:
        req_idx = batch_idx

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
        tile_idx,
        COPY_BLOCK_SIZE,
        CONV_STATE_DIM_FIRST,
        TEMPORAL_TILES,
    )
