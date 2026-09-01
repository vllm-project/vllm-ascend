# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/mamba_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs"])
def precopy_mamba_align_fused_kernel(
    # Per-request-slot inputs (indexed by req_idx via idx_mapping), produced by
    # the V2 fused align preprocess kernel for the current step:
    mamba_state_idx_ptr,  # post-advance dst block column
    src_col_ptr,  # pre-advance src block column (-1 = fresh)
    token_bias_ptr,  # accepted-token bias = num_accepted - 1 (pre-reset)
    # Same flattened state-layout metadata as postprocess_mamba_fused_kernel
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
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_reqs,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
    HAS_IDX_MAPPING: tl.constexpr = True,
):
    """NPU-safe variant of upstream ``precopy_mamba_align_fused_kernel``.

    Same grid, signature and copy semantics as upstream (V2 mamba align
    pre-copy across block boundaries, driven by GPU-resident src columns),
    with two Ascend-specific fixes taken from vllm-ascend's
    ``postprocess.py``:

    1. Byte-wise uint8 copies instead of the upstream uint64-vectorized
       temporal copy: 8-byte vector loads/stores are not supported by the
       Ascend vector core and raise a "vector core exception".
    2. The pointer-type cast is hoisted out of the copy loop: triton-ascend's
       ``PtrOffsetInfo::AxisInfo`` analysis aborts on
       ``(addr + i + offsets).to(...)`` inside the loop.

    Grid: (num_reqs, num_layers * num_state_types).
    """
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

    src_col = tl.load(src_col_ptr + req_idx)
    dst_col = tl.load(mamba_state_idx_ptr + req_idx)
    # Fresh state, or still writing the same block: kernels locate the initial
    # state in-block via num_accepted (preserved when no boundary is crossed),
    # so there is nothing to copy.
    if src_col < 0 or src_col == dst_col:
        return

    token_bias = tl.load(token_bias_ptr + req_idx)

    # Load state metadata for this layer/state_type
    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)

    # Load the group index for this state, then index into the correct
    # group's block table. Each mamba group has independently allocated
    # physical blocks. Reinterpret as int32* since block ids are int32.
    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)
    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table_typed = group_base_addr.to(tl.pointer_type(tl.int32))
    block_table_base = block_table_typed + batch_idx * block_table_stride_req

    # Widen block ids to int64 before `block_id * state_block_stride`:
    # state_block_stride can exceed 2**31 bytes for large mamba caches, and
    # Triton would otherwise do the multiply in int32 and wrap.
    dst_block_id = tl.load(block_table_base + dst_col).to(tl.int64)
    dst_addr = state_base_addr + dst_block_id * state_block_stride

    is_conv_state = conv_width > 0
    offsets = tl.arange(0, COPY_BLOCK_SIZE)

    if CONV_STATE_DIM_FIRST and is_conv_state:
        # DS conv layout: state[block, dim, state_len]. state_len is the slide
        # axis, so copy per dim row: dim_row_count rows, each of
        # (conv_width - token_bias) * elem_size bytes, skipping token_bias
        # elements at the row start and advancing by dim_row_stride per row.
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        dim_rows = tl.load(state_dim_row_count_ptr + state_idx)
        row_stride = tl.load(state_dim_row_stride_ptr + state_idx)
        per_row_bytes = (conv_width - token_bias).to(tl.int64) * state_elem_size
        bias_bytes = token_bias.to(tl.int64) * state_elem_size
        src_block_addr = state_base_addr + src_block_id * state_block_stride
        for d in range(0, dim_rows):
            row_src = (src_block_addr + d * row_stride + bias_bytes).to(tl.pointer_type(tl.uint8))
            row_dst = (dst_addr + d * row_stride).to(tl.pointer_type(tl.uint8))
            for i in range(0, per_row_bytes, COPY_BLOCK_SIZE):
                mask = (i + offsets) < per_row_bytes
                data = tl.load(row_src + i + offsets, mask=mask)
                tl.store(row_dst + i + offsets, data, mask=mask)
    elif is_conv_state:
        # SD conv: copy
        #   state[bt[src_col], token_bias:] ->
        #   state[bt[dst_col], :conv_width - token_bias]
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        src_offset = token_bias.to(tl.int64) * state_inner_size * state_elem_size
        src_addr = state_base_addr + src_block_id * state_block_stride + src_offset
        num_elems_to_copy = (conv_width - token_bias).to(tl.int64) * state_inner_size
        copy_size = num_elems_to_copy * state_elem_size
        src_ptr = src_addr.to(tl.pointer_type(tl.uint8))
        dst_ptr = dst_addr.to(tl.pointer_type(tl.uint8))
        for i in range(0, copy_size, COPY_BLOCK_SIZE):
            mask = (i + offsets) < copy_size
            data = tl.load(src_ptr + i + offsets, mask=mask)
            tl.store(dst_ptr + i + offsets, data, mask=mask)
    else:
        # Temporal state: copy state[bt[src_col + token_bias]] ->
        # state[bt[dst_col]]. Byte-wise uint8 copy: the upstream uint64
        # vectorization is not supported by the Ascend vector core.
        actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)
        src_addr = state_base_addr + actual_src_block_id * state_block_stride
        # Use natural block data size (inner_size * elem_size), NOT
        # state_block_stride which is the page stride and can exceed the
        # actual data when the state tensor uses as_strided page padding.
        copy_size = state_inner_size * state_elem_size
        src_ptr = src_addr.to(tl.pointer_type(tl.uint8))
        dst_ptr = dst_addr.to(tl.pointer_type(tl.uint8))
        for i in range(0, copy_size, COPY_BLOCK_SIZE):
            mask = (i + offsets) < copy_size
            data = tl.load(src_ptr + i + offsets, mask=mask)
            tl.store(dst_ptr + i + offsets, data, mask=mask)
