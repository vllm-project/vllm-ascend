# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/mamba_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Only the shared copy device function ``_copy_mamba_state_block`` is provided
# here. It is a faithful copy of the upstream helper
# (vllm/v1/worker/mamba_utils.py) with a single triton-ascend workaround: the
# ``.to(tl.pointer_type(tl.uint8))`` cast is hoisted out of every copy loop.
# triton-ascend's PtrOffsetInfo::AxisInfo analysis otherwise aborts with a
# SmallVector ``idx < size()`` assertion on ``(addr + i + offsets).to(...)``
# evaluated inside a loop.
#
# The upstream ``postprocess_mamba_fused_kernel`` and
# ``precopy_mamba_align_fused_kernel`` (V1 and V2 align paths) both call this
# helper by name. Triton resolves the callee from the caller's ``__globals__``
# snapshot taken at compile time (see triton's
# ``ast_to_ttir``/``get_capture_scope``), so patching the
# ``mamba_utils._copy_mamba_state_block`` symbol once (see patch_mamba_utils.py)
# makes both kernels inline this ascend-safe copy body. No kernel replacement
# is needed, which avoids signature drift with the upstream callers.

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
    # DS conv row metadata. Zero keeps the single-region copy path.
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
):
    """Copy one (layer, state-type) mamba state block between block columns.

    Shared copy body of ``postprocess_mamba_fused_kernel`` and
    ``precopy_mamba_align_fused_kernel``, mirroring the V1 copy specs
    (``get_conv_copy_spec`` / ``get_temporal_copy_spec``):
    - conv state (conv_width > 0): shift the window by ``token_bias`` tokens,
      ``state[bt[src_col], token_bias:] ->
      state[bt[dst_col], :conv_width - token_bias]``
    - temporal state: ``token_bias`` selects the accepted speculative column,
      ``state[bt[src_col + token_bias]] -> state[bt[dst_col]]``

    The caller owns the decision logic (which columns, whether to copy); this
    device function only performs the byte copy for the given metadata slot.
    """
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
    block_table_base = block_table_typed + bt_row_idx * block_table_stride_req

    # Widen block ids to int64 before they reach `block_id * state_block_stride`
    # below: state_block_stride can exceed 2**31 bytes for large mamba caches,
    # and Triton would otherwise do the multiply in int32 and wrap.
    dest_block_id = tl.load(block_table_base + dst_col).to(tl.int64)
    dst_addr = state_base_addr + dest_block_id * state_block_stride

    is_conv_state = conv_width > 0

    if CONV_STATE_DIM_FIRST and is_conv_state:
        # DS conv layout: state_len is the slide axis; copy per dim row.
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        dim_rows = tl.load(state_dim_row_count_ptr + state_idx)
        row_stride = tl.load(state_dim_row_stride_ptr + state_idx)
        per_row_bytes = (conv_width - token_bias).to(tl.int64) * state_elem_size
        bias_bytes = token_bias.to(tl.int64) * state_elem_size
        src_block_addr = state_base_addr + src_block_id * state_block_stride
        offsets = tl.arange(0, COPY_BLOCK_SIZE)
        for d in range(0, dim_rows):
            row_src = src_block_addr + d * row_stride + bias_bytes
            row_dst = dst_addr + d * row_stride
            # Hoist the pointer-type cast out of the copy loop. triton-ascend's
            # PtrOffsetInfo::AxisInfo analysis aborts on
            # `(addr + i + offsets).to(...)` inside the loop (SmallVector
            # assertion `idx < size()`); casting once per row and doing plain
            # pointer arithmetic inside the loop is the same fix vllm-ascend
            # applies to batch_memcpy_kernel.
            row_src_ptr = row_src.to(tl.pointer_type(tl.uint8))
            row_dst_ptr = row_dst.to(tl.pointer_type(tl.uint8))
            for i in range(0, per_row_bytes, COPY_BLOCK_SIZE):
                mask = (i + offsets) < per_row_bytes
                data = tl.load(row_src_ptr + i + offsets, mask=mask)
                tl.store(row_dst_ptr + i + offsets, data, mask=mask)
        return

    if is_conv_state:
        # SD conv: copy
        #   state[bt[src_col], token_bias:] ->
        #   state[bt[dst_col], :conv_width - token_bias]
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        src_offset = token_bias.to(tl.int64) * state_inner_size * state_elem_size
        src_addr = state_base_addr + src_block_id * state_block_stride + src_offset
        num_elems_to_copy = (conv_width - token_bias).to(tl.int64) * state_inner_size
        copy_size = num_elems_to_copy * state_elem_size
    else:
        # Temporal state: copy state[bt[src_col + token_bias]] -> state[bt[dst_col]]
        actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(
            tl.int64
        )
        src_addr = state_base_addr + actual_src_block_id * state_block_stride
        # Use natural block data size (inner_size * elem_size), NOT
        # state_block_stride which is the page stride and can exceed the
        # actual data when the state tensor uses as_strided page padding.
        copy_size = state_inner_size * state_elem_size

    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    # Hoist the pointer-type cast out of the copy loop (see note above).
    src_ptr = src_addr.to(tl.pointer_type(tl.uint8))
    dst_ptr = dst_addr.to(tl.pointer_type(tl.uint8))
    for i in range(0, copy_size, COPY_BLOCK_SIZE):
        mask = (i + offsets) < copy_size
        data = tl.load(src_ptr + i + offsets, mask=mask)
        tl.store(dst_ptr + i + offsets, data, mask=mask)
