# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/mamba_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


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
    # DS conv row metadata. Zero keeps the single-region copy path.
    state_dim_row_count_ptr,  # int32: per-block dim row count for DS conv
    state_dim_row_stride_ptr,  # int64: bytes between rows for DS conv
    # Output: num_accepted_tokens update (for src==dst case)
    num_accepted_tokens_out_ptr,
    # Optional: batch_idx -> req_idx mapping (V2 model runner / PP). The
    # per-request decision arrays are in req-state-slot order; the block table
    # is in batch order, so HAS_IDX_MAPPING splits the two indexings.
    idx_mapping_ptr,
    # Runtime parameter (varies per batch - NOT constexpr to avoid recompilation)
    num_reqs,
    # Compile-time constants (fixed after model initialization)
    # block_size: determined by model config, constant for all invocations
    block_size: tl.constexpr,
    # COPY_BLOCK_SIZE: fixed tuning parameter for memory copy loop
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
    # HAS_IDX_MAPPING: when True, program_id(0) is a batch index resolved to a
    # req-state slot via idx_mapping_ptr (V2). When False, it is the req index.
    HAS_IDX_MAPPING: tl.constexpr = False,
    # PRECOMPUTED_NEW_COMPUTED: when True, num_computed_tokens_ptr already holds
    # the post-step new_num_computed value (V2 supplies the advanced count).
    PRECOMPUTED_NEW_COMPUTED: tl.constexpr = False,
):
    """
    Fused GPU kernel for postprocess_mamba that computes decisions AND performs
    mamba state copies without any CPU-GPU synchronization.

    Grid: (num_reqs, num_layers * num_state_types)
    - program_id(0) = request index
    - program_id(1) = state_idx (flattened index into layer/state_type metadata)

    Note: num_layers and num_state_types are not passed as kernel parameters
    because the kernel indexes directly into pre-flattened metadata arrays
    using program_id(1). The grid dimensions encode the total state count.
    """
    batch_idx = tl.program_id(0)
    state_idx = tl.program_id(1)

    # Bounds check: num_reqs is the number of active batch rows. With
    # HAS_IDX_MAPPING, req_idx is a (possibly sparse) request-state slot, so it
    # must NOT be checked against num_reqs.
    if batch_idx >= num_reqs:
        return

    if HAS_IDX_MAPPING:
        req_idx = tl.load(idx_mapping_ptr + batch_idx)
        # -1 is the skip sentinel for inactive batch rows.
        if req_idx < 0:
            return
    else:
        req_idx = batch_idx

    # Compute decision logic (mirrors postprocess_mamba Python reference)
    num_accepted = tl.load(num_accepted_tokens_ptr + req_idx)
    src_block_idx = tl.load(mamba_state_idx_ptr + req_idx)

    if PRECOMPUTED_NEW_COMPUTED:
        # num_computed_tokens_ptr already holds the post-step new_num_computed
        # value (V2 supplies the advanced count). num_scheduled/num_draft are
        # unused on this path and are passed as None, so they must not be
        # loaded here.
        new_num_computed = tl.load(num_computed_tokens_ptr + req_idx)
        num_tokens_running_state = new_num_computed - num_accepted + 1
    else:
        num_scheduled = tl.load(num_scheduled_tokens_ptr + req_idx)
        num_computed = tl.load(num_computed_tokens_ptr + req_idx)
        num_draft = tl.load(num_draft_tokens_ptr + req_idx)
        num_tokens_running_state = num_computed + num_scheduled - num_draft
        new_num_computed = num_tokens_running_state + num_accepted - 1

    aligned_new_computed = (new_num_computed // block_size) * block_size

    needs_copy = aligned_new_computed >= num_tokens_running_state

    if not needs_copy:
        return

    # Compute copy parameters
    accept_token_bias = aligned_new_computed - num_tokens_running_state
    dest_block_idx = aligned_new_computed // block_size - 1

    # Load state metadata for this layer/state_type
    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)

    # Load the group index for this state, then index into the correct
    # group's block table. Each mamba group has independently allocated
    # physical blocks.
    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)

    # block_table_ptrs_ptr holds one pointer per group (each group owns its own
    # block table). Reinterpret as int32* since block ids are int32.
    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table_typed = group_base_addr.to(tl.pointer_type(tl.int32))

    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx
    block_table_base = block_table_typed + bt_row_idx * block_table_stride_req

    # Widen block ids to int64 before they reach `block_id * state_block_stride`
    # below: state_block_stride can exceed 2**31 bytes for large mamba caches,
    # and Triton would otherwise do the multiply in int32 and wrap.
    src_block_id = tl.load(block_table_base + src_block_idx).to(tl.int64)
    dest_block_id = tl.load(block_table_base + dest_block_idx).to(tl.int64)

    # Compute source and destination addresses based on state type
    # conv_width > 0 means this is a conv state (get_conv_copy_spec logic)
    # conv_width == 0 means this is a temporal state (get_temporal_copy_spec logic)
    is_conv_state = conv_width > 0

    if is_conv_state:
        # Conv state: copy
        #   state[block_table[req_idx, src_block_idx],  accept_token_bias:]
        # to
        #   state[block_table[req_idx, dest_block_idx], :conv_width - accept_token_bias]
        src_offset = accept_token_bias.to(tl.int64) * state_inner_size * state_elem_size
        src_addr = state_base_addr + src_block_id * state_block_stride + src_offset
        dst_addr = state_base_addr + dest_block_id * state_block_stride
        # Number of elements to copy:
        # (conv_width - accept_token_bias) * inner_size
        if CONV_STATE_DIM_FIRST:
            # DS conv layout: state[block, dim, state_len]. state_len is the
            # slide axis, so copy per dim row (dim_row_count rows of
            # (conv_width - accept_token_bias) * elem_size bytes each, advancing
            # by dim_row_stride per row).
            copy_size = (conv_width - accept_token_bias).to(tl.int64) * state_elem_size
        else:
            num_elems_to_copy = (conv_width - accept_token_bias).to(tl.int64) * state_inner_size
            copy_size = num_elems_to_copy * state_elem_size
    else:
        # Temporal state: copy
        #   state[block_table[req_idx, src_block_idx + accept_token_bias]]
        # to
        #   state[block_table[req_idx, dest_block_idx]]
        actual_src_block_idx = src_block_idx + accept_token_bias
        actual_src_block_id = tl.load(block_table_base + actual_src_block_idx).to(tl.int64)
        src_addr = state_base_addr + actual_src_block_id * state_block_stride
        dst_addr = state_base_addr + dest_block_id * state_block_stride
        # Use natural block data size (inner_size * elem_size), NOT
        # state_block_stride which is the page stride and can exceed the
        # actual data when the state tensor uses as_strided page padding.
        copy_size = state_inner_size * state_elem_size

    # Mirror postprocess_mamba's trailing
    #     if src_block_idx == dest_block_idx: num_accepted_tokens_cpu[i] = 1
    # This runs whether or not the copy below is skipped (it's per-request, so
    # only state_idx == 0 writes). The write target depends on the caller:
    # main (after vLLM #50432) passes a non-null output buffer and reads from a
    # snapshot; v0.26.0 passes None for the output and updates the input buffer
    # in place under HAS_IDX_MAPPING. Distinguish by whether an output buffer
    # was provided, since both lanes use HAS_IDX_MAPPING=True on the V2 path.
    if src_block_idx == dest_block_idx and state_idx == 0:
        if num_accepted_tokens_out_ptr is None:
            tl.store(num_accepted_tokens_ptr + req_idx, 1)
        else:
            tl.store(num_accepted_tokens_out_ptr + req_idx, 1)

    # Mirror collect_mamba_copy_meta's early return: src==dst with no token
    # bias means source and destination ranges coincide, so the copy is a
    # no-op.
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    # Hoist the pointer-type cast out of the copy loop. triton-ascend's
    # PtrOffsetInfo::AxisInfo analysis aborts on `(addr + i + offsets).to(...)`
    # inside the loop (SmallVector assertion `idx < size()`); casting once
    # here and doing plain pointer arithmetic inside the loop is the same fix
    # vllm-ascend applies to batch_memcpy_kernel.
    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    if CONV_STATE_DIM_FIRST and is_conv_state:
        # DS conv layout: state[block, dim, state_len]. state_len is the slide
        # axis, so copy per dim row: dim_row_count rows, each of
        # (conv_width - accept_token_bias) * elem_size bytes, advancing both
        # src and dst by dim_row_stride per row. Load the row metadata here
        # (not earlier) so it stays in the same scope as the copy loop.
        dim_row_count = tl.load(state_dim_row_count_ptr + state_idx)
        dim_row_stride = tl.load(state_dim_row_stride_ptr + state_idx)
        for row in range(dim_row_count):
            row_src = (src_addr + row * dim_row_stride).to(tl.pointer_type(tl.uint8))
            row_dst = (dst_addr + row * dim_row_stride).to(tl.pointer_type(tl.uint8))
            for i in range(0, copy_size, COPY_BLOCK_SIZE):
                mask = (i + offsets) < copy_size
                data = tl.load(row_src + i + offsets, mask=mask)
                tl.store(row_dst + i + offsets, data, mask=mask)
    else:
        # SD conv / temporal: single contiguous region.
        src_ptr = src_addr.to(tl.pointer_type(tl.uint8))
        dst_ptr = dst_addr.to(tl.pointer_type(tl.uint8))
        for i in range(0, copy_size, COPY_BLOCK_SIZE):
            mask = (i + offsets) < copy_size
            data = tl.load(src_ptr + i + offsets, mask=mask)
            tl.store(dst_ptr + i + offsets, data, mask=mask)
