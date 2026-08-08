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
    # DS conv-state row metadata. Only read when CONV_STATE_DIM_FIRST is set,
    # which this kernel does not support (see the static assert below).
    state_dim_row_count_ptr,  # int32: per-block dim row count for DS conv
    state_dim_row_stride_ptr,  # int64: bytes between rows for DS conv
    # Output: num_accepted_tokens update (for src==dst case)
    num_accepted_tokens_out_ptr,
    # batch_idx -> req-state slot mapping, only read when HAS_IDX_MAPPING is set
    idx_mapping_ptr,
    # Runtime parameter (varies per batch - NOT constexpr to avoid recompilation)
    num_reqs,
    # Compile-time constants (fixed after model initialization)
    # block_size: determined by model config, constant for all invocations
    block_size: tl.constexpr,
    # COPY_BLOCK_SIZE: fixed tuning parameter for memory copy loop
    COPY_BLOCK_SIZE: tl.constexpr,
    # CONV_STATE_DIM_FIRST: conv state stored as (dim, state_len) per block.
    # The copy loop below assumes the (state_len, dim) layout, so the DS layout
    # is rejected instead of being copied with the wrong strides.
    CONV_STATE_DIM_FIRST: tl.constexpr = False,
    # HAS_IDX_MAPPING: when True, program_id(0) is a batch index resolved to a
    # req-state slot via idx_mapping_ptr (V2). When False, it is the req index.
    HAS_IDX_MAPPING: tl.constexpr = False,
    # PRECOMPUTED_NEW_COMPUTED: when True, num_computed_tokens_ptr already holds
    # the post-step computed count.
    PRECOMPUTED_NEW_COMPUTED: tl.constexpr = False,
):
    """
    Fused GPU kernel for postprocess_mamba that computes decisions AND performs
    mamba state copies without any CPU-GPU synchronization.

    Grid: (num_reqs, num_layers * num_state_types)
    - program_id(0) = request index (or batch index when HAS_IDX_MAPPING)
    - program_id(1) = state_idx (flattened index into layer/state_type metadata)

    Note: num_layers and num_state_types are not passed as kernel parameters
    because the kernel indexes directly into pre-flattened metadata arrays
    using program_id(1). The grid dimensions encode the total state count.

    The signature mirrors ``vllm.v1.worker.mamba_utils`` because this kernel is
    swapped in for the upstream one by ``patch_mamba_utils`` while the launcher
    stays upstream; the parameter list must therefore stay in sync with it.
    """
    # The copy loop below walks conv states as (state_len, dim); the DS layout
    # would need a different stride walk, so reject it at compile time rather
    # than silently copying the wrong bytes.
    tl.static_assert(
        not CONV_STATE_DIM_FIRST,
        "Ascend postprocess_mamba_fused_kernel only supports the SD conv state "
        "layout; unset VLLM_SSM_CONV_STATE_LAYOUT=DS.",
    )

    batch_idx = tl.program_id(0)
    state_idx = tl.program_id(1)

    # Bounds check
    if batch_idx >= num_reqs:
        return

    if HAS_IDX_MAPPING:
        req_idx = tl.load(idx_mapping_ptr + batch_idx)
        if req_idx < 0:
            return
    else:
        req_idx = batch_idx

    # Compute decision logic (mirrors postprocess_mamba Python reference)
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
    block_table_base = block_table_typed + req_idx * block_table_stride_req

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
    # only state_idx == 0 writes).
    if src_block_idx == dest_block_idx and state_idx == 0:
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
    src_ptr = src_addr.to(tl.pointer_type(tl.uint8))
    dst_ptr = dst_addr.to(tl.pointer_type(tl.uint8))
    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    for i in range(0, copy_size, COPY_BLOCK_SIZE):
        mask = (i + offsets) < copy_size
        data = tl.load(src_ptr + i + offsets, mask=mask)
        tl.store(dst_ptr + i + offsets, data, mask=mask)
