# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/mamba_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def _memcpy_u64_tiled(
    src_addr,
    dst_addr,
    copy_size,
    tile_idx,
    COPY_BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    """Head/body/tail memcpy with the u64 body split across ``NUM_TILES`` CTAs.

    Fast path (``src`` and ``dst`` share sub-8B alignment): tile 0 owns the
    byte head that lifts dst to 8B and the 0-7 byte tail; body tiles vectorize
    as u64 over the aligned interior. ``NUM_TILES=1`` collapses to a single-
    CTA memcpy.

    Slow path (mismatched sub-8B alignment): byte-wide tiled copy.
    """
    src_addr_i = src_addr.to(tl.int64)
    dst_addr_i = dst_addr.to(tl.int64)

    if ((src_addr_i ^ dst_addr_i) & 7) == 0:
        head_bytes = tl.minimum(((-dst_addr_i) & 7).to(tl.int64), copy_size)
        if tile_idx == 0:
            head_off = tl.arange(0, 8)
            head_mask = head_off < head_bytes
            head_src = src_addr.to(tl.pointer_type(tl.uint8))
            head_dst = dst_addr.to(tl.pointer_type(tl.uint8))
            tl.store(
                head_dst + head_off,
                tl.load(head_src + head_off, mask=head_mask),
                mask=head_mask,
            )

        body_bytes = copy_size - head_bytes
        body_u64 = body_bytes // 8
        per_tile_u64_raw = tl.cdiv(body_u64, NUM_TILES)
        per_tile_u64 = tl.cdiv(per_tile_u64_raw, COPY_BLOCK_SIZE) * COPY_BLOCK_SIZE
        tile_start = tile_idx.to(tl.int64) * per_tile_u64
        tile_end = tl.minimum(tile_start + per_tile_u64, body_u64)

        src_body_u64 = (src_addr + head_bytes).to(tl.pointer_type(tl.uint64))
        dst_body_u64 = (dst_addr + head_bytes).to(tl.pointer_type(tl.uint64))
        offsets = tl.arange(0, COPY_BLOCK_SIZE)
        for i in range(tile_start, tile_end, COPY_BLOCK_SIZE):
            mask = (i + offsets) < tile_end
            data = tl.load(src_body_u64 + i + offsets, mask=mask)
            tl.store(dst_body_u64 + i + offsets, data, mask=mask)

        if tile_idx == 0:
            tail_start = head_bytes + body_u64 * 8
            tail_bytes = copy_size - tail_start
            tail_off = tl.arange(0, 8)
            tail_src = (src_addr + tail_start).to(tl.pointer_type(tl.uint8))
            tail_dst = (dst_addr + tail_start).to(tl.pointer_type(tl.uint8))
            tail_mask = tail_off < tail_bytes
            tl.store(
                tail_dst + tail_off,
                tl.load(tail_src + tail_off, mask=tail_mask),
                mask=tail_mask,
            )
    else:
        src_u8 = src_addr.to(tl.pointer_type(tl.uint8))
        dst_u8 = dst_addr.to(tl.pointer_type(tl.uint8))
        per_tile_bytes_raw = tl.cdiv(copy_size, NUM_TILES)
        per_tile_bytes = tl.cdiv(per_tile_bytes_raw, COPY_BLOCK_SIZE) * COPY_BLOCK_SIZE
        tile_start = tile_idx.to(tl.int64) * per_tile_bytes
        tile_end = tl.minimum(tile_start + per_tile_bytes, copy_size)
        offsets = tl.arange(0, COPY_BLOCK_SIZE)
        for i in range(tile_start, tile_end, COPY_BLOCK_SIZE):
            mask = (i + offsets) < tile_end
            data = tl.load(src_u8 + i + offsets, mask=mask)
            tl.store(dst_u8 + i + offsets, data, mask=mask)


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
    # TEMPORAL_TILES: when > 1, the temporal copy body is partitioned across
    # TEMPORAL_TILES CTAs along the u64 inner range. Callers must launch a
    # 3D grid (num_reqs, total_states, TEMPORAL_TILES). Default 1 preserves
    # the existing 2D-grid contract.
    TEMPORAL_TILES: tl.constexpr = 1,
):
    """
    Fused GPU kernel for postprocess_mamba that computes decisions AND performs
    mamba state copies without any CPU-GPU synchronization.

    Grid: (num_reqs, num_layers * num_state_types [, TEMPORAL_TILES])
    - program_id(0) = request/batch index
    - program_id(1) = state_idx (flattened index into layer/state_type metadata)
    - program_id(2) = temporal-copy tile index (0 when TEMPORAL_TILES == 1)

    Note: num_layers and num_state_types are not passed as kernel parameters
    because the kernel indexes directly into pre-flattened metadata arrays
    using program_id(1). The grid dimensions encode the total state count.
    """
    batch_idx = tl.program_id(0)
    state_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)

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

    # Mirror postprocess_mamba's trailing
    #     if src_block_idx == dest_block_idx: num_accepted_tokens_cpu[i] = 1
    # This runs whether or not the copy below is skipped (it's per-request, so
    # only state_idx == 0 writes). Guard on tile_idx == 0 so tiles > 0
    # (when TEMPORAL_TILES > 1) do not duplicate the store. The write target
    # depends on the caller: main (after vLLM #50432) passes a non-null output
    # buffer and reads from a snapshot; v0.26.0 passes None for the output and
    # updates the input buffer in place under HAS_IDX_MAPPING.
    if src_block_idx == dest_block_idx and state_idx == 0 and tile_idx == 0:
        if num_accepted_tokens_out_ptr is None:
            tl.store(num_accepted_tokens_ptr + req_idx, 1)
        else:
            tl.store(num_accepted_tokens_out_ptr + req_idx, 1)

    # Mirror collect_mamba_copy_meta's early return: src==dst with no token
    # bias means source and destination ranges coincide, so the copy is a
    # no-op.
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

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

    if CONV_STATE_DIM_FIRST and is_conv_state:
        # Conv states are small; only tile 0 does the copy. Higher tiles
        # early-return so they contribute nothing beyond a bounds check.
        if tile_idx > 0:
            return
        # DS conv layout: state[block, dim, state_len]. state_len is the slide
        # axis, so copy per dim row: dim_row_count rows, each of
        # (conv_width - accept_token_bias) * elem_size bytes, advancing both
        # src and dst by dim_row_stride per row.
        dim_row_count = tl.load(state_dim_row_count_ptr + state_idx)
        dim_row_stride = tl.load(state_dim_row_stride_ptr + state_idx)
        per_row_bytes = (conv_width - accept_token_bias).to(tl.int64) * state_elem_size
        bias_bytes = accept_token_bias.to(tl.int64) * state_elem_size
        src_block_addr = state_base_addr + src_block_id * state_block_stride
        offsets = tl.arange(0, COPY_BLOCK_SIZE)
        for row in range(dim_row_count):
            row_src = src_block_addr + row * dim_row_stride + bias_bytes
            row_dst = state_base_addr + dest_block_id * state_block_stride + row * dim_row_stride
            for i in range(0, per_row_bytes, COPY_BLOCK_SIZE):
                mask = (i + offsets) < per_row_bytes
                curr_src = (row_src + i + offsets).to(tl.pointer_type(tl.uint8))
                curr_dst = (row_dst + i + offsets).to(tl.pointer_type(tl.uint8))
                data = tl.load(curr_src, mask=mask)
                tl.store(curr_dst, data, mask=mask)
        return

    if is_conv_state:
        if tile_idx > 0:
            return
        # SD conv: copy
        #   state[bt[src_idx], accept_token_bias:] ->
        #   state[bt[dest_idx], :conv_width - accept_token_bias]
        # Small per-block bytes make tiling degenerate, so conv runs as a
        # single-CTA memcpy (NUM_TILES=1).
        src_offset = accept_token_bias.to(tl.int64) * state_inner_size * state_elem_size
        src_addr = state_base_addr + src_block_id * state_block_stride + src_offset
        dst_addr = state_base_addr + dest_block_id * state_block_stride
        copy_size = (conv_width - accept_token_bias).to(tl.int64) * state_inner_size * state_elem_size
        _memcpy_u64_tiled(
            src_addr,
            dst_addr,
            copy_size,
            tile_idx,
            COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,
            NUM_TILES=1,
        )
        return

    # Temporal state: copy state[bt[src + bias]] -> state[bt[dest]]
    # Body u64 range is partitioned across TEMPORAL_TILES CTAs to keep the
    # SMs filled at small batch.
    actual_src_block_idx = src_block_idx + accept_token_bias
    actual_src_block_id = tl.load(block_table_base + actual_src_block_idx).to(tl.int64)
    src_addr = state_base_addr + actual_src_block_id * state_block_stride
    dst_addr = state_base_addr + dest_block_id * state_block_stride
    # Use natural block data size (inner_size * elem_size), NOT
    # state_block_stride which is the page stride and can exceed the
    # actual data when the state tensor uses as_strided page padding.
    copy_size = state_inner_size * state_elem_size
    _memcpy_u64_tiled(
        src_addr,
        dst_addr,
        copy_size,
        tile_idx,
        COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,
        NUM_TILES=TEMPORAL_TILES,
    )
