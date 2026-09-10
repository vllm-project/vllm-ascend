# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mapping_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)
    block_table_stride,  # max_num_blocks_per_req
    block_size,  # Logical block size used by the attention kernel
    slot_mapping_ptr,  # [max_num_tokens], int32
    KV_CACHE_BLOCK_SIZE: tl.constexpr,  # Physical KV cache allocation block size
    BLOCKS_PER_KV_BLOCK: tl.constexpr,  # KV_CACHE_BLOCK_SIZE = BLOCKS_PER_KV_BLOCK * block_size
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TILE_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx == tl.num_programs(0) - 1:
        # Pad remaining slots for CUDA graph compatibility.
        for i in range(num_tokens, max_num_tokens, TILE_BLOCK_SIZE):
            offsets = i + tl.arange(0, TILE_BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    row_offset = req_idx * block_table_stride
    block_table_offsets = tl.arange(0, BLOCK_TABLE_WINDOW_SIZE)
    for i in range(start_idx, end_idx, TILE_BLOCK_SIZE):
        offsets = i + tl.arange(0, TILE_BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0).to(tl.int32)
        if TOTAL_CP_WORLD_SIZE == 1:
            block_indices = pos // block_size
            slot_offsets = pos - block_indices * block_size
        else:
            virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
            virtual_block_indices = pos // virtual_block_size
            virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
            is_local = (virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
            local_block_offsets = (
                virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
            ) * CP_KV_CACHE_INTERLEAVE_SIZE + (virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE)

            block_indices = virtual_block_indices * BLOCKS_PER_KV_BLOCK + local_block_offsets // block_size
            slot_offsets = local_block_offsets % block_size

        INT32_MAX = 2147483647
        valid_block_indices = tl.where(mask, block_indices, INT32_MAX)
        block_idx_base = tl.min(valid_block_indices, axis=0)
        block_table_window_offsets = block_idx_base + block_table_offsets
        block_table_window = tl.load(
            block_table_ptr + row_offset + block_table_window_offsets,
            mask=block_table_window_offsets < block_table_stride,
            other=0,
        ).to(tl.float32)
        if TOTAL_CP_WORLD_SIZE == 1:
            relative_block_indices = tl.where(mask, block_indices - block_idx_base, 0)
        else:
            relative_block_indices = tl.where(mask & is_local, block_indices - block_idx_base, 0)
        block_numbers = tl.gather(block_table_window, relative_block_indices, 0).to(tl.int32)
        slot_ids = block_numbers * block_size + slot_offsets
        if TOTAL_CP_WORLD_SIZE != 1:
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


@triton.jit
def _compute_slot_mapping_request(
    start_idx,
    end_idx,
    req_idx,
    positions_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    slot_mapping_ptr,
    KV_CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_KV_BLOCK: tl.constexpr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TILE_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    row_offset = req_idx * block_table_stride
    block_table_offsets = tl.arange(0, BLOCK_TABLE_WINDOW_SIZE)
    for i in range(start_idx, end_idx, TILE_BLOCK_SIZE):
        offsets = i + tl.arange(0, TILE_BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0).to(tl.int32)
        if TOTAL_CP_WORLD_SIZE == 1:
            block_indices = pos // block_size
            slot_offsets = pos - block_indices * block_size
        else:
            virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
            virtual_block_indices = pos // virtual_block_size
            virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
            is_local = (virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
            local_block_offsets = (
                virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
            ) * CP_KV_CACHE_INTERLEAVE_SIZE + (virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE)

            block_indices = virtual_block_indices * BLOCKS_PER_KV_BLOCK + local_block_offsets // block_size
            slot_offsets = local_block_offsets % block_size

        INT32_MAX = 2147483647
        valid_block_indices = tl.where(mask, block_indices, INT32_MAX)
        block_idx_base = tl.min(valid_block_indices, axis=0)
        block_table_window_offsets = block_idx_base + block_table_offsets
        block_table_window = tl.load(
            block_table_ptr + row_offset + block_table_window_offsets,
            mask=block_table_window_offsets < block_table_stride,
            other=0,
        ).to(tl.float32)
        if TOTAL_CP_WORLD_SIZE == 1:
            relative_block_indices = tl.where(mask, block_indices - block_idx_base, 0)
        else:
            relative_block_indices = tl.where(mask & is_local, block_indices - block_idx_base, 0)
        block_numbers = tl.gather(block_table_window, relative_block_indices, 0).to(tl.int32)
        slot_ids = block_numbers * block_size + slot_offsets
        if TOTAL_CP_WORLD_SIZE != 1:
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mapping_fused_groups_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_addrs_ptr,
    slot_mapping_addrs_ptr,
    block_table_strides_ptr,
    block_sizes_ptr,
    PAD_ID: tl.constexpr,
    NUM_REQS: tl.constexpr,
    TILE_BLOCK_SIZE: tl.constexpr,
    PARALLEL_TILES: tl.constexpr,
    BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    program_idx = tl.program_id(0)
    programs_per_group: tl.constexpr = NUM_REQS * PARALLEL_TILES + 1
    group_idx = program_idx // programs_per_group
    group_program_idx = program_idx - group_idx * programs_per_group

    block_table_addr = tl.load(block_table_addrs_ptr + group_idx)
    slot_mapping_addr = tl.load(slot_mapping_addrs_ptr + group_idx)
    block_table_ptr = tl.cast(block_table_addr, tl.pointer_type(tl.int32))
    slot_mapping_ptr = tl.cast(slot_mapping_addr, tl.pointer_type(tl.int32))

    if group_program_idx == programs_per_group - 1:
        for i in range(num_tokens, max_num_tokens, TILE_BLOCK_SIZE):
            offsets = i + tl.arange(0, TILE_BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    req_idx = group_program_idx // PARALLEL_TILES
    tile_idx = group_program_idx - req_idx * PARALLEL_TILES
    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)
    block_table_stride = tl.load(block_table_strides_ptr + group_idx)
    block_size = tl.load(block_sizes_ptr + group_idx)
    row_offset = req_idx * block_table_stride
    block_table_offsets = tl.arange(0, BLOCK_TABLE_WINDOW_SIZE)
    for i in range(
        start_idx + tile_idx * TILE_BLOCK_SIZE,
        end_idx,
        TILE_BLOCK_SIZE * PARALLEL_TILES,
    ):
        offsets = i + tl.arange(0, TILE_BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0).to(tl.int32)
        block_indices = pos // block_size
        slot_offsets = pos - block_indices * block_size

        INT32_MAX = 2147483647
        valid_block_indices = tl.where(mask, block_indices, INT32_MAX)
        block_idx_base = tl.min(valid_block_indices, axis=0)
        block_table_window_offsets = block_idx_base + block_table_offsets
        block_table_window = tl.load(
            block_table_ptr + row_offset + block_table_window_offsets,
            mask=block_table_window_offsets < block_table_stride,
            other=0,
        ).to(tl.float32)
        relative_block_indices = tl.where(mask, block_indices - block_idx_base, 0)
        block_numbers = tl.gather(block_table_window, relative_block_indices, 0).to(tl.int32)
        slot_ids = block_numbers * block_size + slot_offsets
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mapping_fused_groups_adaptive_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_addrs_ptr,
    slot_mapping_addrs_ptr,
    block_table_strides_ptr,
    block_sizes_ptr,
    PAD_ID: tl.constexpr,
    NUM_REQS: tl.constexpr,
    SMALL_TILE_BLOCK_SIZE: tl.constexpr,
    SMALL_BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
    LARGE_BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    program_idx = tl.program_id(0)
    programs_per_group: tl.constexpr = NUM_REQS + 1
    group_idx = program_idx // programs_per_group
    group_program_idx = program_idx - group_idx * programs_per_group

    block_table_addr = tl.load(block_table_addrs_ptr + group_idx)
    slot_mapping_addr = tl.load(slot_mapping_addrs_ptr + group_idx)
    block_table_ptr = tl.cast(block_table_addr, tl.pointer_type(tl.int32))
    slot_mapping_ptr = tl.cast(slot_mapping_addr, tl.pointer_type(tl.int32))

    if group_program_idx == NUM_REQS:
        for i in range(num_tokens, max_num_tokens, 1024):
            offsets = i + tl.arange(0, 1024)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    req_idx = group_program_idx
    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)
    block_table_stride = tl.load(block_table_strides_ptr + group_idx)
    block_size = tl.load(block_sizes_ptr + group_idx)
    request_tokens = end_idx - start_idx
    if request_tokens <= SMALL_TILE_BLOCK_SIZE:
        _compute_slot_mapping_request(
            start_idx,
            end_idx,
            req_idx,
            positions_ptr,
            block_table_ptr,
            block_table_stride,
            block_size,
            slot_mapping_ptr,
            1,
            1,
            1,
            0,
            1,
            PAD_ID,
            SMALL_TILE_BLOCK_SIZE,
            SMALL_BLOCK_TABLE_WINDOW_SIZE,
        )
    else:
        _compute_slot_mapping_request(
            start_idx,
            end_idx,
            req_idx,
            positions_ptr,
            block_table_ptr,
            block_table_stride,
            block_size,
            slot_mapping_ptr,
            1,
            1,
            1,
            0,
            1,
            PAD_ID,
            1024,
            LARGE_BLOCK_TABLE_WINDOW_SIZE,
        )


def _select_slot_mapping_launch_config(
    num_reqs: int,
    num_tokens: int,
    max_tile_block_size: int = 1024,
) -> tuple[int, int]:
    tile_block_size = max_tile_block_size
    if num_reqs > 1:
        tokens_per_req = (num_tokens + num_reqs - 1) // num_reqs
        tokens_per_req = 1 if tokens_per_req < 1 else tokens_per_req
        request_tile = _next_power_of_2(tokens_per_req)
        request_tile = 16 if request_tile < 16 else request_tile
        tile_block_size = request_tile if request_tile < max_tile_block_size else max_tile_block_size

    parallel_tiles = (num_tokens + tile_block_size - 1) // tile_block_size
    parallel_tiles = 4 if parallel_tiles > 4 else parallel_tiles
    if num_reqs == 1:
        parallel_tiles = parallel_tiles if num_tokens >= 2 * tile_block_size else 1
    elif num_reqs == 2 and num_tokens >= 4 * tile_block_size:
        parallel_tiles = 2
    else:
        parallel_tiles = 1
    return tile_block_size, parallel_tiles


def compute_slot_mapping_fused_groups(
    group_count,
    num_reqs,
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_addrs_ptr,
    slot_mapping_addrs_ptr,
    block_table_strides_ptr,
    block_sizes_ptr,
    min_block_size,
    *,
    pad_id,
):
    tile_block_size, parallel_tiles = _select_slot_mapping_launch_config(num_reqs, num_tokens)
    block_table_window_size = _next_power_of_2((tile_block_size + min_block_size - 1) // min_block_size + 1)
    if num_reqs > 1 and tile_block_size < 1024:
        _compute_slot_mapping_fused_groups_adaptive_kernel[(group_count * (num_reqs + 1),)](
            num_tokens,
            max_num_tokens,
            query_start_loc_ptr,
            positions_ptr,
            block_table_addrs_ptr,
            slot_mapping_addrs_ptr,
            block_table_strides_ptr,
            block_sizes_ptr,
            PAD_ID=pad_id,
            NUM_REQS=num_reqs,
            SMALL_TILE_BLOCK_SIZE=tile_block_size,
            SMALL_BLOCK_TABLE_WINDOW_SIZE=block_table_window_size,
            LARGE_BLOCK_TABLE_WINDOW_SIZE=_next_power_of_2((1024 + min_block_size - 1) // min_block_size + 1),
        )
    else:
        programs_per_group = num_reqs * parallel_tiles + 1
        _compute_slot_mapping_fused_groups_kernel[(group_count * programs_per_group,)](
            num_tokens,
            max_num_tokens,
            query_start_loc_ptr,
            positions_ptr,
            block_table_addrs_ptr,
            slot_mapping_addrs_ptr,
            block_table_strides_ptr,
            block_sizes_ptr,
            PAD_ID=pad_id,
            NUM_REQS=num_reqs,
            TILE_BLOCK_SIZE=tile_block_size,
            PARALLEL_TILES=parallel_tiles,
            BLOCK_TABLE_WINDOW_SIZE=block_table_window_size,
        )
