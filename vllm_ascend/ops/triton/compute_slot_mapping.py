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
def _compute_slot_mapping_adaptive_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
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
    SMALL_TILE_BLOCK_SIZE: tl.constexpr,
    SMALL_BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
    LARGE_BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx == tl.num_programs(0) - 1:
        for i in range(num_tokens, max_num_tokens, 1024):
            offsets = i + tl.arange(0, 1024)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)
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
            KV_CACHE_BLOCK_SIZE,
            BLOCKS_PER_KV_BLOCK,
            TOTAL_CP_WORLD_SIZE,
            TOTAL_CP_RANK,
            CP_KV_CACHE_INTERLEAVE_SIZE,
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
            KV_CACHE_BLOCK_SIZE,
            BLOCKS_PER_KV_BLOCK,
            TOTAL_CP_WORLD_SIZE,
            TOTAL_CP_RANK,
            CP_KV_CACHE_INTERLEAVE_SIZE,
            PAD_ID,
            1024,
            LARGE_BLOCK_TABLE_WINDOW_SIZE,
        )


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mapping_parallel_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
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
    PARALLEL_TILES: tl.constexpr,
    BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    program_idx = tl.program_id(0)

    if program_idx == tl.num_programs(0) - 1:
        for i in range(num_tokens, max_num_tokens, 1024):
            offsets = i + tl.arange(0, 1024)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    req_idx = program_idx // PARALLEL_TILES
    tile_idx = program_idx - req_idx * PARALLEL_TILES
    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

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


def compute_slot_mapping(
    num_reqs,
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    slot_mapping_ptr,
    *,
    kv_cache_block_size,
    blocks_per_kv_block,
    total_cp_world_size,
    total_cp_rank,
    cp_kv_cache_interleave_size,
    pad_id,
    max_tile_block_size=1024,
    large_block_table_window_size=None,
):
    tile_block_size, parallel_tiles = _select_slot_mapping_launch_config(
        num_reqs,
        num_tokens,
        max_tile_block_size,
    )
    common_kernel_kwargs = {
        "KV_CACHE_BLOCK_SIZE": kv_cache_block_size,
        "BLOCKS_PER_KV_BLOCK": blocks_per_kv_block,
        "TOTAL_CP_WORLD_SIZE": total_cp_world_size,
        "TOTAL_CP_RANK": total_cp_rank,
        "CP_KV_CACHE_INTERLEAVE_SIZE": cp_kv_cache_interleave_size,
        "PAD_ID": pad_id,
    }
    block_table_window_size = _next_power_of_2((tile_block_size + block_size - 1) // block_size + 1)
    if large_block_table_window_size is None:
        large_block_table_window_size = _next_power_of_2((1024 + block_size - 1) // block_size + 1)

    if num_reqs > 1 and tile_block_size < 1024:
        _compute_slot_mapping_adaptive_kernel[(num_reqs + 1,)](
            num_tokens,
            max_num_tokens,
            query_start_loc_ptr,
            positions_ptr,
            block_table_ptr,
            block_table_stride,
            block_size,
            slot_mapping_ptr,
            SMALL_TILE_BLOCK_SIZE=tile_block_size,
            SMALL_BLOCK_TABLE_WINDOW_SIZE=block_table_window_size,
            LARGE_BLOCK_TABLE_WINDOW_SIZE=large_block_table_window_size,
            **common_kernel_kwargs,
        )
    elif parallel_tiles > 1:
        _compute_slot_mapping_parallel_kernel[(num_reqs * parallel_tiles + 1,)](
            num_tokens,
            max_num_tokens,
            query_start_loc_ptr,
            positions_ptr,
            block_table_ptr,
            block_table_stride,
            block_size,
            slot_mapping_ptr,
            TILE_BLOCK_SIZE=tile_block_size,
            PARALLEL_TILES=parallel_tiles,
            BLOCK_TABLE_WINDOW_SIZE=block_table_window_size,
            **common_kernel_kwargs,
        )
    else:
        _compute_slot_mapping_kernel[(num_reqs + 1,)](
            num_tokens,
            max_num_tokens,
            query_start_loc_ptr,
            positions_ptr,
            block_table_ptr,
            block_table_stride,
            block_size,
            slot_mapping_ptr,
            TILE_BLOCK_SIZE=tile_block_size,
            BLOCK_TABLE_WINDOW_SIZE=block_table_window_size,
            **common_kernel_kwargs,
        )
    return slot_mapping_ptr
