# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
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
    IS_CIRCULAR: tl.constexpr = False,
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
        if IS_CIRCULAR:
            block_indices = tl.full((TILE_BLOCK_SIZE,), 0, tl.int32)
            slot_offsets = pos % block_size
        elif TOTAL_CP_WORLD_SIZE == 1:
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
        if IS_CIRCULAR:
            slot_ids = tl.where(pos >= 0, slot_ids, PAD_ID)
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
    is_circular,
    KV_CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_KV_BLOCK: tl.constexpr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    HAS_CIRCULAR: tl.constexpr,
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
            page_idx = pos // block_size
            block_indices = tl.where(is_circular, 0, page_idx)
            slot_offsets = pos - page_idx * block_size
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
        if HAS_CIRCULAR:
            # Negative positions (circular tail tokens) have no valid block:
            # keep them out of the window base and clamp the base so the
            # window load and the gather below stay in range even for
            # all-negative tiles.
            pos_valid = mask & (pos >= 0)
            valid_block_indices = tl.where(pos_valid, block_indices, INT32_MAX)
            block_idx_base = tl.min(valid_block_indices, axis=0)
            block_idx_base = tl.minimum(block_idx_base, block_table_stride - 1)
        else:
            valid_block_indices = tl.where(mask, block_indices, INT32_MAX)
            block_idx_base = tl.min(valid_block_indices, axis=0)
        block_table_window_offsets = block_idx_base + block_table_offsets
        block_table_window = tl.load(
            block_table_ptr + row_offset + block_table_window_offsets,
            mask=block_table_window_offsets < block_table_stride,
            other=0,
        ).to(tl.float32)
        if TOTAL_CP_WORLD_SIZE == 1:
            if HAS_CIRCULAR:
                relative_block_indices = tl.where(pos_valid, block_indices - block_idx_base, 0)
            else:
                relative_block_indices = tl.where(mask, block_indices - block_idx_base, 0)
        else:
            relative_block_indices = tl.where(mask & is_local, block_indices - block_idx_base, 0)
        block_numbers = tl.gather(block_table_window, relative_block_indices, 0).to(tl.int32)
        slot_ids = block_numbers * block_size + slot_offsets
        if HAS_CIRCULAR:
            slot_ids = tl.where(pos < 0, PAD_ID, slot_ids)
        if TOTAL_CP_WORLD_SIZE != 1:
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens", "num_reqs", "parallel_tiles"])
def _compute_slot_mapping_fused_groups_kernel(
    num_tokens,
    max_num_tokens,
    num_reqs,
    parallel_tiles,
    query_start_loc_ptr,
    positions_ptr,
    block_table_addrs_ptr,
    slot_mapping_addrs_ptr,
    block_table_strides_ptr,
    block_sizes_ptr,
    is_circular_ptr,
    HAS_CIRCULAR: tl.constexpr,
    PAD_ID: tl.constexpr,
    TILE_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    # Grid: (group_count, num_reqs + 1, parallel_tiles). num_reqs and
    # parallel_tiles are runtime arguments excluded from the specialization
    # key via do_not_specialize, so the JIT specialization count cannot grow
    # with the batch shape. Reading them from tl.num_programs() instead was
    # measured to cost ~27us of scalar time per launch on 910B (bisected via
    # msprof aiv_time: tl.num_programs() lowers to a pathological scalar path
    # on this backend), while do_not_specialize'd arguments are free.
    group_idx = tl.program_id(0)
    req_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)

    block_table_addr = tl.load(block_table_addrs_ptr + group_idx)
    slot_mapping_addr = tl.load(slot_mapping_addrs_ptr + group_idx)
    block_table_ptr = tl.cast(block_table_addr, tl.pointer_type(tl.int32))
    slot_mapping_ptr = tl.cast(slot_mapping_addr, tl.pointer_type(tl.int32))

    if req_idx == num_reqs:
        if tile_idx == 0:
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
    # Keep the stride in int32: an int64 scalar cascades through
    # row_offset / the window clamp / the window offset vector and forces
    # int64 vector arithmetic, which the Ascend backend lowers to ~2x more
    # scalar instructions (bisected: +9us per launch on 910B).
    block_table_stride = tl.load(block_table_strides_ptr + group_idx).to(tl.int32)
    block_size = tl.load(block_sizes_ptr + group_idx)
    is_circular = tl.load(is_circular_ptr + group_idx) if HAS_CIRCULAR else False
    row_offset = req_idx * block_table_stride
    block_table_offsets = tl.arange(0, BLOCK_TABLE_WINDOW_SIZE)
    for i in range(
        start_idx + tile_idx * TILE_BLOCK_SIZE,
        end_idx,
        TILE_BLOCK_SIZE * parallel_tiles,
    ):
        offsets = i + tl.arange(0, TILE_BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0).to(tl.int32)
        page_idx = pos // block_size
        block_indices = tl.where(is_circular, 0, page_idx)
        slot_offsets = pos - page_idx * block_size
        INT32_MAX = 2147483647
        if HAS_CIRCULAR:
            # Negative positions (circular tail tokens) have no valid block:
            # keep them out of the window base and clamp the base so the
            # window load and the gather below stay in range even for
            # all-negative tiles.
            pos_valid = mask & (pos >= 0)
            valid_block_indices = tl.where(pos_valid, block_indices, INT32_MAX)
            block_idx_base = tl.min(valid_block_indices, axis=0)
            block_idx_base = tl.minimum(block_idx_base, block_table_stride - 1)
        else:
            valid_block_indices = tl.where(mask, block_indices, INT32_MAX)
            block_idx_base = tl.min(valid_block_indices, axis=0)
        block_table_window_offsets = block_idx_base + block_table_offsets
        block_table_window = tl.load(
            block_table_ptr + row_offset + block_table_window_offsets,
            mask=block_table_window_offsets < block_table_stride,
            other=0,
        ).to(tl.float32)
        if HAS_CIRCULAR:
            relative_block_indices = tl.where(pos_valid, block_indices - block_idx_base, 0)
        else:
            relative_block_indices = tl.where(mask, block_indices - block_idx_base, 0)
        block_numbers = tl.gather(block_table_window, relative_block_indices, 0).to(tl.int32)
        slot_ids = block_numbers * block_size + slot_offsets
        if HAS_CIRCULAR:
            slot_ids = tl.where(pos < 0, PAD_ID, slot_ids)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens", "num_reqs"])
def _compute_slot_mapping_fused_groups_adaptive_kernel(
    num_tokens,
    max_num_tokens,
    num_reqs,
    query_start_loc_ptr,
    positions_ptr,
    block_table_addrs_ptr,
    slot_mapping_addrs_ptr,
    block_table_strides_ptr,
    block_sizes_ptr,
    is_circular_ptr,
    HAS_CIRCULAR: tl.constexpr,
    PAD_ID: tl.constexpr,
    SMALL_TILE_BLOCK_SIZE: tl.constexpr,
    SMALL_BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
    LARGE_BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    # Grid: (group_count, num_reqs + 1); the extra program per group pads
    # [num_tokens, max_num_tokens) with PAD_ID. num_reqs is a
    # do_not_specialize runtime argument, so the specialization count
    # cannot grow with the batch shape.
    group_idx = tl.program_id(0)
    req_idx = tl.program_id(1)

    block_table_addr = tl.load(block_table_addrs_ptr + group_idx)
    slot_mapping_addr = tl.load(slot_mapping_addrs_ptr + group_idx)
    block_table_ptr = tl.cast(block_table_addr, tl.pointer_type(tl.int32))
    slot_mapping_ptr = tl.cast(slot_mapping_addr, tl.pointer_type(tl.int32))

    if req_idx == num_reqs:
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
    block_table_stride = tl.load(block_table_strides_ptr + group_idx).to(tl.int32)
    block_size = tl.load(block_sizes_ptr + group_idx)
    is_circular = tl.load(is_circular_ptr + group_idx) if HAS_CIRCULAR else False
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
            is_circular,
            1,
            1,
            1,
            0,
            1,
            HAS_CIRCULAR,
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
            is_circular,
            1,
            1,
            1,
            0,
            1,
            HAS_CIRCULAR,
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
    is_circular_ptr=None,
):
    tile_block_size, parallel_tiles = _select_slot_mapping_launch_config(num_reqs, num_tokens)
    block_table_window_size = _next_power_of_2((tile_block_size + min_block_size - 1) // min_block_size + 1)
    if num_reqs > 1 and tile_block_size < 1024:
        _compute_slot_mapping_fused_groups_adaptive_kernel[(group_count, num_reqs + 1)](
            num_tokens,
            max_num_tokens,
            num_reqs,
            query_start_loc_ptr,
            positions_ptr,
            block_table_addrs_ptr,
            slot_mapping_addrs_ptr,
            block_table_strides_ptr,
            block_sizes_ptr,
            is_circular_ptr,
            HAS_CIRCULAR=is_circular_ptr is not None,
            PAD_ID=pad_id,
            SMALL_TILE_BLOCK_SIZE=tile_block_size,
            SMALL_BLOCK_TABLE_WINDOW_SIZE=block_table_window_size,
            LARGE_BLOCK_TABLE_WINDOW_SIZE=_next_power_of_2((1024 + min_block_size - 1) // min_block_size + 1),
        )
    else:
        _compute_slot_mapping_fused_groups_kernel[(group_count, num_reqs + 1, parallel_tiles)](
            num_tokens,
            max_num_tokens,
            num_reqs,
            parallel_tiles,
            query_start_loc_ptr,
            positions_ptr,
            block_table_addrs_ptr,
            slot_mapping_addrs_ptr,
            block_table_strides_ptr,
            block_sizes_ptr,
            is_circular_ptr,
            HAS_CIRCULAR=is_circular_ptr is not None,
            PAD_ID=pad_id,
            TILE_BLOCK_SIZE=tile_block_size,
            BLOCK_TABLE_WINDOW_SIZE=block_table_window_size,
        )


_FUSED_SLOT_MAPPING_TILE_LADDER = (16, 32, 64, 128, 256, 512)


def prewarm_fused_slot_mapping_kernels(
    min_block_size: int,
    max_num_batched_tokens: int,
    pad_id: int,
    device,
    is_circular_ptr=None,
) -> None:
    """Pre-compile every fused slot-mapping kernel specialization.

    ``num_reqs`` and ``parallel_tiles`` are ``do_not_specialize`` runtime
    arguments, so the JIT key set is bounded to one adaptive-kernel
    specialization per reachable tile size in
    ``_FUSED_SLOT_MAPPING_TILE_LADDER`` plus a single parallel-kernel
    specialization, per HAS_CIRCULAR variant.  Calling this during engine
    warmup ensures no request ever pays the multi-second Triton compile
    latency on the hot path.
    """
    max_tokens = 2048
    qsl_adaptive = torch.zeros(3, dtype=torch.int32, device=device)
    positions = torch.arange(max_tokens, dtype=torch.int64, device=device)
    # 256 columns keep the dummy window loads in range even for a
    # min_block_size as small as 8 at 2048 positions.
    block_table = torch.zeros((3, 256), dtype=torch.int32, device=device)
    block_table_addrs = torch.tensor([block_table.data_ptr()], dtype=torch.uint64, device=device)
    block_table_strides = torch.tensor([block_table.stride(0)], dtype=torch.int64, device=device)
    block_sizes = torch.tensor([min_block_size], dtype=torch.int32, device=device)
    slot_mapping = torch.zeros(max_tokens, dtype=torch.int32, device=device)
    slot_mapping_addrs = torch.tensor([slot_mapping.data_ptr()], dtype=torch.uint64, device=device)
    common = (block_table_addrs, slot_mapping_addrs, block_table_strides, block_sizes)

    for tile in _FUSED_SLOT_MAPPING_TILE_LADDER:
        # The adaptive selector maps an average tokens-per-request in
        # (tile / 2, tile] to this tile, so the tile is reachable as soon
        # as a mixed step can hold tile + 1 tokens (two requests); the
        # smallest tile is reachable whenever a mixed step is possible.
        if tile + 1 > max_num_batched_tokens and tile > _FUSED_SLOT_MAPPING_TILE_LADDER[0]:
            continue
        qsl_adaptive[1] = tile
        qsl_adaptive[2] = 2 * tile
        compute_slot_mapping_fused_groups(
            1,
            2,
            2 * tile,
            2 * tile,
            qsl_adaptive,
            positions,
            *common,
            min_block_size,
            pad_id=pad_id,
            is_circular_ptr=is_circular_ptr,
        )
    # Parallel kernel: TILE_BLOCK_SIZE is always 1024 on this path, so any
    # single-request batch compiles the only specialization.
    parallel_tokens = min(max_tokens, max_num_batched_tokens)
    qsl_parallel = torch.tensor([0, parallel_tokens], dtype=torch.int32, device=device)
    compute_slot_mapping_fused_groups(
        1,
        1,
        parallel_tokens,
        parallel_tokens,
        qsl_parallel,
        positions[:parallel_tokens],
        *common,
        min_block_size,
        pad_id=pad_id,
        is_circular_ptr=is_circular_ptr,
    )
    torch.npu.synchronize()
