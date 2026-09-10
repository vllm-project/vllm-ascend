import pytest
import torch

from vllm_ascend.ops.triton.compute_slot_mapping import (
    _compute_slot_mapping_kernel,
    _next_power_of_2,
)

PAD_ID = -1
TRITON_BLOCK_SIZE = 1024


def _compute_slot_mapping_ref(
    query_start_loc: list[int],
    positions: list[int],
    block_table: list[list[int]],
    block_size: int,
    max_num_tokens: int,
    kv_cache_block_size: int,
    blocks_per_kv_block: int,
    total_cp_world_size: int,
    total_cp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> list[int]:
    slot_mapping = [PAD_ID] * max_num_tokens

    for req_idx, (start_idx, end_idx) in enumerate(zip(query_start_loc[:-1], query_start_loc[1:])):
        for token_idx in range(start_idx, end_idx):
            position = positions[token_idx]
            if total_cp_world_size == 1:
                block_idx = position // block_size
                slot_offset = position % block_size
            else:
                virtual_block_size = kv_cache_block_size * total_cp_world_size
                virtual_block_idx = position // virtual_block_size
                virtual_block_offset = position % virtual_block_size
                is_local = (virtual_block_offset // cp_kv_cache_interleave_size) % total_cp_world_size == total_cp_rank
                if not is_local:
                    continue

                local_block_offset = (
                    virtual_block_offset
                    // (total_cp_world_size * cp_kv_cache_interleave_size)
                    * cp_kv_cache_interleave_size
                    + virtual_block_offset % cp_kv_cache_interleave_size
                )
                block_idx = virtual_block_idx * blocks_per_kv_block + local_block_offset // block_size
                slot_offset = local_block_offset % block_size

            slot_mapping[token_idx] = block_table[req_idx][block_idx] * block_size + slot_offset

    return slot_mapping


@pytest.mark.parametrize(
    (
        "query_start_loc",
        "positions",
        "kv_cache_block_size",
        "block_size",
        "total_cp_world_size",
        "total_cp_rank",
        "cp_kv_cache_interleave_size",
    ),
    [
        pytest.param(
            [0, 10, 17],
            [0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 2, 5, 6, 9, 10, 13, 14],
            4,
            4,
            1,
            0,
            1,
            id="single_cp_rank",
        ),
        pytest.param(
            [0, 12, 22],
            list(range(12)) + [12, 13, 14, 15, 16, 17, 18, 19, 22, 23],
            8,
            4,
            2,
            1,
            2,
            id="interleaved_cp_rank",
        ),
        pytest.param(
            [0, 16, 32],
            [
                0,
                1,
                2,
                3,
                64,
                65,
                66,
                67,
                128,
                129,
                130,
                131,
                192,
                193,
                194,
                195,
                256,
                257,
                258,
                259,
                320,
                321,
                322,
                323,
                384,
                385,
                386,
                387,
                448,
                449,
                450,
                451,
            ],
            128,
            32,
            2,
            0,
            2,
            id="interleaved_cp_rank_zero_hybrid_blocks",
        ),
    ],
)
def test_compute_slot_mapping_kernel(
    query_start_loc,
    positions,
    kv_cache_block_size,
    block_size,
    total_cp_world_size,
    total_cp_rank,
    cp_kv_cache_interleave_size,
):
    device = "npu"
    max_num_tokens = 32
    block_table = [
        [5, 7, 11, 13, 17, 19, 23, 29],
        [31, 37, 41, 43, 47, 53, 59, 61],
    ]
    blocks_per_kv_block = kv_cache_block_size // block_size
    num_tokens = len(positions)

    query_start_loc_tensor = torch.tensor(query_start_loc, dtype=torch.int32, device=device)
    positions_tensor = torch.tensor(positions, dtype=torch.int64, device=device)
    block_table_tensor = torch.tensor(block_table, dtype=torch.int32, device=device)
    slot_mapping = torch.full((max_num_tokens,), 123456, dtype=torch.int32, device=device)

    _compute_slot_mapping_kernel[(len(query_start_loc),)](
        num_tokens,
        max_num_tokens,
        query_start_loc_tensor,
        positions_tensor,
        block_table_tensor,
        block_table_tensor.stride(0),
        block_size,
        slot_mapping,
        KV_CACHE_BLOCK_SIZE=kv_cache_block_size,
        BLOCKS_PER_KV_BLOCK=blocks_per_kv_block,
        TOTAL_CP_WORLD_SIZE=total_cp_world_size,
        TOTAL_CP_RANK=total_cp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
        PAD_ID=PAD_ID,
        TILE_BLOCK_SIZE=TRITON_BLOCK_SIZE,
        BLOCK_TABLE_WINDOW_SIZE=_next_power_of_2((TRITON_BLOCK_SIZE + block_size - 1) // block_size + 1),
    )

    expected = _compute_slot_mapping_ref(
        query_start_loc,
        positions,
        block_table,
        block_size,
        max_num_tokens,
        kv_cache_block_size,
        blocks_per_kv_block,
        total_cp_world_size,
        total_cp_rank,
        cp_kv_cache_interleave_size,
    )
    torch.testing.assert_close(slot_mapping.cpu(), torch.tensor(expected, dtype=torch.int32))


def test_compute_slot_mapping_kernel_four_requests_large_sequences():
    device = "npu"
    sequence_lengths = (1024, 2048, 4096, 8192)
    query_start_loc = [0, 1024, 3072, 7168, 15360]
    positions = [position for seq_len in sequence_lengths for position in range(seq_len)]
    block_size = 128
    kv_cache_block_size = 128
    blocks_per_kv_block = kv_cache_block_size // block_size
    blocks_per_request = max(sequence_lengths) // block_size
    max_num_tokens = 16384
    num_tokens = len(positions)

    block_table = [
        list(range(req_idx * blocks_per_request, (req_idx + 1) * blocks_per_request))
        for req_idx in range(len(sequence_lengths))
    ]

    query_start_loc_tensor = torch.tensor(query_start_loc, dtype=torch.int32, device=device)
    positions_tensor = torch.tensor(positions, dtype=torch.int64, device=device)
    block_table_tensor = torch.tensor(block_table, dtype=torch.int32, device=device)
    slot_mapping = torch.full((max_num_tokens,), 123456, dtype=torch.int32, device=device)

    _compute_slot_mapping_kernel[(len(query_start_loc),)](
        num_tokens,
        max_num_tokens,
        query_start_loc_tensor,
        positions_tensor,
        block_table_tensor,
        block_table_tensor.stride(0),
        block_size,
        slot_mapping,
        KV_CACHE_BLOCK_SIZE=kv_cache_block_size,
        BLOCKS_PER_KV_BLOCK=blocks_per_kv_block,
        TOTAL_CP_WORLD_SIZE=1,
        TOTAL_CP_RANK=0,
        CP_KV_CACHE_INTERLEAVE_SIZE=1,
        PAD_ID=PAD_ID,
        TILE_BLOCK_SIZE=TRITON_BLOCK_SIZE,
        BLOCK_TABLE_WINDOW_SIZE=_next_power_of_2((TRITON_BLOCK_SIZE + block_size - 1) // block_size + 1),
    )

    expected = _compute_slot_mapping_ref(
        query_start_loc,
        positions,
        block_table,
        block_size,
        max_num_tokens,
        kv_cache_block_size,
        blocks_per_kv_block,
        1,
        0,
        1,
    )
    torch.testing.assert_close(slot_mapping.cpu(), torch.tensor(expected, dtype=torch.int32))
