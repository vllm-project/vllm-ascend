# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.v2.block_table.compute_slot_mappings import _compute_slot_mappings_kernel


@pytest.mark.parametrize("cp_size, cp_rank, expected", [(1, 0, [40, 43, 44, 47]), (2, 1, [-1, 41, -1, 43])])
@pytest.mark.parametrize("has_enablement", [False, True])
@pytest.mark.parametrize("use_block_table_staging", [False, True])
def test_circular_buffer_slot_mapping_disabled(cp_size, cp_rank, expected, has_enablement, use_block_table_staging):
    """PR #53896: disabled state groups emit PAD without token-indexing rows."""
    device = "npu"
    block_tables = [
        torch.tensor([[10, 11]], dtype=torch.int32, device=device),
        torch.tensor([[20, 21]], dtype=torch.int32, device=device),
    ]
    pointers = torch.tensor([table.data_ptr() for table in block_tables], dtype=torch.uint64, device=device)
    strides = torch.tensor([2, 2], dtype=torch.int64, device=device)
    block_sizes = torch.tensor([4, 4], dtype=torch.int32, device=device)
    enabled = torch.tensor([True, False], dtype=torch.bool, device=device) if has_enablement else None
    slots = torch.full((2, 8), 777, dtype=torch.int32, device=device)
    _compute_slot_mappings_kernel[(2, 2)](
        8,
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.tensor([0, 4], dtype=torch.int32, device=device),
        torch.tensor([0, 3, 4, 7], dtype=torch.int64, device=device),
        pointers,
        strides,
        block_sizes,
        block_sizes,
        slots,
        slots.stride(0),
        cp_rank,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=1,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=1024,
        BLOCK_TABLE_PAD_SIZE=2,
        USE_BLOCK_TABLE_STAGING=use_block_table_staging,
        slot_mapping_enabled=enabled,
        HAS_SLOT_MAPPING_ENABLED=has_enablement,
    )
    second_group = [-1] * 4 if has_enablement else [slot + 40 if slot != -1 else -1 for slot in expected]
    torch.testing.assert_close(
        slots.cpu(), torch.tensor([expected + [-1] * 4, second_group + [-1] * 4], dtype=torch.int32)
    )


@pytest.mark.parametrize("kv_block_size", [128, 1536, 2048])
@pytest.mark.parametrize("cp_size, cp_rank, cp_interleave", [(1, 0, 1), (2, 0, 1), (2, 1, 4)])
@pytest.mark.parametrize("use_block_table_staging", [False, True])
@pytest.mark.parametrize("has_enablement", [False, True])
def test_split_kernel_block_slot_mapping(
    kv_block_size, cp_size, cp_rank, cp_interleave, use_block_table_staging, has_enablement
):
    """Expanded block IDs must use kernel sizes, including hybrid MTP caches."""
    device = "npu"
    kernel_block_size = 128
    blocks_per_kv_block = kv_block_size // kernel_block_size
    physical_blocks = [[3, 1], [2, 5]]
    rows = [
        [block * blocks_per_kv_block + i for block in row for i in range(blocks_per_kv_block)]
        for row in physical_blocks
    ]
    block_table = torch.tensor(rows, dtype=torch.int32, device=device)
    positions_per_req = [
        0,
        1,
        3,
        127,
        128,
        kv_block_size * cp_size - 1,
        kv_block_size * cp_size,
        kv_block_size * cp_size + 127,
    ]
    positions = positions_per_req * 2
    actual_tokens = len(positions)
    max_tokens = actual_tokens + 5
    slots = torch.full((1, max_tokens), 777, dtype=torch.int32, device=device)
    enabled = torch.tensor([True], dtype=torch.bool, device=device) if has_enablement else None
    _compute_slot_mappings_kernel[(1, 3)](
        max_tokens,
        torch.tensor([1, 0], dtype=torch.int32, device=device),
        torch.tensor([0, len(positions_per_req), actual_tokens], dtype=torch.int32, device=device),
        torch.tensor(positions, dtype=torch.int64, device=device),
        torch.tensor([block_table.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([block_table.stride(0)], dtype=torch.int64, device=device),
        torch.tensor([kv_block_size], dtype=torch.int32, device=device),
        torch.tensor([kernel_block_size], dtype=torch.int32, device=device),
        slots,
        slots.stride(0),
        cp_rank,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=1024,
        BLOCK_TABLE_PAD_SIZE=1 << (len(rows[0]) - 1).bit_length(),
        USE_BLOCK_TABLE_STAGING=use_block_table_staging,
        slot_mapping_enabled=enabled,
        HAS_SLOT_MAPPING_ENABLED=has_enablement,
    )
    # Oracle uses physical KV pages, independent of expanded kernel-block IDs.
    expected = []
    for req in [1, 0]:
        for position in positions_per_req:
            logical_block, offset = divmod(position, kv_block_size * cp_size)
            if offset // cp_interleave % cp_size != cp_rank:
                expected.append(-1)
            else:
                local_offset = offset // (cp_interleave * cp_size) * cp_interleave + offset % cp_interleave
                expected.append(physical_blocks[req][logical_block] * kv_block_size + local_offset)
    expected += [-1] * (max_tokens - actual_tokens)
    torch.testing.assert_close(slots.cpu(), torch.tensor([expected], dtype=torch.int32))
