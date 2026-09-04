# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.v2.block_table.compute_slot_mappings import _compute_slot_mappings_kernel


@pytest.mark.parametrize("cp_size, cp_rank, expected", [(1, 0, [40, 43, 44, 47]), (2, 1, [-1, 41, -1, 43])])
@pytest.mark.parametrize("has_enablement", [False, True])
def test_circular_buffer_slot_mapping_disabled(cp_size, cp_rank, expected, has_enablement):
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
        slots,
        slots.stride(0),
        cp_rank,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=1,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=1024,
        BLOCK_TABLE_PAD_SIZE=2,
        slot_mapping_enabled=enabled,
        HAS_SLOT_MAPPING_ENABLED=has_enablement,
    )
    second_group = [-1] * 4 if has_enablement else [slot + 40 if slot != -1 else -1 for slot in expected]
    torch.testing.assert_close(
        slots.cpu(), torch.tensor([expected + [-1] * 4, second_group + [-1] * 4], dtype=torch.int32)
    )
