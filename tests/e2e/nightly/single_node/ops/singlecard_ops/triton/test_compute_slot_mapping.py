# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.block_table import (
    _compute_slot_mappings_kernel as ref_compute_slot_mappings_kernel,
)

from vllm_ascend.ops.triton.v2.block_table.compute_slot_mappings import (
    _compute_slot_mappings_kernel as ascend_compute_slot_mappings_kernel,
)
from vllm_ascend.worker.v2.block_table import (
    AscendBlockTables,
)


@pytest.mark.parametrize(
    ("cp_size", "cp_rank", "cp_interleave"),
    [
        pytest.param(1, 0, 1, id="cp1"),
        pytest.param(2, 1, 2, id="cp2_interleaved"),
        pytest.param(4, 2, 1, id="cp4_interleaved"),
    ],
)
def test_compute_slot_mapping_npu_kernel_cp(cp_size: int, cp_rank: int, cp_interleave: int) -> None:
    """Check the Ascend V2 kernel against the upstream kernel."""
    device = "npu"
    max_num_tokens = 8192
    idx_mapping = torch.tensor([2, 0], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 5, 10], dtype=torch.int32, device=device)
    positions = torch.tensor(
        [0, 1, 63, 64, 127, 0, 2, 64, 128, 255],
        dtype=torch.int64,
        device=device,
    )

    num_groups = 2
    block_tables = [torch.randint(0, 320, (3, 320), dtype=torch.int32, device=device) for _ in range(num_groups)]
    block_table_ptrs = torch.tensor(
        [table.data_ptr() for table in block_tables],
        dtype=torch.uint64,
        device=device,
    )
    block_table_strides = torch.tensor(
        [table.stride(0) for table in block_tables],
        dtype=torch.int64,
        device=device,
    )
    block_sizes = torch.tensor([64, 128], dtype=torch.int32, device=device)
    slot_mappings = torch.zeros((num_groups, max_num_tokens), dtype=torch.int32, device=device)
    ref_slot_mappings = torch.zeros_like(slot_mappings)

    kernel_args = (
        max_num_tokens,
        idx_mapping,
        query_start_loc,
        positions,
        block_table_ptrs,
        block_table_strides,
        block_sizes,
    )
    grid = (num_groups, idx_mapping.shape[0] + 1)
    kernel_kwargs = {
        "CP_SIZE": cp_size,
        "CP_INTERLEAVE": cp_interleave,
        "PAD_ID": -1,
        "TRITON_BLOCK_SIZE": 1024,
    }
    ascend_compute_slot_mappings_kernel[grid](
        *kernel_args,
        slot_mappings,
        slot_mappings.stride(0),
        cp_rank,
        **kernel_kwargs,
        BLOCK_TABLE_PAD_SIZE=triton.next_power_of_2(max(table.stride(0) for table in block_tables)),
    )
    ref_compute_slot_mappings_kernel[grid](
        *kernel_args,
        ref_slot_mappings,
        ref_slot_mappings.stride(0),
        cp_rank,
        **kernel_kwargs,
    )

    torch.testing.assert_close(slot_mappings, ref_slot_mappings)


def test_ascend_block_tables_compute_slot_mappings_out() -> None:
    """Exercise the V2 override, including its current ``out`` argument."""
    device = torch.device("npu")
    block_table = torch.tensor(
        [[10, 11, 0, 0], [20, 21, 0, 0], [30, 31, 0, 0]],
        dtype=torch.int32,
        device=device,
    )
    # This is a method-level test. Constructing BlockTables also exercises the
    # unrelated UvaBuffer lifecycle, which is covered independently upstream.
    block_tables = object.__new__(AscendBlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.slot_mappings = torch.empty((1, 12), dtype=torch.int32, device=device)
    block_tables.block_table_ptrs = torch.tensor([block_table.data_ptr()], dtype=torch.uint64, device=device)
    block_tables.block_table_strides = torch.tensor([block_table.stride(0)], dtype=torch.int64, device=device)
    block_tables.block_sizes_tensor = torch.tensor([4], dtype=torch.int32, device=device)
    block_tables.cp_rank = 0
    block_tables.cp_size = 1
    block_tables.cp_interleave = 1
    block_tables._block_table_pad_size = triton.next_power_of_2(block_table.stride(0))

    out = torch.full((1, 12), 777, dtype=torch.int32, device=device)
    result = block_tables.compute_slot_mappings(
        torch.tensor([2, 0], dtype=torch.int32, device=device),
        torch.tensor([0, 3, 6], dtype=torch.int32, device=device),
        torch.tensor([0, 3, 4, 0, 4, 7], dtype=torch.int64, device=device),
        num_tokens_padded=8,
        out=out,
    )

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        result.cpu(),
        torch.tensor([[120, 123, 124, 40, 44, 47, -1, -1]], dtype=torch.int32),
    )
    torch.testing.assert_close(out[:, 8:].cpu(), torch.full((1, 4), -1, dtype=torch.int32))
