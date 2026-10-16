# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.triton_utils import triton

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.v2.block_table.compute_slot_mappings import (
    _compute_slot_mappings_kernel as ascend_compute_slot_mappings_kernel,
)
from vllm_ascend.worker.v2.block_table import (
    AscendBlockTables,
)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("replicated_draft", [False, True])
@pytest.mark.parametrize(
    ("cp_size", "cp_rank", "cp_interleave"),
    [
        pytest.param(1, 0, 1, id="cp1"),
        pytest.param(2, 1, 2, id="cp2_interleaved"),
        pytest.param(4, 2, 1, id="cp4_interleaved"),
        pytest.param(4, 2, 128, id="cp4_logical_block_interleaved"),
    ],
)
def test_compute_slot_mapping_npu_kernel_cp(
    cp_size: int, cp_rank: int, cp_interleave: int, replicated_draft: bool
) -> None:
    """Check sharded and replicated groups against global-token CPU mapping."""
    device = "npu"
    max_num_tokens = 8192
    # Empty requests use -1 as their mapping sentinel. The Ascend kernel must
    # skip that row rather than resolving it while staging a block-table window.
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device=device)
    # Each non-empty request crosses a 1024-token tile boundary and starts one
    # token before a block boundary. This exercises the window's largest span.
    tokens_per_request = 1025
    query_start_loc = torch.tensor(
        [0, tokens_per_request, tokens_per_request, 2 * tokens_per_request],
        dtype=torch.int32,
        device=device,
    )
    positions = torch.cat(
        (
            torch.arange(63, 63 + tokens_per_request, dtype=torch.int64, device=device),
            torch.arange(127, 127 + tokens_per_request, dtype=torch.int64, device=device),
        )
    )

    num_groups = 2
    sharded = torch.tensor([True, not replicated_draft], dtype=torch.bool, device=device)
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
    # The block table is expanded in kernel-block units. The allocation and
    # kernel sizes deliberately differ to cover the V2 DCP mapping regression.
    kv_block_sizes = torch.tensor([128, 256], dtype=torch.int32, device=device)
    # Starting at positions 63 and 127 makes a 1024-token tile span the largest
    # window for the 64-token group.
    kernel_block_sizes = torch.tensor([64, 128], dtype=torch.int32, device=device)
    min_kernel_block_size = min(kernel_block_sizes.tolist())
    block_table_window_size = triton.next_power_of_2((1024 + min_kernel_block_size - 1) // min_kernel_block_size + 1)
    slot_mappings = torch.zeros((num_groups, max_num_tokens), dtype=torch.int32, device=device)
    ref_slot_mappings = torch.full((num_groups, max_num_tokens), -1, dtype=torch.int32)

    kernel_args = (
        max_num_tokens,
        idx_mapping,
        query_start_loc,
        positions,
        block_table_ptrs,
        block_table_strides,
        kv_block_sizes,
        kernel_block_sizes,
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
        BLOCK_TABLE_WINDOW_SIZE=block_table_window_size,
        dcp_sharded=sharded,
    )
    # Resolve ownership in global interleave units independently of the
    # kernel's allocation-page arithmetic and block-table window staging.
    boundaries = query_start_loc.cpu().tolist()
    global_positions = positions.cpu()
    for group, (table, kernel_size, is_sharded) in enumerate(
        zip(block_tables, kernel_block_sizes.cpu().tolist(), sharded.cpu().tolist())
    ):
        table = table.cpu()
        for req, row in enumerate(idx_mapping.cpu().tolist()):
            if row < 0:
                continue
            start, end = boundaries[req : req + 2]
            pos = global_positions[start:end]
            if is_sharded:
                local = pos // (cp_size * cp_interleave) * cp_interleave + pos % cp_interleave
                owned = pos // cp_interleave % cp_size == cp_rank
            else:
                local = pos
                owned = torch.ones_like(pos, dtype=torch.bool)
            slots = table[row, local // kernel_size] * kernel_size + local % kernel_size
            ref_slot_mappings[group, start:end] = torch.where(owned, slots, -1).to(torch.int32)

    torch.testing.assert_close(slot_mappings.cpu(), ref_slot_mappings)
    if replicated_draft:
        assert torch.all(slot_mappings[1, : positions.numel()] >= 0)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
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
    block_tables.kernel_block_sizes_tensor = torch.tensor([4], dtype=torch.int32, device=device)
    block_tables.cp_rank = 0
    block_tables.cp_size = 1
    block_tables.cp_interleave = 1
    block_tables._triton_block_size = 1024
    block_tables._block_table_window_size = 512
    block_tables.slot_mapping_enabled = torch.tensor([True], dtype=torch.bool, device=device)
    block_tables.dcp_sharded = torch.tensor([True], dtype=torch.bool, device=device)

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
