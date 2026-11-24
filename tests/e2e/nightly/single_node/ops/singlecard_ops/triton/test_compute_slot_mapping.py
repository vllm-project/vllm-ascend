# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.block_table.compute_slot_mappings import (
    _compute_slot_mappings_kernel as ascend_compute_slot_mappings_kernel,
)
from vllm_ascend.worker.v2.block_table import (
    AscendBlockTables,
)


def _cpu_slot_ids(table: torch.Tensor, row: int, positions: torch.Tensor, block_size: int) -> torch.Tensor:
    return table[row, positions // block_size] * block_size + positions % block_size


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize(
    ("cp_size", "cp_rank", "cp_interleave", "replicated_draft"),
    [
        # CP_SIZE=1 uses the same mapping for either draft layout.
        pytest.param(1, 0, 1, False, id="cp1-False"),
        pytest.param(2, 1, 2, False, id="cp2_interleaved-False"),
        pytest.param(2, 1, 2, True, id="cp2_interleaved-True"),
        pytest.param(4, 2, 1, False, id="cp4_interleaved-False"),
        pytest.param(4, 2, 1, True, id="cp4_interleaved-True"),
        pytest.param(4, 2, 128, False, id="cp4_logical_block_interleaved-False"),
        pytest.param(4, 2, 128, True, id="cp4_logical_block_interleaved-True"),
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
    manager_block_sizes = (128, 256)
    # Each non-empty request crosses a 1024-token tile boundary and starts one
    # token before a manager block boundary.
    position_starts = (127, 255)
    tokens_per_request = 1025
    query_start_loc = torch.tensor(
        [0, tokens_per_request, tokens_per_request, 2 * tokens_per_request],
        dtype=torch.int32,
        device=device,
    )
    positions = torch.cat(
        [torch.arange(start, start + tokens_per_request, dtype=torch.int64, device=device) for start in position_starts]
    )

    num_groups = len(manager_block_sizes)
    sharded = torch.tensor([True, not replicated_draft], dtype=torch.bool, device=device)
    num_req_states = idx_mapping.numel()
    block_tables = []
    for block_size in manager_block_sizes:
        table_width = triton.cdiv(max(position_starts) + tokens_per_request, block_size)
        table = torch.arange(1, num_req_states * table_width + 1, dtype=torch.int32, device=device)
        block_tables.append(table.view(num_req_states, table_width))
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
    kv_block_sizes = torch.tensor(manager_block_sizes, dtype=torch.int32, device=device)
    # A tile that starts just before a manager block boundary needs one extra entry.
    min_block_size = min(manager_block_sizes)
    block_table_window_size = triton.next_power_of_2(triton.cdiv(1024, min_block_size) + 1)
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
    host_mapping = idx_mapping.cpu().tolist()
    for group, (table, block_size, is_sharded) in enumerate(zip(block_tables, manager_block_sizes, sharded.tolist())):
        table = table.cpu()
        for req, row in enumerate(host_mapping):
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
            slots = _cpu_slot_ids(table, row, local, block_size)
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


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("disabled_group", [None, 4], ids=["all-groups", "disabled-group"])
def test_compute_slot_mapping_deepseek_v4_prefill(disabled_group: int | None) -> None:
    """Replay the slot-mapping input shapes captured from a DeepSeek V4 Flash prefill."""
    device = torch.device("npu")
    num_tokens = 79
    block_sizes = [128, 4096, 32, 32, 2, 8]
    block_counts = [triton.cdiv(num_tokens, size) for size in block_sizes]
    req_row = 1
    slot_mapping_enabled = [group != disabled_group for group in range(len(block_sizes))]
    block_tables = AscendBlockTables(
        block_sizes=block_sizes,
        max_num_reqs=4,
        max_num_batched_tokens=4096,
        max_num_blocks_per_group=block_counts,
        device=device,
        cp_size=1,
        slot_mapping_enabled=slot_mapping_enabled,
    )

    # The shape and block sizes came from the runner. Use synthetic block
    # IDs and give every group a distinct range to check group selection.
    block_ids = []
    next_id = 1
    for count in block_counts:
        block_ids.append(list(range(next_id, next_id + count)))
        next_id += count
    block_tables.append_block_ids(req_row, tuple(block_ids), overwrite=True)
    block_tables.apply_staged_writes()

    positions = torch.arange(num_tokens, dtype=torch.int64)
    result = block_tables.compute_slot_mappings(
        torch.tensor([req_row], dtype=torch.int64, device=device),
        torch.tensor([0, num_tokens], dtype=torch.int32, device=device),
        positions.to(device),
        num_tokens_padded=num_tokens,
    )

    expected = torch.stack(
        [
            _cpu_slot_ids(torch.tensor(ids, dtype=torch.int32).unsqueeze(0), 0, positions, size)
            for ids, size in zip(block_ids, block_sizes)
        ]
    )
    if disabled_group is not None:
        expected[disabled_group].fill_(PAD_SLOT_ID)
        disabled_slots = block_tables.slot_mappings[disabled_group].cpu()
        torch.testing.assert_close(disabled_slots, torch.full_like(disabled_slots, PAD_SLOT_ID), rtol=0, atol=0)
    torch.testing.assert_close(result.cpu(), expected.to(torch.int32), rtol=0, atol=0)
