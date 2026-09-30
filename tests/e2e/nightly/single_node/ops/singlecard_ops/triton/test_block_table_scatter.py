# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.triton.block_table_scatter import scatter_block_table
from vllm_ascend.worker.block_table import OptimizedBlockTable


@pytest.mark.parametrize("length", [1, 1023, 1024, 1025, 4097])
def test_scatter_preserves_other_rows_and_tails(length):
    # The view has a larger row stride than its logical width.
    storage = torch.full((4, length + 32), -7, dtype=torch.int32, device="npu")
    table = storage[:, : length + 8]
    packed = torch.arange(length + 3, dtype=torch.int32, device="npu")
    metadata = torch.tensor([[0, 2, length, 0], [3, 1, 3, length]], dtype=torch.int32, device="npu")
    expected = storage.cpu()
    expected[0, 2 : 2 + length] = torch.arange(length, dtype=torch.int32)
    expected[3, 1:4] = torch.arange(length, length + 3, dtype=torch.int32)
    scatter_block_table(packed, metadata, table, 2)
    torch.testing.assert_close(storage.cpu(), expected)


def test_empty_scatter():
    table = torch.full((2, 4), -7, dtype=torch.int32, device="npu")
    scatter_block_table(
        torch.empty(0, dtype=torch.int32, device="npu"), torch.empty((0, 4), dtype=torch.int32, device="npu"), table, 0
    )
    assert torch.all(table == -7)


@pytest.mark.parametrize("pin_memory", [False, True])
def test_async_commits_keep_staging_alive(pin_memory):
    with patch(
        "vllm_ascend.worker.block_table.get_dcp_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0)
    ):
        table = OptimizedBlockTable(128, 4, 4096, 64, pin_memory, torch.device("npu"))
    snapshots = []
    expected = []
    address = table.block_table.gpu.data_ptr()
    # Do not synchronize between steps: reuse both slots and trigger growth.
    for step in range(20):
        size = min(1 << (step // 3), 4096)
        table.add_row([step + 1] * size, 0)
        table.commit_dirty_ranges(1)
        snapshots.append(table.block_table.gpu[0, :size].clone())
        expected.append(torch.full((size,), step + 1, dtype=torch.int32))
    torch.npu.synchronize()
    for result, reference in zip(snapshots, expected):
        torch.testing.assert_close(result.cpu(), reference)
    assert table.block_table.gpu.data_ptr() == address
