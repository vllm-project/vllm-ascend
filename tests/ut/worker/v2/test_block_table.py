# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables

from vllm_ascend.worker.v2.block_table import AscendBlockTables, _MAX_STAGED_BLOCK_TABLE_PAD_SIZE


def _parent_init(
    self,
    block_sizes,
    max_num_reqs,
    max_num_batched_tokens,
    max_num_blocks_per_group,
    device,
    kernel_block_sizes,
    cp_size,
    cp_rank,
    cp_interleave,
):
    self.kernel_block_sizes = kernel_block_sizes
    self.num_kv_cache_groups = len(block_sizes)
    self.max_num_batched_tokens = max_num_batched_tokens
    self.device = device
    self.cp_size = cp_size
    self.cp_rank = cp_rank
    self.cp_interleave = cp_interleave
    self.block_tables = [SimpleNamespace(gpu=torch.zeros(2, 8))]
    self.block_table_ptrs = MagicMock()
    self.block_table_strides = MagicMock()
    self.block_sizes_tensor = MagicMock()
    self.slot_mappings = object()


def test_init_defaults_kernel_sizes_and_rebuilds_int32_slots():
    with (
        patch.object(BlockTables, "__init__", _parent_init),
        patch("vllm_ascend.worker.v2.block_table.triton.next_power_of_2", return_value=16),
    ):
        tables = AscendBlockTables([4], 2, 8, [4], torch.device("cpu"))
    assert tables.kernel_block_sizes == [4]
    assert tables._block_table_pad_size == 16
    assert tables.slot_mappings.dtype == torch.int32
    assert tables.slot_mappings.shape == (1, 8)


def test_init_keeps_explicit_kernel_block_sizes():
    with (
        patch.object(BlockTables, "__init__", _parent_init),
        patch("vllm_ascend.worker.v2.block_table.triton.next_power_of_2", return_value=8),
    ):
        tables = AscendBlockTables(
            [8],
            2,
            4,
            [2],
            torch.device("cpu"),
            kernel_block_sizes=[4],
        )
    assert tables.kernel_block_sizes == [4]


def test_compute_slot_mappings_launches_kernel_and_honors_out():
    tables = AscendBlockTables.__new__(AscendBlockTables)
    tables.num_kv_cache_groups = 2
    tables.slot_mappings = torch.zeros(2, 6, dtype=torch.int32)
    tables.block_table_ptrs = MagicMock(name="ptrs")
    tables.block_table_strides = MagicMock(name="strides")
    tables.block_sizes_tensor = MagicMock(name="sizes")
    tables.cp_rank = 1
    tables.cp_size = 2
    tables.cp_interleave = 4
    tables._block_table_pad_size = _MAX_STAGED_BLOCK_TABLE_PAD_SIZE
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    positions = torch.zeros(6, dtype=torch.int64)
    custom_out = torch.full((2, 6), 7, dtype=torch.int32)
    kernel = MagicMock()

    with patch("vllm_ascend.worker.v2.block_table._compute_slot_mappings_kernel", kernel):
        sliced = tables.compute_slot_mappings(idx_mapping, query_start_loc, positions, 3)
        reused = tables.compute_slot_mappings(
            idx_mapping, query_start_loc, positions, 4, out=custom_out
        )

    kernel.__getitem__.assert_called_with((2, 3))
    assert kernel.__getitem__.call_count == 2
    kwargs = kernel.__getitem__.return_value.call_args.kwargs
    assert kwargs["PAD_ID"] == PAD_SLOT_ID
    assert kwargs["USE_BLOCK_TABLE_STAGING"] is True
    assert sliced.shape == (2, 3)
    assert reused.shape == (2, 4)
    assert reused.data_ptr() == custom_out.data_ptr()
