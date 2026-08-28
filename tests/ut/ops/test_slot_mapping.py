#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from tests.ut.base import TestBase
from vllm_ascend.ops.triton.compute_slot_mapping import _next_power_of_2
from vllm_ascend.ops.triton.slot_mapping import compute_slot_mapping_fused_kernel


def launch_slot_mapping_fused(
    block_tables: list[Any],
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """One-shot launcher that builds per-group param arrays on the fly.

    .. note::
       This helper mirrors the test-only path. In production the
       ``MultiGroupBlockTable`` pre-caches param tensors during ``__init__``
       and calls the kernel directly, avoiding the ``torch.tensor(…)``
       construction overhead.

    Args:
        block_tables: List of non-mamba ``BlockTable`` instances.
        num_reqs: Number of requests in the batch.
        query_start_loc: GPU tensor ``[num_reqs + 1]`` (int32).
        positions: GPU tensor ``[num_tokens]`` (int64).
    """
    num_groups = len(block_tables)
    if num_groups == 1:
        block_tables[0].compute_slot_mapping(num_reqs, query_start_loc, positions)
        return

    device = query_start_loc.device
    num_tokens = positions.shape[0]
    num_reqs_plus_one = num_reqs + 1

    # ---- build per-group parameter arrays ------------------------------
    group_block_table_ptrs = torch.tensor(
        [bt.block_table.gpu.data_ptr() for bt in block_tables],
        dtype=torch.int64,
        device=device,
    )
    group_block_table_strides = torch.tensor(
        [bt.block_table.gpu.stride(0) for bt in block_tables],
        dtype=torch.int32,
        device=device,
    )
    group_block_sizes = torch.tensor(
        [bt.block_size for bt in block_tables],
        dtype=torch.int32,
        device=device,
    )
    group_slot_mapping_ptrs = torch.tensor(
        [bt.slot_mapping.gpu.data_ptr() for bt in block_tables],
        dtype=torch.int64,
        device=device,
    )
    group_kv_cache_block_sizes = torch.tensor(
        [bt.physical_block_size for bt in block_tables],
        dtype=torch.int32,
        device=device,
    )
    group_blocks_per_kv = torch.tensor(
        [bt.blocks_per_phys_block for bt in block_tables],
        dtype=torch.int32,
        device=device,
    )

    # ---- CP parameters & compile-time constants ------------------------
    bt0 = block_tables[0]
    total_cp_world_size = bt0.dcp_world_size
    total_cp_rank = bt0.dcp_rank

    tile_block_size = 1024
    min_block_size = min(bt.block_size for bt in block_tables)
    window_size = _next_power_of_2(((tile_block_size + min_block_size - 1) // min_block_size) + 1)

    # ---- launch --------------------------------------------------------
    compute_slot_mapping_fused_kernel[(num_reqs_plus_one, num_groups)](
        num_tokens,
        bt0.max_num_batched_tokens,
        query_start_loc,
        positions,
        group_block_table_ptrs,
        group_block_table_strides,
        group_block_sizes,
        group_slot_mapping_ptrs,
        group_kv_cache_block_sizes,
        group_blocks_per_kv,
        TOTAL_CP_WORLD_SIZE=total_cp_world_size,
        TOTAL_CP_RANK=total_cp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=bt0.cp_kv_cache_interleave_size,
        PAD_ID=PAD_SLOT_ID,
        TILE_BLOCK_SIZE=tile_block_size,
        BLOCK_TABLE_WINDOW_SIZE=window_size,
    )


class TestFusedSlotMapping(TestBase):
    """Test suite for the fused multi-group slot mapping Triton kernel."""

    def setUp(self):
        self.block_size = 128
        self.max_num_reqs = 4
        self.max_num_blocks_per_req = 128
        self.max_num_batched_tokens = 512
        self.device = torch.device("npu")
        self.kernel_sizes = [128]

    def create_block_table(self):
        """Create a real BlockTable with a mocked (single-rank) DCP group."""
        with patch("vllm_ascend.worker.block_table.get_dcp_group") as mock_get_dcp_group:
            mock_dcp_group = MagicMock(spec=GroupCoordinator)
            mock_dcp_group.world_size = 1
            mock_dcp_group.rank_in_group = 0
            mock_get_dcp_group.return_value = mock_dcp_group

            from vllm_ascend.worker.block_table import BlockTable

            return BlockTable(
                block_size=self.block_size,
                max_num_reqs=self.max_num_reqs,
                max_num_blocks_per_req=self.max_num_blocks_per_req,
                max_num_batched_tokens=self.max_num_batched_tokens,
                pin_memory=False,
                device=self.device,
                kernel_sizes=self.kernel_sizes,
                cp_kv_cache_interleave_size=1,
                num_speculative_tokens=0,
            )

    def setup_block_tables(self):
        """Create two groups with two requests each and return (bt0, bt1)."""
        bt0 = self.create_block_table()
        bt1 = self.create_block_table()
        bt0.add_row([0, 1, 2, 3], 0)
        bt0.add_row([8, 9, 10, 11], 1)
        bt1.add_row([4, 5, 6, 7], 0)
        bt1.add_row([12, 13, 14, 15], 1)
        return bt0, bt1

    def build_inputs(self):
        """4 tokens per request, 2 requests -> query_start_loc=[0,4,8]."""
        num_reqs = 2
        query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32, device=self.device)
        positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int64, device=self.device)
        return num_reqs, query_start_loc, positions

    def test_fused_matches_sequential_multi_group(self):
        """Fused kernel output must match per-group compute_slot_mapping."""
        bt0, bt1 = self.setup_block_tables()
        num_reqs, query_start_loc, positions = self.build_inputs()

        # Reference: sequential per-group computation.
        ref0 = bt0.slot_mapping.cpu.clone()
        ref1 = bt1.slot_mapping.cpu.clone()
        bt0.compute_slot_mapping(num_reqs, query_start_loc, positions)
        bt1.compute_slot_mapping(num_reqs, query_start_loc, positions)
        expected0 = bt0.slot_mapping.cpu.clone()
        expected1 = bt1.slot_mapping.cpu.clone()

        # Fused: one kernel launch for both groups.
        launch_slot_mapping_fused([bt0, bt1], num_reqs, query_start_loc, positions)
        torch.npu.synchronize()

        num_tokens = positions.shape[0]
        actual0 = bt0.slot_mapping.cpu[:num_tokens]
        actual1 = bt1.slot_mapping.cpu[:num_tokens]

        self.assertTrue(torch.equal(actual0, expected0[:num_tokens]), "group 0 mismatch")
        self.assertTrue(torch.equal(actual1, expected1[:num_tokens]), "group 1 mismatch")

        # Sanity check: unmapped region must still be untouched.
        self.assertTrue(torch.equal(bt0.slot_mapping.cpu, ref0))
        self.assertTrue(torch.equal(bt1.slot_mapping.cpu, ref1))

    def test_single_group_fallback(self):
        """With one group, launch_slot_mapping_fused delegates to compute_slot_mapping."""
        bt0, _ = self.setup_block_tables()
        num_reqs, query_start_loc, positions = self.build_inputs()

        expected = bt0.slot_mapping.cpu.clone()
        bt0.compute_slot_mapping(num_reqs, query_start_loc, positions)
        expected = bt0.slot_mapping.cpu.clone()
        bt0.slot_mapping.cpu.fill_(0)

        launch_slot_mapping_fused([bt0], num_reqs, query_start_loc, positions)
        torch.npu.synchronize()

        num_tokens = positions.shape[0]
        self.assertTrue(torch.equal(bt0.slot_mapping.cpu[:num_tokens], expected[:num_tokens]))

    def test_padding_uses_pad_slot_id(self):
        """Tokens beyond the allocated blocks must be padded with PAD_SLOT_ID."""
        bt0 = self.create_block_table()
        # Single request owning a single block (128 tokens).
        bt0.add_row([0], 0)
        num_reqs = 1
        query_start_loc = torch.tensor([0, 4], dtype=torch.int32, device=self.device)
        # Positions 0..3 map into block 0; positions 4..7 exceed it.
        positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64, device=self.device)

        launch_slot_mapping_fused([bt0], num_reqs, query_start_loc, positions)
        torch.npu.synchronize()

        actual = bt0.slot_mapping.cpu[:8]
        self.assertEqual(actual[0].item(), 0)
        self.assertEqual(actual[1].item(), 1)
        self.assertEqual(actual[2].item(), 2)
        self.assertEqual(actual[3].item(), 3)
        self.assertTrue(torch.all(actual[4:] == PAD_SLOT_ID))


if __name__ == "__main__":
    unittest.main()
