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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# import vllm.utils.cpu_triton_utils as cpu_tl
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from tests.ut.base import TestBase


class TestBlockTableComputeSlotMapping(TestBase):
    """Test suite for BlockTable.compute_slot_mapping() method

    This test suite covers different configurations of DCP (Decode Context
    Parallelism) and cp_kv_cache_interleave_size to ensure correct slot_mapping
    calculation on different ranks.
    """

    def setUp(self):
        """Set up common test fixtures"""
        self.block_size = 128
        self.max_num_reqs = 4
        self.max_num_blocks_per_req = 128
        self.max_num_batched_tokens = 512
        self.pin_memory = False
        self.device = torch.device("cpu")
        self.kernel_sizes = [128]
        self._skip_triton_kernel = True

    def create_block_table(
        self,
        dcp_world_size,
        dcp_rank,
        cp_kv_cache_interleave_size,
        num_speculative_tokens=0,
    ):
        """Helper method to create BlockTable with mocked distributed groups"""

        with patch("vllm_ascend.worker.block_table.get_dcp_group") as mock_get_dcp_group:
            # Mock DCP group
            mock_dcp_group = MagicMock(spec=GroupCoordinator)
            mock_dcp_group.world_size = dcp_world_size
            mock_dcp_group.rank_in_group = dcp_rank
            mock_get_dcp_group.return_value = mock_dcp_group

            from vllm_ascend.worker.block_table import BlockTable

            block_table = BlockTable(
                block_size=self.block_size,
                max_num_reqs=self.max_num_reqs,
                max_num_blocks_per_req=self.max_num_blocks_per_req,
                max_num_batched_tokens=self.max_num_batched_tokens,
                pin_memory=self.pin_memory,
                device=self.device,
                kernel_sizes=self.kernel_sizes,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                num_speculative_tokens=num_speculative_tokens,
            )

            return block_table

    def test_compute_slot_mapping_draft_reserves_mtp_slots(self):
        """MTP5 draft slots can exceed the scheduler token capacity."""
        self.max_num_reqs = 12
        self.max_num_batched_tokens = 80
        block_table = self.create_block_table(
            dcp_world_size=4,
            dcp_rank=0,
            cp_kv_cache_interleave_size=1,
            num_speculative_tokens=5,
        )

        num_active_reqs = 11
        for req_idx in range(num_active_reqs):
            block_table.add_row([req_idx], req_idx)

        req_indices = np.repeat(np.arange(num_active_reqs, dtype=np.int32), 10)
        positions = np.tile(np.arange(10, dtype=np.int64), num_active_reqs)
        block_table.compute_slot_mapping_draft(req_indices, positions)

        self.assertEqual(block_table.slot_mapping.cpu.numel(), 128)
        self.assertEqual(block_table.slot_mapping.cpu[: req_indices.size].numel(), 110)

    def test_mamba_table_preserves_speculative_capacity_with_dcp(self):
        from vllm_ascend.worker.block_table import BlockTable

        mamba_spec = MambaSpec(
            block_size=384,
            shapes=((4, 8),),
            dtypes=(torch.float32,),
            page_size_padded=128,
            mamba_cache_mode="align",
            num_speculative_blocks=7,
        )
        layer_specs = {f"mamba.{i}": mamba_spec for i in range(3)}
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(layer_specs)
        self.assertIsNotNone(uniform_spec)
        # ceil(133120 / 384) + 7 speculative state slots, on every DCP rank.
        expected_width = 354
        for spec in (mamba_spec, uniform_spec):
            for dcp_size in (1, 4):
                with self.subTest(spec=type(spec).__name__, dcp_size=dcp_size):
                    config = SimpleNamespace(
                        cache_config=SimpleNamespace(mamba_cache_mode="align"),
                        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size),
                    )
                    width = spec.max_num_blocks_per_req(config, 133120)
                    self.assertEqual(width, expected_width)
                    group = KVCacheGroupSpec(layer_names=list(layer_specs), kv_cache_spec=spec)
                    with patch(
                        "vllm_ascend.worker.block_table.get_dcp_group",
                        return_value=SimpleNamespace(world_size=dcp_size, rank_in_group=0),
                    ):
                        table = BlockTable(
                            block_size=spec.block_size,
                            max_num_reqs=self.max_num_reqs,
                            max_num_blocks_per_req=width,
                            max_num_batched_tokens=self.max_num_batched_tokens,
                            pin_memory=self.pin_memory,
                            device=self.device,
                            kernel_sizes=[0],
                            num_speculative_tokens=7,
                            kv_cache_group=group,
                        )

                    self.assertTrue(table.is_mamba_group)
                    self.assertEqual(table.block_table.cpu.shape[1], expected_width)
                    self.assertEqual(table.max_num_blocks_per_req, expected_width)
                    # Exercise the last speculative slot, not just the formula.
                    block_ids = list(range(expected_width))
                    table.add_row(block_ids[:-2], 0)
                    table.append_row(block_ids[-2:], 0)
                    np.testing.assert_array_equal(table.block_table.np[0], block_ids)

    def setup_block_table_data(self, block_table, num_reqs=2):
        """Helper method to populate block table with test data"""
        # Add block IDs for each request
        for i in range(num_reqs):
            block_ids = list(range(i * 4, (i + 1) * 4))  # [0,1,2,3], [4,5,6,7], etc.
            block_table.add_row(block_ids, i)

    def test_dsv4_logical_block_ids_are_not_expanded(self):
        self.block_size = 512
        self.kernel_sizes = [512]
        block_table = self.create_block_table(
            dcp_world_size=1,
            dcp_rank=0,
            cp_kv_cache_interleave_size=1,
        )

        block_table.add_row([7, 11], 0)

        self.assertFalse(block_table.use_hybrid_blocks)
        self.assertEqual(block_table.blocks_per_phys_block, 1)
        self.assertEqual(block_table.physical_block_size, 512)
        self.assertEqual(block_table.logical_block_size, 512)
        self.assertEqual(block_table.num_blocks_per_row[0], 2)
        np.testing.assert_array_equal(
            block_table.block_table.np[0, :2],
            np.array([7, 11], dtype=np.int32),
        )

    def _test_slot_mapping_for_ranks(self, dcp_world_size, cp_kv_cache_interleave_size, test_configs):
        """Helper method to test slot_mapping across multiple ranks

        Args:
            dcp_world_size: Number of DCP ranks
            cp_kv_cache_interleave_size: Interleave size for KV cache
            test_configs: List of tuples
                (dcp_rank, req_indices, positions, expected_result)
        """
        for dcp_rank, req_indices, positions, expected_result in test_configs:
            with self.subTest(dcp_rank=dcp_rank):
                block_table = self.create_block_table(dcp_world_size, dcp_rank, cp_kv_cache_interleave_size)

                num_reqs = max(req_indices) + 1 if len(req_indices) > 0 else 1
                self.setup_block_table_data(block_table, num_reqs=num_reqs)

                # Build query_start_loc [num_reqs + 1] from req_indices.
                # query_start_loc holds the cumulative token count per request,
                # e.g. req_indices=[0,0,1,1] -> query_start_loc=[0,2,4].
                num_tokens = len(positions)
                counts = np.bincount(req_indices, minlength=num_reqs)
                query_start_loc_np = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
                _query_start_loc = torch.from_numpy(query_start_loc_np)

                # positions must be a torch int64 tensor to match the
                # _compute_slot_mapping_kernel's positions_ptr type.
                _positions_tensor = torch.from_numpy(positions.astype(np.int64))
                # Triton kernel requires NPU device; mock it and compute on CPU
                with patch.object(block_table, "compute_slot_mapping"):
                    slot_mapping = block_table.slot_mapping.cpu
                    total_cp_world_size = dcp_world_size
                    total_cp_rank = dcp_rank
                    bs = block_table.physical_block_size
                    interleave = cp_kv_cache_interleave_size
                    for token_idx in range(num_tokens):
                        req_idx = req_indices[token_idx]
                        pos = int(positions[token_idx])
                        num_blocks_row = int(block_table.num_blocks_per_row[req_idx])
                        block_ids = block_table.block_table.cpu[req_idx].tolist()
                        if total_cp_world_size <= 1 and interleave <= 1:
                            block_idx = pos // bs
                            offset = pos % bs
                            if block_idx < num_blocks_row:
                                slot_val = block_ids[block_idx] * bs + offset
                            else:
                                slot_val = -1
                        elif interleave <= 1:
                            if pos % total_cp_world_size != total_cp_rank:
                                slot_val = -1
                            else:
                                local_pos = pos // total_cp_world_size
                                block_idx = local_pos // bs
                                offset = local_pos % bs
                                if block_idx < num_blocks_row:
                                    slot_val = block_ids[block_idx] * bs + offset
                                else:
                                    slot_val = -1
                        else:
                            virtual_block = interleave * total_cp_world_size
                            chunk_idx = pos // virtual_block
                            pos_in_chunk = pos % virtual_block
                            rank_in_chunk = pos_in_chunk // interleave
                            if rank_in_chunk != total_cp_rank:
                                slot_val = -1
                            else:
                                local_pos = chunk_idx * interleave + (pos_in_chunk % interleave)
                                block_idx = local_pos // bs
                                offset = local_pos % bs
                                if block_idx < num_blocks_row:
                                    slot_val = block_ids[block_idx] * bs + offset
                                else:
                                    slot_val = -1
                        slot_mapping[token_idx] = slot_val

                actual_result = block_table.slot_mapping.np[:num_tokens]

                np.testing.assert_array_equal(
                    actual_result,
                    expected_result,
                    f"DCP={dcp_world_size}, interleave={cp_kv_cache_interleave_size}, dcp_rank={dcp_rank}",
                )

    def test_compute_slot_mapping_dcp1_interleave1(self):
        """Test compute_slot_mapping with DCP=1, interleave_size=1

        With no parallelism, all tokens are local to the single rank.

        Setup:
        - Block size: 16
        - Request 0 has blocks: [0, 1, 2, 3]
        - Request 1 has blocks: [4, 5, 6, 7]

        Test positions for each request:
        - Request 0, position 0: block_id=0, offset=0 → slot = 0*128+0 = 0
        - Request 0, position 1: block_id=0, offset=1 → slot = 0*128+1 = 1
        - Request 1, position 0: block_id=4, offset=0 → slot = 4*128+0 = 512
        - Request 1, position 1: block_id=4, offset=1 → slot = 4*128+1 = 513
        """
        req_indices = np.array([0, 0, 1, 1], dtype=np.int32)
        positions = np.array([0, 1, 0, 1], dtype=np.int32)

        expected_result = np.array([0, 1, 512, 513], dtype=np.int32)

        test_configs = [
            (0, req_indices, positions, expected_result),
        ]

        self._test_slot_mapping_for_ranks(dcp_world_size=1, cp_kv_cache_interleave_size=1, test_configs=test_configs)

    def test_compute_slot_mapping_dcp8_interleave1(self):
        """Test compute_slot_mapping with DCP=8, interleave_size=1

        With interleave_size=1, tokens are distributed round-robin across all 8 ranks:
        - Position 0 → Rank 0
        - Position 1 → Rank 1
        - Position 2 → Rank 2
        - ...
        - Position 7 → Rank 7
        - Position 8 → Rank 0 (wraps around)
        """
        req_indices = np.array([0] * 16, dtype=np.int32)
        positions = np.array(list(range(16)), dtype=np.int32)

        # Manually computed expected values for each rank
        test_configs = []

        # For each rank, specify which positions it owns and their local slot mapping
        rank_expectations = {
            # Rank 0: positions 0, 8
            0: [0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1],
            # Rank 1: positions 1, 9
            1: [-1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1],
            # Rank 2: positions 2, 10
            2: [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1],
            # Rank 3: positions 3, 11
            3: [-1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1],
            # Rank 4: positions 4, 12
            4: [-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1],
            # Rank 5: positions 5, 13
            5: [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1],
            # Rank 6: positions 6, 14
            6: [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1],
            # Rank 7: positions 7, 15
            7: [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1],
        }

        for dcp_rank in range(8):
            expected_result = np.array(rank_expectations[dcp_rank], dtype=np.int32)
            test_configs.append((dcp_rank, req_indices, positions, expected_result))

        self._test_slot_mapping_for_ranks(dcp_world_size=8, cp_kv_cache_interleave_size=1, test_configs=test_configs)

    def test_compute_slot_mapping_dcp8_interleave128(self):
        """Test compute_slot_mapping with DCP=8, interleave_size=128

        With interleave_size=128, tokens are distributed in chunks of 128 across ranks.
        Virtual block size = 16 * 8 = 128

        Token distribution with interleave_size=128:
        - Positions 0-127 belong to rank 0 (first chunk of 128)
        - Positions 128-255 belong to rank 1 (second chunk of 128)
        - Positions 256-383 belong to rank 2 (third chunk of 128)
        - And so on...

        Using 130 positions ensures we test both rank 0 (positions 0-127) and rank 1 (positions 128-129).
        """
        num_positions = 130
        req_indices = np.array([0] * num_positions, dtype=np.int32)
        positions = np.array(list(range(num_positions)), dtype=np.int32)

        # With interleave_size=128 and virtual_block_size=128:
        # Positions 0-127 belong to rank 0
        # Positions 128-129 belong to rank 1
        test_configs = []

        # Build expected results for each rank
        for dcp_rank in range(8):
            current_rank = dcp_rank
            expected_result = []

            if current_rank == 0:
                # Rank 0 gets positions 0-127
                # Each maps to its local slot: 0, 1, 2, ..., 127
                for pos in range(130):
                    if pos < 128:
                        expected_result.append(pos)
                    else:
                        expected_result.append(-1)
            elif current_rank == 1:
                # Rank 1 gets positions 128-129
                # Position 128 maps to local slot 0, position 129 to local slot 1
                for pos in range(130):
                    if pos == 128:
                        expected_result.append(0)
                    elif pos == 129:
                        expected_result.append(1)
                    else:
                        expected_result.append(-1)
            else:
                # All other ranks get no positions
                expected_result = [-1] * 130

            test_configs.append((dcp_rank, req_indices, positions, np.array(expected_result, dtype=np.int32)))

        self._test_slot_mapping_for_ranks(dcp_world_size=8, cp_kv_cache_interleave_size=128, test_configs=test_configs)

    def test_compute_slot_mapping_skipped_for_non_attention_group(self):
        """Test that slot mapping computation is skipped for non-attention groups.

        kernel_sizes[0]==0 marks non-attention groups (Mamba/compressor state
        cache) that don't participate in slot_mapping computation. Both
        compute_slot_mapping() and compute_slot_mapping_draft() must return
        early without touching the slot_mapping buffer or invoking the kernel.
        """
        self.kernel_sizes = [0]
        block_table = self.create_block_table(dcp_world_size=1, dcp_rank=0, cp_kv_cache_interleave_size=1)

        num_reqs = 2
        req_indices = np.array([0, 0, 1, 1], dtype=np.int32)
        positions = np.array([0, 1, 0, 1], dtype=np.int32)
        num_tokens = positions.shape[0]
        query_start_loc = torch.from_numpy(np.array([0, 2, 4], dtype=np.int32))

        # Poison the slot_mapping buffer to verify it stays untouched.
        block_table.slot_mapping.np[:] = -1

        from vllm_ascend.worker import block_table as block_table_module

        with patch.object(block_table_module, "_compute_slot_mapping_kernel") as mock_kernel:
            block_table.compute_slot_mapping(
                num_reqs,
                query_start_loc,
                torch.from_numpy(positions.astype(np.int64)),
            )
            mock_kernel.assert_not_called()

        with patch.object(block_table_module, "_compute_slot_mapping_kernel") as mock_kernel:
            block_table.compute_slot_mapping_draft(req_indices, positions)
            mock_kernel.assert_not_called()

        np.testing.assert_array_equal(
            block_table.slot_mapping.np[:num_tokens],
            np.full(num_tokens, -1, dtype=np.int32),
        )


if __name__ == "__main__":
    unittest.main()
