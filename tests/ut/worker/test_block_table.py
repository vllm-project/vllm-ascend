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
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.base import TestBase


class TestBlockTableComputeSlotMapping(TestBase):
    """Test suite for BlockTable.compute_slot_mapping() method
    
    This test suite covers different configurations of DCP (Decode Context Parallelism),
    PCP (Prefill Context Parallelism), and cp_kv_cache_interleave_size to ensure
    correct slot_mapping calculation on different ranks.
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

    def create_block_table(self, dcp_world_size, dcp_rank, pcp_world_size,
                           pcp_rank, cp_kv_cache_interleave_size):
        """Helper method to create BlockTable with mocked distributed groups"""

        with patch('vllm_ascend.worker.block_table.get_dcp_group') as mock_get_dcp_group, \
             patch('vllm_ascend.worker.block_table.get_pcp_group') as mock_get_pcp_group:

            # Mock DCP group
            mock_dcp_group = MagicMock(spec=GroupCoordinator)
            mock_dcp_group.world_size = dcp_world_size
            mock_dcp_group.rank_in_group = dcp_rank
            mock_get_dcp_group.return_value = mock_dcp_group

            # Mock PCP group
            mock_pcp_group = MagicMock(spec=GroupCoordinator)
            mock_pcp_group.world_size = pcp_world_size
            mock_pcp_group.rank_in_group = pcp_rank
            mock_get_pcp_group.return_value = mock_pcp_group

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
                num_speculative_tokens=0)

            return block_table

    def setup_block_table_data(self, block_table, num_reqs=2):
        """Helper method to populate block table with test data"""
        # Add block IDs for each request
        for i in range(num_reqs):
            block_ids = list(range(i * 4,
                                   (i + 1) * 4))  # [0,1,2,3], [4,5,6,7], etc.
            block_table.add_row(block_ids, i)

    def _test_slot_mapping_for_ranks(self, dcp_world_size, pcp_world_size,
                                     cp_kv_cache_interleave_size,
                                     test_configs):
        """Helper method to test slot_mapping across multiple ranks
        
        Args:
            dcp_world_size: Number of DCP ranks
            pcp_world_size: Number of PCP ranks
            cp_kv_cache_interleave_size: Interleave size for KV cache
            test_configs: List of tuples (dcp_rank, pcp_rank, req_indices, positions, expected_result)
        """
        for dcp_rank, pcp_rank, req_indices, positions, expected_result in test_configs:
            with self.subTest(dcp_rank=dcp_rank, pcp_rank=pcp_rank):
                block_table = self.create_block_table(
                    dcp_world_size, dcp_rank, pcp_world_size, pcp_rank,
                    cp_kv_cache_interleave_size)

                num_reqs = max(req_indices) + 1 if len(req_indices) > 0 else 1
                self.setup_block_table_data(block_table, num_reqs=num_reqs)

                block_table.compute_slot_mapping(req_indices, positions)

                actual_result = block_table.slot_mapping.np[:len(positions)]
                np.testing.assert_array_equal(
                    actual_result, expected_result,
                    f"DCP={dcp_world_size}, PCP={pcp_world_size}, "
                    f"interleave={cp_kv_cache_interleave_size}, "
                    f"dcp_rank={dcp_rank}, pcp_rank={pcp_rank}")

    def test_compute_slot_mapping_dcp1_pcp1_interleave1(self):
        """Test compute_slot_mapping with DCP=1, PCP=1, interleave_size=1
        
        With no parallelism (DCP=1, PCP=1), all tokens are local to the single rank.
        
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
            (0, 0, req_indices, positions, expected_result),
        ]

        self._test_slot_mapping_for_ranks(dcp_world_size=1,
                                          pcp_world_size=1,
                                          cp_kv_cache_interleave_size=1,
                                          test_configs=test_configs)

    def test_compute_slot_mapping_dcp4_pcp2_interleave1(self):
        """Test compute_slot_mapping with DCP=4, PCP=2, interleave_size=1
        
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
        # Rank assignment: current_rank = 4 * pcp_rank + dcp_rank
        test_configs = []

        # For each rank, specify which positions it owns and their local slot mapping
        rank_expectations = {
            # Rank 0 (pcp=0, dcp=0): positions 0, 8
            0: [0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1],
            # Rank 1 (pcp=0, dcp=1): positions 1, 9
            1: [-1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1],
            # Rank 2 (pcp=0, dcp=2): positions 2, 10
            2: [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1],
            # Rank 3 (pcp=0, dcp=3): positions 3, 11
            3: [-1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1],
            # Rank 4 (pcp=1, dcp=0): positions 4, 12
            4: [-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1],
            # Rank 5 (pcp=1, dcp=1): positions 5, 13
            5: [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1],
            # Rank 6 (pcp=1, dcp=2): positions 6, 14
            6: [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1],
            # Rank 7 (pcp=1, dcp=3): positions 7, 15
            7: [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1],
        }

        for pcp_rank in range(2):
            for dcp_rank in range(4):
                current_rank = 4 * pcp_rank + dcp_rank
                expected_result = np.array(rank_expectations[current_rank],
                                           dtype=np.int32)
                test_configs.append((dcp_rank, pcp_rank, req_indices,
                                     positions, expected_result))

        self._test_slot_mapping_for_ranks(dcp_world_size=4,
                                          pcp_world_size=2,
                                          cp_kv_cache_interleave_size=1,
                                          test_configs=test_configs)

    def test_compute_slot_mapping_dcp4_pcp2_interleave128(self):
        """Test compute_slot_mapping with DCP=4, PCP=2, interleave_size=128
        
        With interleave_size=128, tokens are distributed in chunks of 128 across ranks.
        Virtual block size = 16 * 4 * 2 = 128
        
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
        for pcp_rank in range(2):
            for dcp_rank in range(4):
                current_rank = 4 * pcp_rank + dcp_rank
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

                test_configs.append(
                    (dcp_rank, pcp_rank, req_indices, positions,
                     np.array(expected_result, dtype=np.int32)))

        self._test_slot_mapping_for_ranks(dcp_world_size=4,
                                          pcp_world_size=2,
                                          cp_kv_cache_interleave_size=128,
                                          test_configs=test_configs)


if __name__ == '__main__':
    unittest.main()
