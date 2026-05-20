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


class TestBlockTable310(TestBase):
    def setUp(self):
        self.block_size = 128
        self.max_num_reqs = 4
        self.max_num_blocks_per_req = 128
        self.max_num_batched_tokens = 512
        self.pin_memory = False
        self.device = torch.device("cpu")
        self.kernel_sizes = [128]

    def _create_block_table(self, dcp_world_size, dcp_rank, pcp_world_size, pcp_rank, cp_kv_cache_interleave_size):
        with (
            patch("vllm_ascend.worker.block_table.get_dcp_group") as mock_get_dcp_group,
            patch("vllm_ascend.worker.block_table.get_pcp_group") as mock_get_pcp_group,
        ):
            mock_dcp_group = MagicMock(spec=GroupCoordinator)
            mock_dcp_group.world_size = dcp_world_size
            mock_dcp_group.rank_in_group = dcp_rank
            mock_get_dcp_group.return_value = mock_dcp_group

            mock_pcp_group = MagicMock(spec=GroupCoordinator)
            mock_pcp_group.world_size = pcp_world_size
            mock_pcp_group.rank_in_group = pcp_rank
            mock_get_pcp_group.return_value = mock_pcp_group

            from vllm_ascend._310p.block_table import BlockTable

            return BlockTable(
                block_size=self.block_size,
                max_num_reqs=self.max_num_reqs,
                max_num_blocks_per_req=self.max_num_blocks_per_req,
                max_num_batched_tokens=self.max_num_batched_tokens,
                pin_memory=self.pin_memory,
                device=self.device,
                kernel_sizes=self.kernel_sizes,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                num_speculative_tokens=0,
            )

    @staticmethod
    def _setup_block_table_data(block_table, num_reqs=2):
        for i in range(num_reqs):
            block_ids = list(range(i * 4, (i + 1) * 4))
            block_table.add_row(block_ids, i)

    def test_compute_slot_mapping_with_query_start_loc_signature(self):
        block_table = self._create_block_table(
            dcp_world_size=1,
            dcp_rank=0,
            pcp_world_size=1,
            pcp_rank=0,
            cp_kv_cache_interleave_size=1,
        )
        self._setup_block_table_data(block_table, num_reqs=2)

        query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
        positions = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

        block_table.compute_slot_mapping(2, query_start_loc, positions)

        expected = np.array([0, 1, 512, 513], dtype=np.int32)
        np.testing.assert_array_equal(block_table.slot_mapping.np[:4], expected)
        np.testing.assert_array_equal(block_table.slot_mapping.gpu[:4].cpu().numpy(), expected)

    def test_compute_slot_mapping_with_req_indices_signature(self):
        block_table = self._create_block_table(
            dcp_world_size=4,
            dcp_rank=0,
            pcp_world_size=2,
            pcp_rank=0,
            cp_kv_cache_interleave_size=1,
        )
        self._setup_block_table_data(block_table, num_reqs=1)

        req_indices = np.zeros(16, dtype=np.int32)
        positions = np.arange(16, dtype=np.int32)

        block_table.compute_slot_mapping(req_indices, positions)

        expected = np.array([0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1], dtype=np.int32)
        np.testing.assert_array_equal(block_table.slot_mapping.np[:16], expected)
        np.testing.assert_array_equal(block_table.slot_mapping.gpu[:16].cpu().numpy(), expected)


if __name__ == "__main__":
    unittest.main()
