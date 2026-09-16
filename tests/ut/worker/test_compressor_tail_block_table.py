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
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

from tests.ut.base import TestBase


def _make_compressor_tail_spec(block_size=8, tail_tokens=8, compress_ratio=4):
    from vllm_ascend.core.kv_cache_interface import AscendCompressorTailSpec

    return AscendCompressorTailSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=2048,
        dtype=torch.float32,
        sliding_window=tail_tokens,
        compress_ratio=compress_ratio,
        model_version="deepseek_v4",
        tail_tokens=tail_tokens,
        ring_blocks_per_request=(tail_tokens + block_size - 1) // block_size,
        state_dim=2048,
    )


class TestCompressorTailBlockTable(TestBase):
    """Regression tests for the compressor-tail slot-mapping out-of-bounds bug.

    The compressor-tail block table keeps only ``ring_blocks_per_request``
    columns per row, but the generic slot-mapping path indexes rows by
    absolute position (``pos // block_size``) without ring modulo. Any
    position beyond the tail therefore reads outside the request's own row.
    Tail groups must be skipped like mamba groups: the compressor operator
    addresses the ring internally via ``compressor_metadata``.
    """

    def setUp(self):
        self.max_num_reqs = 4
        self.max_num_batched_tokens = 512
        self.device = torch.device("cpu")

    def _create_multi_group_block_table(self, block_sizes, max_num_blocks, kernel_sizes, kv_cache_groups):
        with patch("vllm_ascend.worker.block_table.get_dcp_group") as mock_get_dcp_group:
            mock_dcp_group = MagicMock(spec=GroupCoordinator)
            mock_dcp_group.world_size = 1
            mock_dcp_group.rank_in_group = 0
            mock_get_dcp_group.return_value = mock_dcp_group

            from vllm_ascend.worker.block_table import MultiGroupBlockTable

            return MultiGroupBlockTable(
                max_num_reqs=self.max_num_reqs,
                max_model_len=4096,
                max_num_batched_tokens=self.max_num_batched_tokens,
                pin_memory=False,
                device=self.device,
                block_sizes=block_sizes,
                max_num_blocks=max_num_blocks,
                kernel_sizes=kernel_sizes,
                cp_kv_cache_interleave_size=1,
                kv_cache_groups=kv_cache_groups,
            )

    def test_tail_group_detected_without_row_shrink(self):
        """Tail group keeps ring-sized rows (no compress_ratio shrink)."""
        tail_spec = _make_compressor_tail_spec(block_size=8, tail_tokens=8)
        multi = self._create_multi_group_block_table(
            block_sizes=[8],
            max_num_blocks=[1],
            kernel_sizes=[[8]],
            kv_cache_groups=[KVCacheGroupSpec(["layer0"], tail_spec)],
        )
        tail_table = multi.block_tables[0]
        self.assertTrue(tail_table.is_compressor_tail_group)
        self.assertFalse(tail_table.is_mamba_group)
        self.assertEqual(tail_table.max_num_blocks_per_req, 1)
        self.assertEqual(tail_table.block_table.np.shape, (self.max_num_reqs, 1))

    def test_tail_group_detected_through_uniform_type_wrapper(self):
        """Runtime packs DSV4 groups in UniformTypeKVCacheSpecs; detection must unwrap."""
        tail_spec = _make_compressor_tail_spec(block_size=32, tail_tokens=128, compress_ratio=128)
        uniform_spec = UniformTypeKVCacheSpecs(kv_cache_specs={"layer0": tail_spec})
        multi = self._create_multi_group_block_table(
            block_sizes=[32],
            max_num_blocks=[4],
            kernel_sizes=[[32]],
            kv_cache_groups=[KVCacheGroupSpec(["layer0"], uniform_spec)],
        )
        tail_table = multi.block_tables[0]
        self.assertTrue(tail_table.is_compressor_tail_group)
        self.assertEqual(tail_table.max_num_blocks_per_req, 4)

    def test_compute_slot_mapping_skips_tail_group(self):
        """Generic slot mapping must not run for tail groups.

        Before the fix this either raised (CPU: no usable triton backend) or
        silently computed out-of-row slots (NPU). After the fix the tail
        group's slot mapping must remain untouched.
        """
        tail_spec = _make_compressor_tail_spec(block_size=8, tail_tokens=8)
        multi = self._create_multi_group_block_table(
            block_sizes=[8],
            max_num_blocks=[1],
            kernel_sizes=[[8]],
            kv_cache_groups=[KVCacheGroupSpec(["layer0"], tail_spec)],
        )
        multi.add_row(([11],), 0)
        multi.add_row(([13],), 1)

        before = multi.block_tables[0].slot_mapping.np.copy()
        query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
        # Positions 8..11 exceed the 8-token tail ring: the buggy kernel
        # indexed block_table rows at pos // 8 >= 1 (out of row).
        positions = torch.tensor([8, 9, 10, 11], dtype=torch.int64)

        multi.compute_slot_mapping(2, query_start_loc, positions)

        np.testing.assert_array_equal(multi.block_tables[0].slot_mapping.np, before)

    def test_compute_slot_mapping_draft_skips_tail_group_but_not_normal_group(self):
        """Draft path: normal group still maps, tail group is skipped.

        With ring=1 block per row and max_num_reqs=4, request row 3 at
        absolute position 8 made the buggy numpy draft path read flat index
        3 * 1 + 8 // 8 = 4 in a 4-element table -> IndexError (or a silent
        cross-row read for smaller indices). After the fix only the normal
        group consumes the mapping.
        """
        tail_spec = _make_compressor_tail_spec(block_size=8, tail_tokens=8)
        multi = self._create_multi_group_block_table(
            block_sizes=[128, 8],
            max_num_blocks=[4, 1],
            kernel_sizes=[[128], [8]],
            kv_cache_groups=[None, KVCacheGroupSpec(["layer0"], tail_spec)],
        )
        normal_table, tail_table = multi.block_tables
        normal_table.add_row([5], 0)
        normal_table.add_row([9], 3)
        tail_table.add_row([11], 0)
        tail_table.add_row([13], 3)

        tail_before = tail_table.slot_mapping.np.copy()
        req_indices = np.array([0, 3], dtype=np.int64)
        positions = np.array([0, 8], dtype=np.int64)

        multi.compute_slot_mapping_draft(req_indices, positions)

        # Normal group: slot = block_id * block_size + pos % block_size.
        np.testing.assert_array_equal(
            normal_table.slot_mapping.np[:2],
            np.array([5 * 128 + 0, 9 * 128 + 8], dtype=np.int32),
        )
        # Tail group: untouched.
        np.testing.assert_array_equal(tail_table.slot_mapping.np, tail_before)


if __name__ == "__main__":
    unittest.main()
