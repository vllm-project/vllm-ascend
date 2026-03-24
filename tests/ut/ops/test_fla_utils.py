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

import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.triton.fla import utils as fla_utils


class TestFlaUtils(TestBase):

    def setUp(self):
        fla_utils._PREPARE_CACHE.clear()

    def test_prepare_chunk_metadata_matches_expected(self):
        cu_seqlens = torch.tensor([0, 65, 129, 190], dtype=torch.long)
        chunk_size = 64

        expected_num_chunks = torch.tensor([2, 1, 1], dtype=torch.long)
        expected_chunk_indices = torch.tensor(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [2, 0],
            ],
            dtype=torch.long,
        )
        expected_chunk_offsets = torch.tensor([0, 2, 3, 4], dtype=torch.long)
        expected_update_offsets = torch.tensor([0, 3, 5, 7], dtype=torch.long)
        expected_final_chunk_indices = torch.tensor([1, 2, 3], dtype=torch.long)

        self.assertTrue(torch.equal(fla_utils.prepare_num_chunks(cu_seqlens, chunk_size), expected_num_chunks))
        self.assertEqual(fla_utils.prepare_num_total_chunks(cu_seqlens, chunk_size), 4)
        self.assertTrue(torch.equal(fla_utils.prepare_chunk_indices(cu_seqlens, chunk_size), expected_chunk_indices))
        self.assertTrue(torch.equal(fla_utils.prepare_chunk_offsets(cu_seqlens, chunk_size), expected_chunk_offsets))
        self.assertTrue(
            torch.equal(fla_utils.prepare_update_chunk_offsets(cu_seqlens, chunk_size), expected_update_offsets)
        )
        self.assertTrue(
            torch.equal(fla_utils.prepare_final_chunk_indices(cu_seqlens, chunk_size), expected_final_chunk_indices)
        )

    def test_prepare_chunk_metadata_cache_reuses_and_invalidates_after_update(self):
        cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.long)
        chunk_size = 64

        first_chunk_indices = fla_utils.prepare_chunk_indices(cu_seqlens, chunk_size)
        first_chunk_offsets = fla_utils.prepare_chunk_offsets(cu_seqlens, chunk_size)
        first_final_indices = fla_utils.prepare_final_chunk_indices(cu_seqlens, chunk_size)

        self.assertIs(first_chunk_indices, fla_utils.prepare_chunk_indices(cu_seqlens, chunk_size))
        self.assertIs(first_chunk_offsets, fla_utils.prepare_chunk_offsets(cu_seqlens, chunk_size))
        self.assertIs(first_final_indices, fla_utils.prepare_final_chunk_indices(cu_seqlens, chunk_size))

        cu_seqlens[1] = 96

        updated_chunk_indices = fla_utils.prepare_chunk_indices(cu_seqlens, chunk_size)
        updated_chunk_offsets = fla_utils.prepare_chunk_offsets(cu_seqlens, chunk_size)
        updated_final_indices = fla_utils.prepare_final_chunk_indices(cu_seqlens, chunk_size)

        self.assertIsNot(first_chunk_indices, updated_chunk_indices)
        self.assertIsNot(first_chunk_offsets, updated_chunk_offsets)
        self.assertIsNot(first_final_indices, updated_final_indices)

        self.assertTrue(
            torch.equal(
                updated_chunk_indices,
                torch.tensor(
                    [
                        [0, 0],
                        [0, 1],
                        [1, 0],
                    ],
                    dtype=torch.long,
                ),
            )
        )
        self.assertTrue(torch.equal(updated_chunk_offsets, torch.tensor([0, 2, 3], dtype=torch.long)))
        self.assertTrue(torch.equal(updated_final_indices, torch.tensor([1, 2], dtype=torch.long)))
