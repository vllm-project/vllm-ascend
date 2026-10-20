# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_read import (
    DirectPrefixSource,
    materialize_direct_prefix,
)


class TestCanonicalRead(unittest.TestCase):
    def setUp(self):
        self.table = torch.arange(24, dtype=torch.int32).reshape(3, 8)
        self.owner = SimpleNamespace(
            block_tables=[SimpleNamespace(gpu=self.table)],
            num_blocks=SimpleNamespace(np=np.array([[8, 8, 8]])),
            kernel_block_sizes=[4],
        )
        self.mapping = np.array([2, 0])

    def read(self, ends=(21, 11), starts=(12, 4), upper=(24, 15)):
        exact = torch.tensor(ends, dtype=torch.int32)
        return materialize_direct_prefix(
            DirectPrefixSource.from_block_tables(self.owner, 0, self.mapping),
            exact_cpu=exact,
            exact_device=exact,
            upper_cpu=torch.tensor(upper),
            starts_cpu=torch.tensor(starts),
            padded_rows=3,
        )

    def test_canonical_projection_preserves_upper_and_padding(self):
        view = self.read()
        self.assertEqual(view.block_table.tolist(), [[19, 20, 21], [1, 2, 0], [0, 0, 0]])
        self.assertEqual(view.seq_lens_cpu.tolist(), [9, 7, 0])
        self.assertEqual(view.seq_lens_cpu_upper_bound.tolist(), [12, 11, 0])
        self.assertNotEqual(view.block_table.untyped_storage().data_ptr(), self.table.untyped_storage().data_ptr())
        self.assertEqual(self.table.tolist(), torch.arange(24).reshape(3, 8).tolist())

    def test_reorder_in_place_and_same_row_replacement(self):
        first = self.read()
        self.mapping[:] = [0, 2]
        self.table[0] = torch.arange(100, 108)
        second = self.read()
        self.assertEqual(second.block_table[0].tolist(), [103, 104, 105])
        self.assertEqual(first.block_table[0].tolist(), [19, 20, 21])

    def test_valid_count_cannot_be_replaced_by_capacity(self):
        self.owner.num_blocks.np[0, 2] = 5
        with self.assertRaisesRegex(ValueError, "does not cover"):
            self.read()

    def test_upper_is_not_used_as_exact(self):
        self.owner.num_blocks.np[0, 2] = 6
        view = self.read(upper=(31, 15))
        self.assertEqual(view.seq_lens.tolist(), [9, 7, 0])
        self.assertEqual(view.seq_lens_cpu_upper_bound.tolist(), [19, 11, 0])

    def test_typed_application_preserves_query_and_write_identity(self):
        slots, positions, query = object(), object(), object()
        common = SimpleNamespace(slot_mapping=slots, positions=positions, query_start_loc=query)
        self.read().apply(common)
        self.assertIs(common.slot_mapping, slots)
        self.assertIs(common.positions, positions)
        self.assertIs(common.query_start_loc, query)

    def test_reject_unrepresentable_start_and_inactive_mapping(self):
        with self.assertRaisesRegex(ValueError, "intra-block"):
            self.read(starts=(13, 4))
        self.mapping[0] = -1
        with self.assertRaisesRegex(ValueError, "inactive"):
            self.read()

    def test_continuing_full_row_and_window_row_share_batch(self):
        view = self.read(starts=(0, 4))
        self.assertEqual(view.seq_lens.tolist(), [21, 7, 0])
        self.assertEqual(view.block_table[0, :6].tolist(), [16, 17, 18, 19, 20, 21])


if __name__ == "__main__":
    unittest.main()
