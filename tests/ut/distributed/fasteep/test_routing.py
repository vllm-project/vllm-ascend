# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import unittest

import torch

from vllm_ascend.distributed.fasteep.policy import local_physical_mask


class TestPhysicalRouting(unittest.TestCase):
    def test_replicated_shards_contribute_once(self):
        # Rank 2 duplicates rank 0's logical experts, but only owns IDs 4, 5.
        ids = torch.tensor([[0, 4], [3, 7]])
        values = torch.tensor([[2.0, 5.0], [7.0, 11.0]])
        masks = torch.stack([local_physical_mask(ids, rank, 2) for rank in range(4)])
        torch.testing.assert_close(masks.sum(0), torch.ones_like(ids))
        torch.testing.assert_close((values * masks).sum((0, 2)), torch.tensor([7.0, 18.0]))
        self.assertFalse(local_physical_mask(torch.tensor([-1, 8]), 0, 2).any())

    def test_inactive_physical_ids_are_excluded(self):
        ids = torch.tensor([0, 1, 2, 3, 4, 5])
        count = sum(local_physical_mask(ids, rank, 2).int() for rank in range(2))
        torch.testing.assert_close(count, torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.int32))
