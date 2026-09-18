# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from unittest.mock import patch

import torch

from vllm_ascend.distributed.eplb.layer_sharding import all_gather_layer_shards, assigned_layer_ids


class FakeGroup:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class TestLayerSharding(unittest.TestCase):
    def test_assignment_uses_stage_local_round_robin(self):
        self.assertEqual(list(assigned_layer_ids(7, 1, 3)), [1, 4])
        self.assertEqual(list(assigned_layer_ids(2, 2, 4)), [])

    @patch("vllm_ascend.distributed.eplb.layer_sharding.torch.distributed.all_gather")
    def test_all_gather_reconstructs_layer_order(self, all_gather):
        shards = [
            torch.tensor([[0, 0], [3, 30], [6, 60]]),
            torch.tensor([[1, 10], [4, 40], [7, 70]]),
            torch.tensor([[2, 20], [5, 50], [0, 0]]),
        ]

        def gather(outputs, local, *, group):
            self.assertEqual(group.rank(), 2)
            torch.testing.assert_close(local, shards[2])
            for output, shard in zip(outputs, shards):
                output.copy_(shard)

        all_gather.side_effect = gather
        result = all_gather_layer_shards(shards[2][:2], 8, FakeGroup(3, 2))

        torch.testing.assert_close(
            result,
            torch.tensor([[0, 0], [1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70]]),
        )

    @patch("vllm_ascend.distributed.eplb.layer_sharding.torch.distributed.all_gather")
    def test_rank_without_layers_still_joins_collective(self, all_gather):
        shards = [torch.tensor([[0]]), torch.tensor([[1]]), torch.tensor([[0]]), torch.tensor([[0]])]

        def gather(outputs, local, *, group):
            self.assertEqual(group.rank(), 2)
            torch.testing.assert_close(local, torch.tensor([[0]]))
            for output, shard in zip(outputs, shards):
                output.copy_(shard)

        all_gather.side_effect = gather
        result = all_gather_layer_shards(torch.empty((0, 1), dtype=torch.int64), 2, FakeGroup(4, 2))

        torch.testing.assert_close(result, torch.tensor([[0], [1]]))

    @patch("vllm_ascend.distributed.eplb.layer_sharding.torch.distributed.all_gather")
    def test_empty_stage_skips_collective(self, all_gather):
        result = all_gather_layer_shards(torch.empty((0, 2), dtype=torch.int64), 0, FakeGroup(2, 0))

        self.assertEqual(result.shape, (0, 2))
        all_gather.assert_not_called()

    def test_rejects_incorrect_local_shard_size(self):
        with self.assertRaisesRegex(ValueError, "static assignment"):
            all_gather_layer_shards(torch.zeros((3, 1)), 4, FakeGroup(2, 0))


if __name__ == "__main__":
    unittest.main()
