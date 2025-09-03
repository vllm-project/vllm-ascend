import torch
import unittest
from vllm_ascend.eplb.core.eplb_utils import determine_default_expert_map


class TestDetermineDefaultExpertMap(unittest.TestCase):

    def test_world_size_1(self):
        global_expert_num = 8
        world_size = 1
        global_redundant_expert_num = 0

        expected_counts = [8]
        expected_maps = [[0, 1, 2, 3, 4, 5, 6, 7]]

        local_count, expert_map = determine_default_expert_map(
            global_expert_num, world_size, 0, global_redundant_expert_num)

        self.assertEqual(local_count, expected_counts[0])

        expected_tensor = torch.tensor(expected_maps[0], dtype=torch.int32)
        self.assertTrue(torch.all(expert_map == expected_tensor).item())

    def test_equal_distribution(self):
        global_expert_num = 6
        world_size = 3
        global_redundant_expert_num = 0

        expected_counts = [2, 2, 2]
        expected_maps = [
            [0, 1, -1, -1, -1, -1],  # rank 0
            [-1, -1, 0, 1, -1, -1],  # rank 1
            [-1, -1, -1, -1, 0, 1]  # rank 2
        ]

        for rank_id in range(world_size):
            local_count, expert_map = determine_default_expert_map(
                global_expert_num, world_size, rank_id,
                global_redundant_expert_num)

            self.assertEqual(
                local_count,
                expected_counts[rank_id],
            )

            expected_tensor = torch.tensor(expected_maps[rank_id],
                                           dtype=torch.int32)
            self.assertTrue(torch.all(expert_map == expected_tensor).item())

    def test_unequal_distribution(self):
        global_expert_num = 10
        world_size = 3
        global_redundant_expert_num = 0

        expected_counts = [3, 3, 4]
        expected_maps = [
            [0, 1, 2, -1, -1, -1, -1, -1, -1, -1],  # rank 0
            [-1, -1, -1, 0, 1, 2, -1, -1, -1, -1],  # rank 1
            [-1, -1, -1, -1, -1, -1, 0, 1, 2, 3]  # rank 2
        ]

        for rank_id in range(world_size):
            local_count, expert_map = determine_default_expert_map(
                global_expert_num, world_size, rank_id,
                global_redundant_expert_num)

            self.assertEqual(local_count, expected_counts[rank_id])

            expected_tensor = torch.tensor(expected_maps[rank_id],
                                           dtype=torch.int32)
            self.assertTrue(torch.all(expert_map == expected_tensor).item())

    def test_with_redundancy(self):
        global_expert_num = 7
        world_size = 3
        global_redundant_expert_num = 2

        expected_counts = [3, 3, 3]
        expected_maps = [
            [0, 1, 2, -1, -1, -1, -1],  # rank 0
            [-1, -1, 0, 1, 2, -1, -1],  # rank 1
            [-1, -1, -1, -1, 0, 1, 2]  # rank 2
        ]

        for rank_id in range(world_size):
            local_count, expert_map = determine_default_expert_map(
                global_expert_num, world_size, rank_id,
                global_redundant_expert_num)

            self.assertEqual(local_count, expected_counts[rank_id])

            expected_tensor = torch.tensor(expected_maps[rank_id],
                                           dtype=torch.int32)
            self.assertTrue(torch.all(expert_map == expected_tensor).item())

    def test_redundancy_at_boundary(self):
        global_expert_num = 5
        world_size = 2
        global_redundant_expert_num = 1

        expected_counts = [3, 3]
        expected_maps = [[0, 1, 2, -1, -1], [-1, -1, 0, 1, 2]]

        for rank_id in range(world_size):
            local_count, expert_map = determine_default_expert_map(
                global_expert_num, world_size, rank_id,
                global_redundant_expert_num)

            self.assertEqual(local_count, expected_counts[rank_id])

            expected_tensor = torch.tensor(expected_maps[rank_id],
                                           dtype=torch.int32)
            self.assertTrue(torch.all(expert_map == expected_tensor).item())
