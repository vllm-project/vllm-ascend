# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import random
import unittest

from vllm_ascend.distributed.fasteep.placement import plan_active_placement, plan_transfers, validate_placement


def logical_map(experts, logical):
    row = [-1] * logical
    for slot, expert in enumerate(experts):
        row[expert] = slot
    return row


class TestExpertMigration(unittest.TestCase):
    def assert_migration(self, current, target):
        # Unique values per expert verify the final contents independently of
        # donor selection; snapshots preserve sources in cyclic permutations.
        logical, capacity = validate_placement(current)
        storage: list[list[int | None]] = [[None] * capacity for _ in current]
        for rank, row in enumerate(current):
            for expert, slot in enumerate(row):
                if slot >= 0:
                    storage[rank][slot] = 1000 + expert
        result = [list(storage[rank]) if rank < len(storage) else [None] * capacity for rank in range(len(target))]
        for move in plan_transfers(current, target):
            result[move.target_rank][move.target_slot] = storage[move.source_rank][move.source_slot]
        for rank, row in enumerate(target):
            for expert, slot in enumerate(row):
                if slot >= 0:
                    self.assertEqual(result[rank][slot], 1000 + expert)
        self.assertEqual(validate_placement(target), (logical, capacity))

    def test_shrink_without_whole_shard_replicas(self):
        current = [logical_map(x, 4) for x in ([0, 1], [1, 2], [2, 3], [3, 0])]
        target = plan_active_placement(current, current, 2)
        self.assert_migration(current, target)
        restored = plan_active_placement(target, current, 4)
        self.assert_migration(target, restored)

    def test_cyclic_slot_permutation(self):
        current = [[0, 1, 2], [2, 0, 1]]
        target = [[2, 0, 1], [1, 2, 0]]
        self.assert_migration(current, target)

    def test_insufficient_capacity_rejected_before_migration(self):
        current = [logical_map(x, 6) for x in ([0, 1], [2, 3], [4, 5])]
        with self.assertRaisesRegex(ValueError, "insufficient"):
            plan_active_placement(current, current, 2)

    def test_random_repeated_shrink_and_grow(self):
        rng = random.Random(7)
        for _ in range(100):
            template = [logical_map(rng.sample(range(8), 4), 8) for _ in range(6)]
            if any(all(row[e] < 0 for row in template) for e in range(8)):
                continue
            current = template
            for count in [3, 2, 5, 2, 6]:
                target = plan_active_placement(current, template, count)
                self.assert_migration(current, target)
                current = target


if __name__ == "__main__":
    unittest.main()
