# SPDX-License-Identifier: Apache-2.0
import random
from unittest import TestCase

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.host_partition import partition_shared_block_groups


class TestSharedHostOwnership(TestCase):
    def test_exact_coverage_for_fragmented_uneven_and_empty_requests(self):
        rng = random.Random(19212)
        for size in (1, 2, 3, 4, 8, 16):
            for count in (0, 1, 2, 7, 8, 9, 31, 32, 33, 97):
                local = rng.sample(range(2048), count)
                remote = rng.sample(range(4096), count)
                # Fragmentation and grouping must not change ownership or
                # accidentally pair a local page with the next source page.
                groups = [local[i : i + 3] for i in range(0, count, 3)]
                peers = [remote[i : i + 3] for i in range(0, count, 3)]
                owned = []
                sorted_local = sorted(local)
                quotient, remainder = divmod(count, size)
                for rank in range(size):
                    dst, src = partition_shared_block_groups(groups, peers, rank, size)
                    pairs = [(a, b) for x, y in zip(dst, src) for a, b in zip(x, y)]
                    start = rank * quotient + min(rank, remainder)
                    owned_count = quotient + int(rank < remainder)
                    self.assertEqual([a for a, _ in pairs], sorted_local[start : start + owned_count])
                    owned.extend(pairs)
                self.assertCountEqual(owned, list(zip(local, remote)))
                self.assertEqual(len(owned), len(set(owned)))

    def test_coalesces_only_when_source_and_destination_are_both_contiguous(self):
        self.assertEqual(
            partition_shared_block_groups([[1, 2], [3], [7, 8]], [[10, 11], [12], [20, 25]], 0, 1),
            ([[1, 2, 3], [7], [8]], [[10, 11, 12], [20], [25]]),
        )

    def test_consecutive_ownership_coalesces_a_complete_run_per_rank(self):
        local = [list(reversed(range(64)))]
        remote = [[block + 100 for block in local[0]]]
        for rank in range(8):
            with self.subTest(rank=rank):
                dst, src = partition_shared_block_groups(local, remote, rank, 8)
                self.assertEqual(dst, [list(range(rank * 8, rank * 8 + 8))])
                self.assertEqual(src, [list(range(100 + rank * 8, 108 + rank * 8))])

    def test_regrouping_input_does_not_change_ownership_or_source_pairs(self):
        local = [12, 2, 9, 4, 5, 20, 1]
        remote = [103, 88, 77, 99, 100, 66, 55]
        for rank in range(3):
            original = partition_shared_block_groups([local], [remote], rank, 3)
            regrouped = partition_shared_block_groups(
                [[local[i]] for i in reversed(range(len(local)))],
                [[remote[i]] for i in reversed(range(len(remote)))],
                rank,
                3,
            )
            self.assertEqual(original, regrouped)

    def test_duplicate_destinations_have_one_owner_and_keep_source_order(self):
        owned = []
        for rank in range(4):
            dst, src = partition_shared_block_groups([[3, 1, 3, 2]], [[30, 10, 31, 20]], rank, 4)
            pairs = [(a, b) for x, y in zip(dst, src) for a, b in zip(x, y)]
            if any(a == 3 for a, _ in pairs):
                owned.append((rank, [b for a, b in pairs if a == 3]))
        self.assertEqual(owned, [(2, [30, 31])])

    def test_noncontiguous_byte_layout_disables_block_id_coalescing(self):
        self.assertEqual(
            partition_shared_block_groups([[0, 1, 2, 3]], [[10, 11, 12, 13]], 1, 2, coalesce=False),
            ([[2], [3]], [[12], [13]]),
        )

    def test_invalid_ownership_or_mismatched_descriptors_fail_before_transfer(self):
        for local, remote, rank, size in (
            ([[1]], [], 0, 1),
            ([[1, 2]], [[3]], 0, 1),
            ([[1]], [[-1]], 0, 1),
            ([], [], -1, 2),
            ([], [], 2, 2),
            ([], [], 0, 0),
        ):
            with self.assertRaises(ValueError):
                partition_shared_block_groups(local, remote, rank, size)
