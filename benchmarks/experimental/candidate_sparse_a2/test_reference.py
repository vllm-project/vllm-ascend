# SPDX-License-Identifier: Apache-2.0
"""Test selection equivalence, not just a particular gather implementation."""

import itertools
import math
import random
import unittest

from reference import block_plan, compact_select, select


class CandidateCompactionTests(unittest.TestCase):
    def test_exhaustive_small_masks_and_partial_blocks(self):
        for visible in range(34):
            blocks = math.ceil(visible / 8)
            for bits in itertools.product((False, True), repeat=blocks):
                candidates = [block for block, live in enumerate(bits) if live] + [-1] * (blocks - sum(bits))
                for topk in (1, 7, 8, 9, 16, 32):
                    for scores in (
                        [0.0] * visible,
                        [float((pos * 13) % 11 - 8) for pos in range(visible)],
                    ):
                        self.assertEqual(
                            compact_select(scores, candidates, topk),
                            select(scores, candidates, topk),
                        )

    def test_long_context_dense_oracle_and_fill(self):
        rng = random.Random(19304)
        for visible in (511, 512, 513, 16383, 16384, 16385, 32771, 131071, 131072):
            for count in (0, 1, 31, 63, 64, 65, 2048):
                count = min(count, math.ceil(visible / 8))
                candidates = rng.sample(range(math.ceil(visible / 8)), count)
                rng.shuffle(candidates)
                candidates += [-1] * (2048 - count)
                scores = [float(rng.randrange(-20, 20)) for _ in range(visible)]
                self.assertEqual(
                    compact_select(scores, candidates, 512),
                    select(scores, candidates, 512),
                )

    def test_plan_is_unique_visible_and_sorted(self):
        candidates = [4096, 0, 63, 64, 100, -1, 7000]
        blocks, mapped, positions = block_plan(candidates, 32771)
        self.assertEqual(blocks, sorted(set(blocks)))
        self.assertEqual(positions, sorted(set(positions)))
        self.assertTrue(all(0 <= pos < 32771 for pos in positions))
        self.assertEqual([blocks[pos] for pos in mapped], [0, 63, 64, 100, 4096])
        self.assertEqual(positions[-1], 32770)

    def test_capacity_and_expected_reduction(self):
        blocks, _, positions = block_plan(list(range(2048, 4096)), 131072)
        self.assertEqual(len(blocks), 2112)
        self.assertEqual(len(positions), 16896)
        self.assertAlmostEqual(131072 / len(positions), 7.757575757575758)

    def test_future_candidates_do_not_become_visible(self):
        candidates = [0, 1, 127, 1 << 20, -1]
        scores = [float(position - 20) for position in range(11)]
        self.assertEqual(compact_select(scores, candidates, 8), select(scores, candidates, 8))
        _, _, positions = block_plan(candidates, len(scores), 8)
        self.assertEqual(positions, list(range(11)))


if __name__ == "__main__":
    unittest.main()
