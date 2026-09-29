# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU regression checks for the A5 comparison harness, not NPU kernels."""

import unittest

import run_a5
import torch

run_a5.torch = torch


class ScoreComparison(unittest.TestCase):
    def setUp(self):
        self.baseline = torch.arange(64 * 875, dtype=torch.float32).reshape(64, 875)
        self.candidate = self.baseline.clone()
        self.candidate[8:] = torch.finfo(torch.float32).min

    def compare(self):
        # Different chunk boundaries must not change which global rows are live.
        run_a5.assert_indexer_scores(list(self.baseline.split(10)), list(self.candidate.split(7)), live=8)

    def test_padding_difference_is_not_numerical_regression(self):
        self.assertEqual(int((self.baseline != self.candidate).sum()), 49000)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(self.baseline, self.candidate, rtol=0, atol=0)
        self.compare()

    def test_single_ulp_difference_in_live_row_still_fails(self):
        self.candidate[7, 0] = torch.nextafter(self.candidate[7, 0], torch.tensor(float("inf")))
        with self.assertRaises(AssertionError):
            self.compare()

    def test_incorrect_padding_sentinel_still_fails(self):
        self.candidate[8, 0] = 0
        with self.assertRaisesRegex(AssertionError, "Candidate padding scores"):
            self.compare()


if __name__ == "__main__":
    unittest.main(verbosity=2)
