# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU oracle for fixed histories with reordered and speculative rows."""

import unittest

import torch

from tests.e2e.common.multi_node.glm53_logits.trace_processor import mask_trace_logits


class TestTraceProcessor(unittest.TestCase):
    def test_reordered_speculative_inactive_and_padded_rows(self):
        raw = torch.arange(30, dtype=torch.float32).reshape(6, 5)
        scores = raw.clone()
        slots = torch.tensor([1, 1, 0, 0, 2, -1])
        positions = torch.tensor([9, 10, 4, 6, 7, 4])
        prompt_lengths = torch.tensor([5, 10, 8])
        traces = torch.tensor([[2, 4, 1], [3, 0, 4], [1, 1, 1]])
        lengths = torch.tensor([2, 3, 0])
        result = mask_trace_logits(scores, slots, positions, prompt_lengths, traces, lengths)
        for row, chosen in [(0, 3), (1, 0), (2, 2)]:
            self.assertEqual(result[row, chosen], raw[row, chosen])
            self.assertTrue(torch.isneginf(result[row, torch.arange(5) != chosen]).all())
        for row in [3, 4, 5]:
            self.assertTrue(torch.equal(result[row], raw[row]))
        self.assertTrue(torch.equal(raw, torch.arange(30, dtype=torch.float32).reshape(6, 5)))


if __name__ == "__main__":
    unittest.main()
