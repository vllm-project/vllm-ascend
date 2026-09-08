# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os
import unittest
from unittest.mock import patch

import torch

from vllm_ascend.attention.sfa_v1 import canonicalize_topk_indices


class TestCanonicalizeTopkIndices(unittest.TestCase):
    def test_valid_indices_are_sorted_and_padding_stays_at_the_tail(self):
        indices = torch.tensor([[[5, 2, -1, 9, -1, 0]]], dtype=torch.int32)

        out = canonicalize_topk_indices(indices)

        self.assertEqual(out.tolist(), [[[0, 2, 5, 9, -1, -1]]])
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.shape, indices.shape)

    def test_the_selected_set_is_preserved_for_every_row(self):
        torch.manual_seed(0)
        num_tokens, topk = 37, 64
        rows = []
        for token in range(num_tokens):
            valid = min(topk, token + 1)
            row = torch.randperm(token + 1, dtype=torch.int64)[:valid].to(torch.int32)
            rows.append(torch.cat([row, torch.full((topk - valid,), -1, dtype=torch.int32)]))
        indices = torch.stack(rows).unsqueeze(1)

        out = canonicalize_topk_indices(indices)

        for before, after in zip(indices.flatten(0, 1), out.flatten(0, 1)):
            valid_after = after[after >= 0]
            self.assertEqual(set(valid_after.tolist()), set(before[before >= 0].tolist()))
            self.assertTrue(torch.equal(valid_after, valid_after.sort().values))
            self.assertTrue(torch.all(after[valid_after.numel() :] == -1))

    def test_large_positions_survive_the_float32_key(self):
        indices = torch.tensor([[[1 << 23, 3, (1 << 23) - 1]]], dtype=torch.int32)

        out = canonicalize_topk_indices(indices)

        self.assertEqual(out.tolist(), [[[3, (1 << 23) - 1, 1 << 23]]])

    def test_disabled_by_env_returns_the_indexer_order(self):
        indices = torch.tensor([[[5, 2, -1, 9]]], dtype=torch.int32)

        with patch.dict(os.environ, {"VLLM_ASCEND_SFA_SORT_TOPK": "0"}):
            out = canonicalize_topk_indices(indices)

        self.assertTrue(torch.equal(out, indices))


if __name__ == "__main__":
    unittest.main()
