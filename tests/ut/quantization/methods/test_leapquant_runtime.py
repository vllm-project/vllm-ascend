# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the LeapQuant runtime glue (row selection + pool flush)."""

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ascend_config import LeapQuantConfig
from vllm_ascend.quantization.methods.mamba_state.leapquant import (
    leapquant_config_or_none,
    leapquant_flush_state_rows,
)


class _FakeStateCopy:
    """Duck-typed stand-in for the KDA compiled state-copy plan (CPU)."""

    def __init__(self):
        self.gather_calls = 0
        self.scatter_calls = 0

    def gather(self, state: torch.Tensor, indices: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        self.gather_calls += 1
        return state.index_select(0, indices)[flags]

    def scatter(self, state: torch.Tensor, source: torch.Tensor, indices: torch.Tensor) -> None:
        self.scatter_calls += 1
        state.index_copy_(0, indices, source)


class TestLeapQuantRuntime(unittest.TestCase):
    def setUp(self):
        self.config = LeapQuantConfig(enabled=True, window=16, compensator_tokens=4)
        torch.manual_seed(11)

    def test_only_due_rows_flushed(self):
        pool = torch.randn(8, 4, 32, 48)
        indices = torch.tensor([3, 5, 6])
        seq_lens = torch.tensor([15, 16, 32])  # rows 1 (slot 5) and 2 (slot 6) due
        before = pool.clone()
        leapquant_flush_state_rows(pool, indices, seq_lens, config=self.config)
        self.assertFalse(bool((pool[3] != before[3]).any()))
        self.assertTrue(bool((pool[5] != before[5]).any()))
        self.assertTrue(bool((pool[6] != before[6]).any()))
        untouched = [0, 1, 2, 4, 7]
        self.assertTrue(bool((pool[untouched] == before[untouched]).all()))

    def test_no_due_rows_is_noop(self):
        pool = torch.randn(4, 2, 16, 16)
        before = pool.clone()
        leapquant_flush_state_rows(pool, torch.tensor([0, 1]), torch.tensor([5, 33]), config=self.config)
        self.assertTrue(torch.equal(pool, before))

    def test_empty_indices(self):
        pool = torch.randn(4, 2, 16, 16)
        before = pool.clone()
        leapquant_flush_state_rows(pool, torch.zeros(0, dtype=torch.long), torch.zeros(0), config=self.config)
        self.assertTrue(torch.equal(pool, before))

    def test_state_copy_plan_used_when_provided(self):
        pool = torch.randn(8, 2, 16, 16)
        indices = torch.tensor([1, 2])
        seq_lens = torch.tensor([16, 17])
        plan = _FakeStateCopy()
        before = pool.clone()
        # The plan path targets NPU tensors; on CPU the helper must ignore it
        # and still flush the due row correctly.
        leapquant_flush_state_rows(pool, indices, seq_lens, config=self.config, state_copy=plan)
        self.assertEqual(plan.gather_calls, 0)
        self.assertTrue(bool((pool[1] != before[1]).any()))
        self.assertFalse(bool((pool[2] != before[2]).any()))

    def test_padded_stride_pool(self):
        # Slots embedded as a non-contiguous view (hybrid padded pools).
        wide = torch.randn(8, 4, 32, 64)
        view = wide[:, :, :, :48]
        indices = torch.tensor([2, 4])
        seq_lens = torch.tensor([31, 48])
        before = view.clone()
        leapquant_flush_state_rows(view, indices, seq_lens, config=self.config)
        self.assertFalse(bool((view[2] != before[2]).any()))  # 31 % 16 != 0
        self.assertTrue(bool((view[4] != before[4]).any()))  # 48 % 16 == 0

    def test_config_or_none_degrades_without_singleton(self):
        with patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            side_effect=RuntimeError("Ascend config is not initialized"),
        ):
            self.assertIsNone(leapquant_config_or_none())

    def test_config_or_none_disabled(self):
        mock = MagicMock()
        mock.leapquant_config = LeapQuantConfig(enabled=False)
        with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=mock):
            self.assertIsNone(leapquant_config_or_none())

    def test_config_or_none_enabled(self):
        mock = MagicMock()
        mock.leapquant_config = self.config
        with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=mock):
            resolved: Any = leapquant_config_or_none()
            self.assertIs(resolved, self.config)


if __name__ == "__main__":
    unittest.main()
