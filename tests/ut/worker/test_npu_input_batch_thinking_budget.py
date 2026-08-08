#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Regression tests for thinking_token_budget enforcement on NPU.

These tests verify that ``NPUInputBatch`` correctly creates the
``thinking_budget_state_holder`` when a ``reasoning_config`` is provided,
instead of hardcoding it to ``None`` (which silently disabled the
``thinking_token_budget`` hard truncation mechanism).

See https://github.com/vllm-project/vllm-ascend/issues/13060
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
from vllm.config.reasoning import ReasoningConfig
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

from vllm_ascend.worker.npu_input_batch import NPUInputBatch


def _make_input_batch(
    reasoning_config: ReasoningConfig | None = None,
    num_speculative_tokens: int = 0,
) -> NPUInputBatch:
    """Create a minimal NPUInputBatch on CPU for testing.

    Mocks the distributed group accessors so that ``MultiGroupBlockTable``
    can be constructed without a real distributed environment.
    """
    mock_group = MagicMock()
    mock_group.world_size = 1
    mock_group.rank_in_group = 0
    with (
        patch("vllm_ascend.worker.block_table.get_dcp_group", return_value=mock_group),
        patch("vllm_ascend.worker.block_table.get_pcp_group", return_value=mock_group),
    ):
        return NPUInputBatch(
            max_num_reqs=4,
            max_model_len=128,
            max_num_batched_tokens=128,
            device=torch.device("cpu"),
            pin_memory=False,
            vocab_size=1024,
            block_sizes=[16],
            kernel_block_sizes=[[16]],
            num_speculative_tokens=num_speculative_tokens,
            reasoning_config=reasoning_config,
        )


class TestThinkingBudgetStateHolder(unittest.TestCase):
    """Tests that NPUInputBatch creates the thinking budget state holder."""

    def test_holder_is_none_without_reasoning_config(self):
        """Without reasoning_config, holder should be None (thinking disabled)."""
        batch = _make_input_batch(reasoning_config=None)
        self.assertIsNone(batch.thinking_budget_state_holder)

    def test_holder_created_with_reasoning_config(self):
        """With reasoning_config, holder should be a ThinkingBudgetStateHolder.

        This is the regression test for issue #13060: previously the holder
        was hardcoded to None, silently disabling thinking_token_budget.
        """
        reasoning_config = ReasoningConfig(
            reasoning_parser="qwen3",
            reasoning_start_str="<think>",
            reasoning_end_str="</think>",
        )
        batch = _make_input_batch(reasoning_config=reasoning_config)
        self.assertIsNotNone(batch.thinking_budget_state_holder)
        self.assertIsInstance(batch.thinking_budget_state_holder, ThinkingBudgetStateHolder)

    def test_holder_is_enabled_with_reasoning_config(self):
        """The created holder should report is_enabled=True."""
        reasoning_config = ReasoningConfig(
            reasoning_parser="qwen3",
            reasoning_start_str="<think>",
            reasoning_end_str="</think>",
        )
        batch = _make_input_batch(reasoning_config=reasoning_config)
        self.assertTrue(batch.thinking_budget_state_holder.is_enabled)

    def test_holder_spec_mode_flag(self):
        """Holder should reflect speculative decoding mode."""
        reasoning_config = ReasoningConfig(
            reasoning_parser="qwen3",
            reasoning_start_str="<think>",
            reasoning_end_str="</think>",
        )
        batch = _make_input_batch(reasoning_config=reasoning_config, num_speculative_tokens=2)
        holder = batch.thinking_budget_state_holder
        self.assertIsNotNone(holder)
        self.assertTrue(holder.in_spec_mode)
        self.assertEqual(holder.num_spec_tokens, 2)

    def test_holder_has_no_tracked_requests_initially(self):
        """A freshly created holder should have no tracked requests."""
        reasoning_config = ReasoningConfig(
            reasoning_parser="qwen3",
            reasoning_start_str="<think>",
            reasoning_end_str="</think>",
        )
        batch = _make_input_batch(reasoning_config=reasoning_config)
        holder = batch.thinking_budget_state_holder
        self.assertFalse(holder.has_tracked_requests())


if __name__ == "__main__":
    unittest.main()
