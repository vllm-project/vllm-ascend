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

from __future__ import annotations

import torch
import pytest
from unittest.mock import MagicMock, patch


class TestRoutedExpertsCapturerCapture:
    """Unit tests for RoutedExpertsCapturer.capture method."""

    def _create_mock_capturer(self, dp_rank=0, tp_size=1):
        """Create a mock RoutedExpertsCapturer instance."""
        capturer = MagicMock()
        capturer.dp_rank = dp_rank
        capturer.tp_size = tp_size
        capturer.device_buffer = torch.zeros((100, 10, 8))  # (max_tokens, num_layers, num_experts)
        return capturer

    def _create_mock_forward_context(self, dp_metadata=None):
        """Create a mock forward context."""
        ctx = MagicMock()
        ctx.dp_metadata = dp_metadata
        return ctx

    def _create_mock_dp_metadata(self, num_tokens_across_dp_cpu):
        """Create mock DP metadata with token counts."""
        dp_metadata = MagicMock()
        dp_metadata.num_tokens_across_dp_cpu = torch.tensor(num_tokens_across_dp_cpu)
        return dp_metadata

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    def test_single_dp(self, mock_get_ctx):
        """Test single DP scenario (no data parallelism)."""
        # Setup
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=1)
        ctx = self._create_mock_forward_context(dp_metadata=None)
        mock_get_ctx.return_value = ctx

        topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)

        # Execute
        capturer.capture(layer_id=0, topk_ids=topk_ids)

        # Verify
        expected = topk_ids
        actual = capturer.device_buffer[:3, 0, :]
        torch.testing.assert_close(actual, expected)

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    def test_multi_dp_naive_dispatch(self, mock_get_ctx):
        """Test multi-DP naive dispatch (n == total)."""
        # Setup
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=1)
        dp_metadata = self._create_mock_dp_metadata([5, 7])  # DP0: 5 tokens, DP1: 7 tokens
        ctx = self._create_mock_forward_context(dp_metadata=dp_metadata)
        mock_get_ctx.return_value = ctx

        # Naive dispatch: all tokens concatenated (5 + 7 = 12)
        topk_ids = torch.arange(12 * 2).view(12, 2).to(torch.int32)

        # Execute
        capturer.capture(layer_id=0, topk_ids=topk_ids)

        # Verify: DP0 should get first 5 tokens
        expected = topk_ids[:5]
        actual = capturer.device_buffer[:5, 0, :]
        torch.testing.assert_close(actual, expected)

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    def test_multi_dp_modular_kernel(self, mock_get_ctx):
        """Test multi-DP modular kernel path (n == token_num_per_dp)."""
        # Setup
        capturer = self._create_mock_capturer(dp_rank=1, tp_size=1)
        dp_metadata = self._create_mock_dp_metadata([5, 7])  # DP0: 5 tokens, DP1: 7 tokens
        ctx = self._create_mock_forward_context(dp_metadata=dp_metadata)
        mock_get_ctx.return_value = ctx

        # Modular kernel: only this rank's tokens (7)
        topk_ids = torch.arange(7 * 2).view(7, 2).to(torch.int32)

        # Execute
        capturer.capture(layer_id=0, topk_ids=topk_ids)

        # Verify: DP1 should get all 7 tokens
        expected = topk_ids
        actual = capturer.device_buffer[:7, 0, :]
        torch.testing.assert_close(actual, expected)

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    def test_multi_dp_padded_all_gather(self, mock_get_ctx):
        """Test multi-DP padded all-gather path (n == total_with_padding)."""
        # Setup
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=1)
        dp_metadata = self._create_mock_dp_metadata([5, 7])  # DP0: 5, DP1: 7, max=7
        ctx = self._create_mock_forward_context(dp_metadata=dp_metadata)
        mock_get_ctx.return_value = ctx

        # Padded all-gather: tokens padded to max_tokens=7, total_with_padding=14
        topk_ids = torch.arange(14 * 2).view(14, 2).to(torch.int32)

        # Execute
        capturer.capture(layer_id=0, topk_ids=topk_ids)

        # Verify: DP0 should get first 5 tokens (skip padding)
        expected = topk_ids[:5]
        actual = capturer.device_buffer[:5, 0, :]
        torch.testing.assert_close(actual, expected)

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_tp_group")
    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.dist")
    def test_sp_modular_kernel_all2all(self, mock_dist, mock_get_tp_group, mock_get_ctx):
        """Test SP + modular kernel path with ALLTOALL comm type."""
        # Setup
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=2)
        dp_metadata = self._create_mock_dp_metadata([10])  # 10 tokens
        ctx = self._create_mock_forward_context(dp_metadata=dp_metadata)
        mock_get_ctx.return_value = ctx

        mock_tp_group = MagicMock()
        mock_get_tp_group.return_value = mock_tp_group
        mock_tp_group.device_group = MagicMock()

        # Mock EXTRA_CTX for ALLTOALL
        from vllm_ascend.core.forward_context import _EXTRA_CTX
        from vllm_ascend.core.forward_context import MoECommType

        original_comm_type = _EXTRA_CTX.moe_comm_type
        _EXTRA_CTX.moe_comm_type = MoECommType.ALLTOALL

        try:
            # SP + modular: ceil(10/2) = 5 tokens per TP rank
            topk_ids = torch.arange(5 * 2).view(5, 2).to(torch.int32)

            # Mock all_gather to simulate gathering from TP ranks
            def mock_all_gather(output_list, input_tensor, device_group):
                # Simulate gathering: TP0 has [0-4], TP1 has [5-9]
                output_list[0].copy_(input_tensor)
                output_list[1].copy_(input_tensor + 5 * 2)

            mock_dist.all_gather = mock_all_gather

            # Execute
            capturer.capture(layer_id=0, topk_ids=topk_ids)

            # Verify: should have all 10 tokens
            expected = torch.arange(10 * 2).view(10, 2).to(torch.int32)
            actual = capturer.device_buffer[:10, 0, :]
            torch.testing.assert_close(actual, expected)
        finally:
            _EXTRA_CTX.moe_comm_type = original_comm_type

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_tp_group")
    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.dist")
    def test_sp_modular_kernel_mc2(self, mock_dist, mock_get_tp_group, mock_get_ctx):
        """Test SP + modular kernel path with MC2 comm type."""
        # Setup
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=2)
        dp_metadata = self._create_mock_dp_metadata([5, 7])  # DP0: 5, DP1: 7, max=7
        ctx = self._create_mock_forward_context(dp_metadata=dp_metadata)
        mock_get_ctx.return_value = ctx

        mock_tp_group = MagicMock()
        mock_get_tp_group.return_value = mock_tp_group
        mock_tp_group.device_group = MagicMock()

        # Mock EXTRA_CTX for MC2
        from vllm_ascend.core.forward_context import _EXTRA_CTX
        from vllm_ascend.core.forward_context import MoECommType

        original_comm_type = _EXTRA_CTX.moe_comm_type
        _EXTRA_CTX.moe_comm_type = MoECommType.MC2

        try:
            # MC2: ceil(max_tokens/tp_size) = ceil(7/2) = 4 tokens per TP rank
            topk_ids = torch.arange(4 * 2).view(4, 2).to(torch.int32)

            # Mock all_gather
            def mock_all_gather(output_list, input_tensor, device_group):
                output_list[0].copy_(input_tensor)
                output_list[1].copy_(input_tensor + 4 * 2)

            mock_dist.all_gather = mock_all_gather

            # Execute
            capturer.capture(layer_id=0, topk_ids=topk_ids)

            # Verify: DP0 should get first 5 tokens
            actual = capturer.device_buffer[:5, 0, :]
            assert actual.shape[0] == 5
        finally:
            _EXTRA_CTX.moe_comm_type = original_comm_type

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    def test_unexpected_batch_dim(self, mock_get_ctx):
        """Test that unexpected batch dimension raises AssertionError."""
        # Setup
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=2)
        dp_metadata = self._create_mock_dp_metadata([5, 7])
        ctx = self._create_mock_forward_context(dp_metadata=dp_metadata)
        mock_get_ctx.return_value = ctx

        # Unexpected batch size (not matching any known pattern)
        topk_ids = torch.randint(0, 8, (100, 2)).to(torch.int32)

        # Execute and verify
        with pytest.raises(AssertionError, match="unexpected topk_ids batch"):
            capturer.capture(layer_id=0, topk_ids=topk_ids)

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    def test_layer_id_out_of_bounds(self, mock_get_ctx):
        """Test that out-of-bounds layer_id is handled gracefully."""
        # Setup
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=1)
        ctx = self._create_mock_forward_context(dp_metadata=None)
        mock_get_ctx.return_value = ctx

        topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
        capturer.device_buffer = torch.zeros((100, 5, 8))  # Only 5 layers

        # Execute with layer_id beyond buffer size
        capturer.capture(layer_id=10, topk_ids=topk_ids)

        # Verify: no error, and buffer should be unchanged
        assert torch.all(capturer.device_buffer == 0)
