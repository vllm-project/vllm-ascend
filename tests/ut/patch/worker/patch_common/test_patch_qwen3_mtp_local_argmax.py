"""Tests for patch_qwen3_mtp_local_argmax.py

Verifies that Qwen3_5MTP and Qwen3NextMTP get the get_top_tokens method
patched onto them.
"""

from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture(autouse=True)
def _apply_patches():
    """Apply ascend_vllm patches before each test."""
    import vllm_ascend.patch.worker.patch_qwen3_mtp_local_argmax  # noqa: F401


class TestQwen3MTPGetTopTokens:
    def _make_mock_mtp(self, model_cls):
        """Create a minimal mock MTP instance with required attrs."""
        instance = MagicMock(spec=model_cls)
        instance.logits_processor = MagicMock()
        instance.lm_head = MagicMock()
        return instance

    def test_qwen3_5_mtp_has_get_top_tokens(self):
        from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

        assert hasattr(Qwen3_5MTP, "get_top_tokens"), "Qwen3_5MTP should have get_top_tokens after patching"
        assert callable(Qwen3_5MTP.get_top_tokens)

    def test_qwen3_next_mtp_has_get_top_tokens(self):
        from vllm.model_executor.models.qwen3_next_mtp import Qwen3NextMTP

        assert hasattr(Qwen3NextMTP, "get_top_tokens"), "Qwen3NextMTP should have get_top_tokens after patching"
        assert callable(Qwen3NextMTP.get_top_tokens)

    def test_get_top_tokens_calls_logits_processor(self):
        from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

        instance = self._make_mock_mtp(Qwen3_5MTP)
        hidden_states = torch.randn(4, 128)

        expected = torch.tensor([1, 2, 3, 4])
        instance.logits_processor.get_top_tokens.return_value = expected

        result = Qwen3_5MTP.get_top_tokens(instance, hidden_states, spec_step_idx=0)

        instance.logits_processor.get_top_tokens.assert_called_once_with(instance.lm_head, hidden_states)
        assert torch.equal(result, expected)

    def test_get_top_tokens_ignores_spec_step_idx(self):
        """The spec_step_idx parameter should be accepted but ignored."""
        from vllm.model_executor.models.qwen3_next_mtp import Qwen3NextMTP

        instance = self._make_mock_mtp(Qwen3NextMTP)
        hidden_states = torch.randn(2, 64)
        instance.logits_processor.get_top_tokens.return_value = torch.tensor([0, 1])

        # Should not raise regardless of spec_step_idx value
        result = Qwen3NextMTP.get_top_tokens(instance, hidden_states, spec_step_idx=999)
        assert result is not None

    def test_patch_is_idempotent(self):
        """Applying the patch a second time should not cause issues."""
        import importlib

        import vllm_ascend.patch.worker.patch_qwen3_mtp_local_argmax as mod1

        importlib.reload(mod1)

        from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

        assert hasattr(Qwen3_5MTP, "get_top_tokens")

    def test_get_top_tokens_on_next_mtp_returns_correct_values(self):
        """Verify get_top_tokens on Qwen3NextMTP returns correct token ids."""
        from vllm.model_executor.models.qwen3_next_mtp import Qwen3NextMTP

        instance = self._make_mock_mtp(Qwen3NextMTP)
        hidden_states = torch.randn(3, 128)
        expected = torch.tensor([5, 10, 15])
        instance.logits_processor.get_top_tokens.return_value = expected

        result = Qwen3NextMTP.get_top_tokens(instance, hidden_states)

        instance.logits_processor.get_top_tokens.assert_called_once_with(instance.lm_head, hidden_states)
        assert torch.equal(result, expected)

    def test_get_top_tokens_handles_single_token(self):
        """Verify get_top_tokens works with a single token batch."""
        from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

        instance = self._make_mock_mtp(Qwen3_5MTP)
        hidden_states = torch.randn(1, 128)
        expected = torch.tensor([42])
        instance.logits_processor.get_top_tokens.return_value = expected

        result = Qwen3_5MTP.get_top_tokens(instance, hidden_states, spec_step_idx=0)

        assert torch.equal(result, expected)

    def test_get_top_tokens_handles_empty_batch(self):
        """Verify get_top_tokens handles zero-length batch gracefully."""
        from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

        instance = self._make_mock_mtp(Qwen3_5MTP)
        hidden_states = torch.randn(0, 128)
        expected = torch.tensor([], dtype=torch.long)
        instance.logits_processor.get_top_tokens.return_value = expected

        result = Qwen3_5MTP.get_top_tokens(instance, hidden_states, spec_step_idx=0)

        assert result.numel() == 0
