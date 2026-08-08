from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import (
    AscendSampler,
    AscendTopKTopPSampler,
    _apply_top_k_top_p_torch_npu,
)


class TestAscendSampler(TestBase):
    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)


class TestApplyTopKTopPTorchNpuNonReduce(TestBase):
    """Test _apply_top_k_top_p_torch_npu in non-reduce_sample mode.

    In non-reduce mode, the function delegates to _apply_top_k_top_p_pytorch
    (sort+mask) instead of npu_top_k_top_p, because the sort-based approach
    is consistently ~1ms regardless of k value, while npu_top_k_top_p
    degrades to 5-28ms when k is large or batch contains mixed k values.
    """

    @patch("vllm_ascend.sample.sampler.get_ascend_config")
    def test_both_k_and_p_none_returns_logits(self, mock_config):
        """When k=None and p=None, logits should be returned unchanged."""
        mock_config.return_value.enable_reduce_sample = False
        logits = torch.tensor([[1.0, 3.0, 2.0, 5.0, 4.0]])
        result = _apply_top_k_top_p_torch_npu(logits, None, None)
        assert torch.equal(result, logits)

    @patch("vllm_ascend.sample.sampler.get_ascend_config")
    def test_top_k_only_filters_correctly(self, mock_config):
        """Top-k only filtering should keep top-k values, mask rest with -inf."""
        mock_config.return_value.enable_reduce_sample = False
        logits = torch.tensor([[1.0, 3.0, 2.0, 5.0, 4.0]])
        k = torch.tensor([2])
        result = _apply_top_k_top_p_torch_npu(logits, k, None)

        # Top-2 values: 5.0(idx=3), 4.0(idx=4)
        assert result[0, 3].item() == 5.0
        assert result[0, 4].item() == 4.0
        assert result[0, 0].item() == -float("inf")
        assert result[0, 1].item() == -float("inf")
        assert result[0, 2].item() == -float("inf")

    @patch("vllm_ascend.sample.sampler.get_ascend_config")
    def test_top_k_eq_vocab_no_filtering(self, mock_config):
        """When k == V (no restriction), all logits should be kept."""
        mock_config.return_value.enable_reduce_sample = False
        logits = torch.tensor([[1.0, 3.0, 2.0]])
        k = torch.tensor([3])  # k == V
        result = _apply_top_k_top_p_torch_npu(logits, k, None)
        # No filtering: all values kept (no -inf)
        assert not torch.isinf(result).any()

    @patch("vllm_ascend.sample.sampler.get_ascend_config")
    def test_mixed_k_batch(self, mock_config):
        """Mixed k values in batch: each row filtered by its own k."""
        mock_config.return_value.enable_reduce_sample = False
        logits = torch.tensor(
            [
                [1.0, 3.0, 2.0, 5.0, 4.0],
                [10.0, 50.0, 30.0, 20.0, 40.0],
            ]
        )
        k = torch.tensor([2, 3])
        result = _apply_top_k_top_p_torch_npu(logits, k, None)

        # Row 0: top-2 = 5.0, 4.0
        valid_row0 = result[0] != -float("inf")
        assert valid_row0.sum().item() == 2

        # Row 1: top-3 = 50.0, 40.0, 30.0
        valid_row1 = result[1] != -float("inf")
        assert valid_row1.sum().item() == 3

    @patch("vllm_ascend.sample.sampler.get_ascend_config")
    def test_top_p_only_filters_correctly(self, mock_config):
        """Top-p only filtering should keep tokens above probability threshold."""
        mock_config.return_value.enable_reduce_sample = False
        logits = torch.tensor([[1.0, 3.0, 2.0, 5.0, 4.0]])
        p = torch.tensor([0.5])
        result = _apply_top_k_top_p_torch_npu(logits, None, p)

        # At least one token should remain
        valid_count = (result[0] != -float("inf")).sum().item()
        assert valid_count >= 1

    @patch("vllm_ascend.sample.sampler.get_ascend_config")
    def test_p_eq_one_no_filtering(self, mock_config):
        """When p=1.0 (no restriction), all logits should be kept."""
        mock_config.return_value.enable_reduce_sample = False
        logits = torch.tensor([[1.0, 3.0, 2.0]])
        p = torch.tensor([1.0])
        result = _apply_top_k_top_p_torch_npu(logits, None, p)
        assert not torch.isinf(result).any()

    @patch("vllm_ascend.sample.sampler.get_ascend_config")
    def test_both_k_and_p_present(self, mock_config):
        """Both k and p filtering should work together."""
        mock_config.return_value.enable_reduce_sample = False
        logits = torch.tensor([[1.0, 3.0, 2.0, 5.0, 4.0]])
        k = torch.tensor([3])
        p = torch.tensor([0.9])
        result = _apply_top_k_top_p_torch_npu(logits, k, p)

        # Should have at most 3 valid tokens (top-k limit)
        valid_count = (result[0] != -float("inf")).sum().item()
        assert valid_count <= 3
        assert valid_count >= 1


class TestApplyTopKTopPTorchNpuReduceSample(TestBase):
    """Test _apply_top_k_top_p_torch_npu in reduce_sample mode."""

    def _make_mock_tp_group(self, world_size=1, rank=0):
        tp_group = MagicMock()
        tp_group.world_size = world_size
        tp_group.rank_in_group = rank
        tp_group.all_gather = MagicMock(side_effect=lambda x, dim=-1: x)
        return tp_group

    @patch("vllm_ascend.sample.sampler.get_tp_group")
    @patch("vllm_ascend.sample.sampler.get_ascend_config")
    @patch("vllm_ascend.sample.sampler.torch_npu")
    def test_reduce_both_none_skips_npu(self, mock_torch_npu, mock_config, mock_tp_group):
        """When k=None and p=None in reduce mode, npu_top_k_top_p is skipped."""
        mock_config.return_value.enable_reduce_sample = True
        mock_tp_group.return_value = self._make_mock_tp_group()

        logits = torch.tensor([[1.0, 3.0, 2.0]])
        vals, idx = _apply_top_k_top_p_torch_npu(logits, None, None)

        # npu_top_k_top_p should NOT be called
        mock_torch_npu.npu_top_k_top_p.assert_not_called()
        assert vals is not None
        assert idx is not None

    @patch("vllm_ascend.sample.sampler.get_tp_group")
    @patch("vllm_ascend.sample.sampler.get_ascend_config")
    @patch("vllm_ascend.sample.sampler.torch_npu")
    def test_reduce_with_k_calls_npu(self, mock_torch_npu, mock_config, mock_tp_group):
        """When k is provided in reduce mode, npu_top_k_top_p is called."""
        mock_config.return_value.enable_reduce_sample = True
        mock_tp_group.return_value = self._make_mock_tp_group()
        mock_torch_npu.npu_top_k_top_p = MagicMock(side_effect=lambda x, k, p: x)

        logits = torch.tensor([[1.0, 3.0, 2.0]])
        k = torch.tensor([2])
        vals, idx = _apply_top_k_top_p_torch_npu(logits, k, None)

        mock_torch_npu.npu_top_k_top_p.assert_called_once()

    @patch("vllm_ascend.sample.sampler.get_tp_group")
    @patch("vllm_ascend.sample.sampler.get_ascend_config")
    @patch("vllm_ascend.sample.sampler.torch_npu")
    def test_reduce_with_p_calls_npu(self, mock_torch_npu, mock_config, mock_tp_group):
        """When p is provided in reduce mode, npu_top_k_top_p is called."""
        mock_config.return_value.enable_reduce_sample = True
        mock_tp_group.return_value = self._make_mock_tp_group()
        mock_torch_npu.npu_top_k_top_p = MagicMock(side_effect=lambda x, k, p: x)

        logits = torch.tensor([[1.0, 3.0, 2.0]])
        p = torch.tensor([0.9])
        vals, idx = _apply_top_k_top_p_torch_npu(logits, None, p)

        mock_torch_npu.npu_top_k_top_p.assert_called_once()
