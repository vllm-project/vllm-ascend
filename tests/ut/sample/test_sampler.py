import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import TestBase
from tests.ut.conftest import _npu_available
from vllm_ascend.sample.sampler import (
    AscendSampler,
    AscendTopKTopPSampler,
    apply_top_k_top_p,
    generate_random_sequence,
)


class TestAscendSampler(TestBase):
    def setUp(self):
        if not _npu_available:
            self.stream_patcher = patch("torch_npu.npu.Stream")
            self.stream_patcher.start()

    def tearDown(self):
        if not _npu_available:
            self.stream_patcher.stop()

    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)


def _mock_ascend_config(enable_reduce_sample=False):
    """Create a mock ascend config for tests that need get_ascend_config()."""
    mock_config = MagicMock()
    mock_config.enable_reduce_sample = enable_reduce_sample
    return mock_config


class TestAscendTopKTopPSampler(TestBase):
    """Test that sampler patches are correctly applied in vllm_ascend source."""

    @pytest.mark.skipif(not _npu_available, reason="NPU not available")
    def test_ascend_top_k_top_p_sampler_has_dsa_stream(self):
        """AscendTopKTopPSampler should have dsa_stream attribute."""
        instance = AscendTopKTopPSampler(logprobs_mode="raw_logprobs")
        self.assertTrue(hasattr(instance, "dsa_stream"))
        self.assertTrue(hasattr(instance, "logprobs_mode"))

    @patch("vllm_ascend.sample.sampler.get_ascend_config", return_value=_mock_ascend_config())
    @patch("vllm_ascend.sample.sampler.generate_random_sequence")
    @patch("vllm_ascend.sample.sampler.torch_npu.npu_top_k_top_p_sample")
    @pytest.mark.skipif(not _npu_available, reason="NPU not available")
    def test_forward_native_logprobs_modes(self, mock_sample, mock_gen_rand, mock_get_cfg):
        """forward_native should handle all logprobs_mode variants."""
        batch_size, vocab_size = 2, 8
        logits = torch.randn(batch_size, vocab_size)
        k = torch.tensor([5] * batch_size, dtype=torch.int32)
        p = torch.tensor([0.9] * batch_size, dtype=torch.bfloat16)
        mock_sample.return_value = (
            torch.zeros(batch_size, dtype=torch.int64),
            logits.type(torch.float32),
        )
        mock_gen_rand.return_value = torch.randn(batch_size, vocab_size, dtype=torch.float32)

        for mode in ["raw_logprobs", "processed_logits", "processed_logprobs"]:
            with self.subTest(mode=mode):
                instance = AscendTopKTopPSampler(logprobs_mode=mode)
                sampled, logprobs = instance.forward_native(logits, {}, k, p)
                self.assertEqual(sampled.shape, (batch_size,))
                self.assertEqual(sampled.dtype, torch.int64)

    @patch("vllm_ascend.sample.sampler.get_ascend_config", return_value=_mock_ascend_config())
    @patch("vllm_ascend.sample.sampler.generate_random_sequence")
    @patch("vllm_ascend.sample.sampler.torch_npu.npu_top_k_top_p_sample")
    @pytest.mark.skipif(not _npu_available, reason="NPU not available")
    def test_forward_native_default_k_p(self, mock_sample, mock_gen_rand, mock_get_cfg):
        """forward_native with k=None or p=None should use defaults."""
        batch_size, vocab_size = 2, 8
        logits = torch.randn(batch_size, vocab_size)
        mock_sample.return_value = (
            torch.zeros(batch_size, dtype=torch.int64),
            logits.type(torch.float32),
        )
        mock_gen_rand.return_value = torch.randn(batch_size, vocab_size, dtype=torch.float32)

        instance = AscendTopKTopPSampler(logprobs_mode="raw_logprobs")

        for k, p, label in [
            (None, torch.tensor([0.9] * batch_size, dtype=torch.bfloat16), "k=None"),
            (torch.tensor([5] * batch_size, dtype=torch.int32), None, "p=None"),
        ]:
            with self.subTest(label=label):
                sampled, logprobs = instance.forward_native(logits, {}, k, p)
                self.assertEqual(sampled.shape, (batch_size,))
                # self.assertEqual(sampled.dtype, torch.bfloat16)

    @pytest.mark.skipif(not _npu_available, reason="NPU not available")
    def test_ascend_sampler_init_creates_topk_topp_sampler(self):
        """AscendSampler.__init__ should create topk_topp_sampler."""
        instance = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertTrue(hasattr(instance, "topk_topp_sampler"))

    @patch("vllm_ascend.sample.sampler.get_ascend_config", return_value=_mock_ascend_config())
    @pytest.mark.skipif(not _npu_available, reason="NPU not available")
    def test_apply_top_k_top_p(self, mock_get_cfg):
        """apply_top_k_top_p should handle all k/p combinations."""
        batch_size, vocab_size = 4, 16
        logits = torch.randn(batch_size, vocab_size, dtype=torch.bfloat16, device="npu")

        for k, p, label in [
            (torch.tensor([3] * batch_size, dtype=torch.int32, device="npu"), None, "k only"),
            (None, torch.tensor([0.9] * batch_size, dtype=torch.bfloat16, device="npu"), "p only"),
            (
                torch.tensor([5] * batch_size, dtype=torch.int32, device="npu"),
                torch.tensor([0.95] * batch_size, dtype=torch.bfloat16, device="npu"),
                "k and p",
            ),
            (None, None, "neither"),
        ]:
            with self.subTest(label=label):
                result = apply_top_k_top_p(logits, k, p)
                self.assertEqual(result.shape, logits.shape)
                expected_dtype = torch.bfloat16 if (k is None and p is None) else torch.float32
                self.assertEqual(result.dtype, expected_dtype)
                if k is None and p is None:
                    torch.testing.assert_close(result, logits, msg="k=None, p=None should be identity")

    @pytest.mark.skipif(not _npu_available, reason="NPU not available")
    def test_ascend_top_k_top_p_sampler_set_q_event(self):
        """set_q_event should store q and event."""
        instance = AscendTopKTopPSampler(logprobs_mode="raw_logprobs")
        q = torch.randn(2, 4)
        event = torch.npu.Event()
        instance.set_q_event(q, event)
        self.assertIs(instance.q, q)
        self.assertIs(instance.async_event, event)

    @pytest.mark.skipif(not _npu_available, reason="NPU not available")
    def test_generate_random_sequence(self):
        """generate_random_sequence should return a tensor of the same shape."""
        logits = torch.randn(2, 4)
        generators: dict[int, torch.Generator] = {}
        stream = torch.npu.Stream()
        result = generate_random_sequence(logits, generators, stream)
        self.assertEqual(result.shape, logits.shape)
        self.assertEqual(result.dtype, torch.float32)

    @pytest.mark.skipif(not _npu_available, reason="NPU not available")
    def test_generate_random_sequence_with_generators(self):
        """generate_random_sequence with non-empty generators."""
        logits = torch.randn(2, 4)
        gen0 = torch.Generator()
        gen1 = torch.Generator()
        generators = {0: gen0, 1: gen1}
        stream = torch.npu.Stream()
        result = generate_random_sequence(logits, generators, stream)
        self.assertEqual(result.shape, logits.shape)
        self.assertEqual(result.dtype, torch.float32)

    @patch("vllm_ascend.sample.sampler.get_ascend_config", return_value=_mock_ascend_config())
    def test_ascend_sampler_greedy_sample(self, mock_get_cfg):
        """AscendSampler.greedy_sample returns argmax over logits."""
        logits = torch.randn(3, 10)
        result = AscendSampler.greedy_sample(logits)
        expected = logits.argmax(dim=-1).view(-1)
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
