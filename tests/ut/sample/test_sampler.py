from unittest import mock

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import AscendSampler, AscendTopKTopPSampler


class TestAscendSampler(TestBase):

    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, 'topk_topp_sampler'))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)


class TestAscendTopKTopPSampler(TestBase):

    @mock.patch("torch_npu.npu_top_k_top_p")
    def test_npu_topk_topp_called_when_optimized(self, mock_npu_op):
        mock_npu_op.return_value = (torch.randn(1, 3))
        sampler = AscendTopKTopPSampler()

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        k = torch.tensor([2])
        p = torch.tensor([0.9])
        generators = {0: torch.Generator()}
        generators[0].manual_seed(42)

        sampler.forward_native(logits, generators, k, p)
        mock_npu_op.assert_called_once_with(logits, p, k)
