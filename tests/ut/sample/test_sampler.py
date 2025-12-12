from unittest import mock

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import (AscendSampler, AscendSamplingMetadata,
                                        AscendTopKTopPSampler)


class TestAscendSampler(TestBase):

    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, 'topk_topp_sampler'))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)


class TestAscendTopKTopPSampler(TestBase):

    @mock.patch("vllm_ascend.sample.sampler.random_sample")
    @mock.patch("torch_npu.npu_top_k_top_p")
    def test_npu_topk_topp_called_when_optimized(self, mock_npu_op,
                                                 mock_random_sample):
        mock_npu_op.return_value = (torch.randn(1, 3))
        mock_random_sample.return_value = torch.randn(3)
        sampler = AscendTopKTopPSampler()

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        sampling_metadata = mock.Mock(spec=AscendSamplingMetadata)
        sampling_metadata.top_k = torch.tensor([2])
        sampling_metadata.top_k_cpu = sampling_metadata.top_k
        sampling_metadata.top_p = torch.tensor([0.9])
        generators = {0: torch.Generator()}
        generators[0].manual_seed(42)
        sampling_metadata.generators = generators

        sampler.forward_native(logits, sampling_metadata)
        mock_npu_op.assert_called_once_with(logits, sampling_metadata.top_p,
                                            sampling_metadata.top_k)
