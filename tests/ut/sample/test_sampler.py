from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import AscendSampler, AscendTopKTopPSampler


class TestAscendSampler(TestBase):
    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)

    def test_external_q_is_consumed(self):
        sampler = AscendTopKTopPSampler()
        logits = torch.tensor([[2.0, 0.5, -1.0]], dtype=torch.float32)
        external_q = torch.tensor([[0.5, 10.0, 10.0]], dtype=torch.float32)

        sampler.set_external_q(external_q)
        with patch(
            "vllm_ascend.sample.sampler.get_ascend_config",
            return_value=SimpleNamespace(enable_reduce_sample=False, enable_async_exponential=False),
        ):
            next_token, _ = sampler.forward_native(logits, generators={}, k=None, p=None)

        self.assertEqual(next_token.item(), 0)
        self.assertIsNone(sampler.external_q)
