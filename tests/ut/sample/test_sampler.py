from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import AscendSampler, AscendTopKTopPSampler


class TestAscendSampler(TestBase):

    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)
        self.assertEqual(sampler.topk_topp_sampler.logprobs_mode, "raw_logprobs")

    def test_init_with_processed_logits(self):
        sampler = AscendSampler(logprobs_mode="processed_logits")
        self.assertEqual(sampler.logprobs_mode, "processed_logits")
        self.assertEqual(sampler.topk_topp_sampler.logprobs_mode, "processed_logits")

    def test_init_with_processed_logprobs(self):
        sampler = AscendSampler(logprobs_mode="processed_logprobs")
        self.assertEqual(sampler.logprobs_mode, "processed_logprobs")
        self.assertEqual(sampler.topk_topp_sampler.logprobs_mode, "processed_logprobs")
