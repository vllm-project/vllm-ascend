from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import AscendSampler, AscendTopKTopPSampler


class TestAscendSampler(TestBase):
    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)


class TestNpuApplyTopKTopP(TestBase):
    """Test that npu_apply_top_k_top_p custom op is registered."""

    def test_op_registered(self):
        import torch
        op = getattr(torch.ops._C_ascend, "npu_apply_top_k_top_p", None)
        self.assertIsNotNone(op, "npu_apply_top_k_top_p op should be registered")
