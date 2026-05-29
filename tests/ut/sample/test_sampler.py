from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import (
    AscendSampler,
    AscendTopKTopPSampler,
    sample_with_runtime_state,
)


class TestAscendSampler(TestBase):
    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)

    @patch("vllm_ascend.sample.sampler.apply_top_k_top_p")
    def test_runtime_sampler_skips_topk_topp_when_unconstrained(
        self,
        mock_apply_top_k_top_p,
    ):
        logits = torch.tensor([[1.0, 3.0], [4.0, 2.0]])
        sampled = sample_with_runtime_state(
            logits,
            idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
            positions=torch.tensor([10, 11], dtype=torch.int32),
            temperature=torch.ones(2, dtype=torch.float32),
            top_k=None,
            top_p=None,
            seeds=torch.zeros(2, dtype=torch.int64),
        )

        mock_apply_top_k_top_p.assert_not_called()
        self.assertTrue(torch.equal(sampled, torch.tensor([1, 0])))
