import importlib
import os
from unittest import mock

import torch
from vllm.v1.sample.ops import topk_topp_sampler

from tests.ut.base import TestBase


class TestTopKTopPSamplerOptimize(TestBase):

    @mock.patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE": "1"})
    @mock.patch("torch_npu.npu_top_k_top_p")
    def test_npu_topk_topp_called_when_optimized(self, mock_npu_op):
        # We have to patch and reload because the patch will take effect
        # only after VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE is set.
        import vllm_ascend.patch.worker.patch_0_9_1.patch_sampler
        importlib.reload(vllm_ascend.patch.worker.patch_0_9_1.patch_sampler)

        mock_npu_op.return_value = (torch.randn(1, 3))
        sampler = topk_topp_sampler.TopKTopPSampler()

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        k = torch.tensor([2])
        p = torch.tensor([0.9])
        generators = {0: torch.Generator()}
        generators[0].manual_seed(42)

        sampler.forward_native(logits, generators, k, p)
        mock_npu_op.assert_called_once_with(logits, p, k)
