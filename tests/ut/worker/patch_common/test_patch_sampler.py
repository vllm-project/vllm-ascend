import importlib
import os
import unittest
from unittest import mock

import torch
from vllm.v1.sample.ops import topk_topp_sampler


class TestTopKTopPSamplerOptimize(unittest.TestCase):

    @mock.patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE": "1"})
    @mock.patch("torch_npu.npu_top_k_top_p")
    def test_npu_topk_topp_called_when_optimized(self, mock_npu_op):
        import vllm_ascend.patch.worker.patch_common.patch_sampler
        importlib.reload(vllm_ascend.patch.worker.patch_common.patch_sampler)

        mock_npu_op.return_value = (torch.randn(1, 3))
        sampler = topk_topp_sampler.TopKTopPSampler()

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        k = torch.tensor([2])
        p = torch.tensor([0.9])
        generators = {0: torch.Generator()}
        generators[0].manual_seed(42)

        sampler.forward_native(logits, generators, k, p)
        mock_npu_op.assert_called_once_with(logits, p, k)
