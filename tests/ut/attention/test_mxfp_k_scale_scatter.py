# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_c8_mxfp import scatter_mxfp_k_scale_cache


class TestKScaleScatterDispatch(TestBase):
    def test_cpu_scatter_preserves_unselected_elements(self):
        cache = torch.full((3, 2, 2, 4, 16, 2), 13, dtype=torch.uint8)
        expected = cache.clone()
        slots = (torch.tensor([1, 2]), torch.tensor([0, 1]), torch.tensor([3, 9]))
        values = torch.arange(32, dtype=torch.uint8).reshape(2, 2, 4, 2)
        expected[1, :, 0, :, 3, :] = values[0]
        expected[2, :, 1, :, 9, :] = values[1]
        scatter_mxfp_k_scale_cache(values, cache, slots)
        torch.testing.assert_close(cache, expected)

    def test_npu_dispatch_uses_targeted_scatter(self):
        module = ModuleType("vllm_ascend.ops.triton.mxfp_k_scale")
        scatter = MagicMock()
        vars(module)["scatter_k_scales"] = scatter
        cache = SimpleNamespace(device=SimpleNamespace(type="npu"))
        values = torch.ones((1, 1, 4, 2), dtype=torch.uint8)
        slots = tuple(torch.tensor([1]) for _ in range(3))
        with patch.dict("sys.modules", {module.__name__: module}):
            scatter_mxfp_k_scale_cache(values, cache, slots)
        scatter.assert_called_once_with(values, cache, slots)

    def test_empty_scatter_does_not_access_cache(self):
        slots = tuple(torch.empty(0, dtype=torch.int64) for _ in range(3))
        scatter_mxfp_k_scale_cache(torch.empty((0, 1, 4, 2), dtype=torch.uint8), None, slots)
