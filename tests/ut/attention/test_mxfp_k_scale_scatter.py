# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import ModuleType
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_c8_mxfp import scatter_mxfp_k_scale_cache


class TestKScaleScatterDispatch(TestBase):
    def test_nonempty_scatter_delegates_without_accessing_the_cache(self):
        module = ModuleType("vllm_ascend.ops.triton.mxfp_k_scale")
        scatter = MagicMock()
        vars(module)["scatter_k_scales"] = scatter
        cache = object()
        values = torch.ones((1, 1, 4, 2), dtype=torch.uint8)
        slots = tuple(torch.tensor([1]) for _ in range(3))
        with patch.dict("sys.modules", {module.__name__: module}):
            scatter_mxfp_k_scale_cache(values, cache, slots)
        scatter.assert_called_once_with(values, cache, slots)

    def test_empty_scatter_does_not_import_the_kernel_or_access_the_cache(self):
        slots = tuple(torch.empty(0, dtype=torch.int64) for _ in range(3))
        with patch.dict("sys.modules", {"vllm_ascend.ops.triton.mxfp_k_scale": None}):
            scatter_mxfp_k_scale_cache(torch.empty((0, 1, 4, 2), dtype=torch.uint8), None, slots)
