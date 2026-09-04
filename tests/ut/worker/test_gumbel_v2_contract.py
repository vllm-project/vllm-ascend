# SPDX-License-Identifier: Apache-2.0

from inspect import signature
from unittest.mock import MagicMock

import torch

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.sample import gumbel


def test_gumbel_sample_preserves_lane_argument_order(monkeypatch):
    logits, indices, temperature, seed, pos, cache, col = [torch.empty(1) for _ in range(7)]
    if vllm_version_is("0.27.1"):
        bound = signature(gumbel.gumbel_sample).bind(logits, indices, temperature, seed, pos, True, cache, col, False)
        assert bound.arguments["output_processed_logits"] is cache
        assert bound.arguments["output_processed_logits_col"] is col
        assert bound.arguments["use_fp64"] is False
    else:
        implementation = MagicMock(return_value=indices)
        monkeypatch.setattr(gumbel, "_gumbel_sample", implementation)
        result = gumbel.gumbel_sample(logits, indices, temperature, seed, pos, True, True, cache, col, False)
        assert result is indices
        implementation.assert_called_once_with(
            logits, indices, temperature, seed, pos, True, cache, col, False, is_drafting=True
        )
