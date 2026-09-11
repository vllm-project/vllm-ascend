# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import torch

from vllm_ascend.worker.v2.sample import gumbel


def test_gumbel_sample_preserves_main_argument_order(monkeypatch):
    logits, indices, temperature, seed, pos, cache, col = [torch.empty(1) for _ in range(7)]
    implementation = MagicMock(return_value=indices)
    monkeypatch.setattr(gumbel, "_gumbel_sample", implementation)
    result = gumbel.gumbel_sample(logits, indices, temperature, seed, pos, True, True, cache, col, False)
    assert result is indices
    implementation.assert_called_once_with(
        logits, indices, temperature, seed, pos, True, cache, col, False, is_drafting=True
    )
