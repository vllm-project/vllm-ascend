# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen3_dspark import AscendQwen3DSparkForCausalLM


@pytest.mark.parametrize(
    "logits, expected",
    [
        ([0.0, 2.0, -2.0], [0.5, 0.880797, 0.119203]),
        ([float("nan"), float("inf"), -float("inf")], [1.0, 1.0, 0.0]),
    ],
)
def test_adaptive_confidence_is_finite_without_changing_valid_values(logits, expected):
    model = AscendQwen3DSparkForCausalLM.__new__(AscendQwen3DSparkForCausalLM)
    torch.nn.Module.__init__(model)
    model.enable_confidence_head = True
    model.model = SimpleNamespace(confidence_head=lambda hidden, markov: torch.tensor(logits))

    probabilities = model.compute_confidence(torch.empty(3), torch.empty(3))

    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities, torch.tensor(expected), atol=1e-6, rtol=0)
