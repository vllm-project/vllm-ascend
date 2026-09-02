# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the GLM-5 DSpark model adapter."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.models.glm5_dspark import Glm5DSparkForCausalLM


def test_load_weights_normalizes_reduced_vocab_mapping() -> None:
    model = SimpleNamespace(
        rotation_path=None,
        hf_to_vllm_mapper=Glm5DSparkForCausalLM.hf_to_vllm_mapper,
        has_own_embed_tokens=False,
        has_own_lm_head=False,
        enable_confidence_head=True,
    )
    loader = MagicMock()
    loader.load_weights.return_value = {"draft_id_to_target_id"}
    d2t = torch.tensor([0, 2, 4], dtype=torch.long)
    weights = [
        ("t2d", torch.tensor([0, 1, 1], dtype=torch.long)),
        ("d2t", d2t),
        ("norm.weight", torch.ones(4)),
        ("lm_head.weight", torch.ones(3, 4)),
    ]

    with patch("vllm_ascend.models.glm5_dspark.AutoWeightsLoader", return_value=loader):
        loaded = Glm5DSparkForCausalLM.load_weights(model, weights)

    normalized_weights = loader.load_weights.call_args.args[0]
    assert [name for name, _ in normalized_weights] == [
        "draft_id_to_target_id",
        "model.final_norm.weight",
        "lm_head.weight",
    ]
    assert normalized_weights[0][1] is d2t
    assert loaded == {"draft_id_to_target_id"}
