# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft checkpoint heads must win regardless of shard completion order."""

import random
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v4 import dspark

HEADS = {
    "embed.weight": ("mtp.0.embed.weight", "model.embed_tokens.weight"),
    "head.weight": ("mtp.2.head.weight", "lm_head.weight"),
    "hc_head_fn": ("mtp.2.hc_head_fn", "model.hc_head_fn"),
    "hc_head_base": ("mtp.2.hc_head_base", "model.hc_head_base"),
    "hc_head_scale": ("mtp.2.hc_head_scale", "model.hc_head_scale"),
}


def make_loader(monkeypatch, rotation_path=None):
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_rank", lambda: 0)
    params = {dest: torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False) for _, dest in HEADS.values()}
    model = SimpleNamespace(
        rotation_path=rotation_path,
        config=SimpleNamespace(num_attention_heads=8, num_hidden_layers=43),
        model=SimpleNamespace(num_dspark_layers=3, get_expert_mapping=lambda: []),
        named_parameters=lambda: params.items(),
    )
    model._remap_dspark_name = MethodType(dspark.DSparkDeepseekV4ForCausalLM._remap_dspark_name, model)
    return model, params


@pytest.mark.parametrize("order", ["target-first", "draft-first", *range(128)])
@pytest.mark.parametrize("rotation_path", [None, "rotation.bin"])
def test_mtp_heads_win_with_shuffled_shards(monkeypatch, order, rotation_path):
    model, params = make_loader(monkeypatch, rotation_path)
    weights = []
    for global_name, (draft_name, _) in HEADS.items():
        weights.extend([(global_name, torch.ones(2, 2)), (draft_name, torch.full((2, 2), 7.0))])
    if isinstance(order, int):
        random.Random(order).shuffle(weights)
    else:
        weights.sort(key=lambda item: item[0].startswith("mtp."), reverse=order == "draft-first")
    loaded = dspark.DSparkDeepseekV4ForCausalLM.load_weights(model, iter(weights))
    assert loaded == set(params)
    for param in params.values():
        torch.testing.assert_close(param, torch.full((2, 2), 7.0))


def test_global_heads_remain_fallbacks(monkeypatch):
    model, params = make_loader(monkeypatch)
    loaded = dspark.DSparkDeepseekV4ForCausalLM.load_weights(model, ((name, torch.full((2, 2), 3.0)) for name in HEADS))
    assert loaded == set(params)
    for param in params.values():
        torch.testing.assert_close(param, torch.full((2, 2), 3.0))


def test_partial_mtp_heads_preserve_other_fallbacks(monkeypatch):
    model, params = make_loader(monkeypatch)
    weights = [("mtp.2.head.weight", torch.full((2, 2), 7.0))]
    weights.extend((name, torch.ones(2, 2)) for name in HEADS)
    dspark.DSparkDeepseekV4ForCausalLM.load_weights(model, iter(weights))
    for name, param in params.items():
        torch.testing.assert_close(param, torch.full((2, 2), 7.0 if name == "lm_head.weight" else 1.0))
