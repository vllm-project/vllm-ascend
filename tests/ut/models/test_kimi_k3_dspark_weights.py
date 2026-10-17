# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm_ascend.models import kimi_k3_dspark, qwen3_dspark


@pytest.fixture
def draft(monkeypatch):
    model_cls = kimi_k3_dspark.AscendK3DSparkForCausalLM
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=4, hidden_size=2)
    model.draft_model_config = SimpleNamespace(dtype=torch.float32)
    model.model = nn.Module()
    model.model.context_proj = nn.Linear(2, 2, bias=False)
    model.lm_head = None
    model._owns_embed_tokens = False
    model.has_own_lm_head = False

    # Consume the generator as AutoWeightsLoader does, retaining the weights
    # passed to it so interception can be checked independently of mapping.
    forwarded = []

    def load_weights(weights, *, mapper):
        forwarded.extend(weights)
        return {name for name, _ in forwarded}

    loader = MagicMock()
    loader.load_weights.side_effect = load_weights
    monkeypatch.setattr(kimi_k3_dspark, "AutoWeightsLoader", lambda model: loader)
    return model, forwarded


@pytest.mark.parametrize("name", ["lm_head.weight", "model.lm_head.weight", "language_model.lm_head.weight"])
@pytest.mark.parametrize("sharded", [False, True])
def test_checkpoint_head_is_loaded_before_mapping(draft, monkeypatch, name, sharded):
    model, forwarded = draft
    checkpoint_head = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    projection = torch.ones(2, 2)
    shard_loader = MagicMock()

    def make_head(vocab_size, hidden_size, *, params_dtype, prefix):
        assert (vocab_size, hidden_size, prefix) == (4, 2, "lm_head")
        assert params_dtype == model.draft_model_config.dtype
        head = nn.Linear(hidden_size, 2 if sharded else vocab_size, bias=False, dtype=params_dtype)
        if sharded:

            def load_shard(param, weight):
                with torch.no_grad():
                    param.copy_(weight[2:])

            shard_loader.side_effect = load_shard
            head.weight.weight_loader = shard_loader
        return head

    monkeypatch.setattr(kimi_k3_dspark, "ParallelLMHead", make_head)
    # A different ambient device catches accidental construction on the
    # caller's device instead of the draft projection's device.
    with torch.device("meta"):
        loaded = model.load_weights(iter([(name, checkpoint_head), ("context_proj.weight", projection)]))

    assert loaded == {"lm_head.weight", "context_proj.weight"}
    assert [name for name, _ in forwarded] == ["context_proj.weight"]
    assert forwarded[0][1] is projection
    assert model.has_own_lm_head
    assert model.lm_head.weight.device == model.model.context_proj.weight.device
    expected = checkpoint_head[2:] if sharded else checkpoint_head
    torch.testing.assert_close(model.lm_head.weight, expected)
    if sharded:
        shard_loader.assert_called_once_with(model.lm_head.weight, checkpoint_head)


@pytest.mark.parametrize("empty", [False, True])
def test_missing_checkpoint_head_preserves_target_fallback(draft, monkeypatch, empty):
    model, forwarded = draft
    make_head = MagicMock()
    monkeypatch.setattr(kimi_k3_dspark, "ParallelLMHead", make_head)
    weights = [] if empty else [("context_proj.weight", torch.ones(2, 2))]

    loaded = model.load_weights(iter(weights))

    assert loaded == {name for name, _ in weights}
    assert len(forwarded) == len(weights)
    assert model.lm_head is None
    assert not model.has_own_lm_head
    make_head.assert_not_called()


@pytest.mark.parametrize("shape", [(3, 2), (4, 3), (8,)])
def test_checkpoint_head_rejects_incompatible_shape(draft, monkeypatch, shape):
    model, _ = draft
    make_head = MagicMock()
    monkeypatch.setattr(kimi_k3_dspark, "ParallelLMHead", make_head)

    with pytest.raises(ValueError, match="does not match"):
        model.load_weights(iter([("lm_head.weight", torch.empty(shape))]))

    assert not model.has_own_lm_head
    make_head.assert_not_called()


@pytest.mark.parametrize("second_name", ["lm_head.weight", "model.lm_head.weight"])
def test_checkpoint_head_rejects_duplicates(draft, second_name):
    model, _ = draft
    weight = torch.ones(4, 2)

    with pytest.raises(ValueError, match="Multiple K3 DSpark checkpoint LM heads"):
        model.load_weights(iter([("lm_head.weight", weight), (second_name, weight)]))

    assert model.lm_head is None
    assert not model.has_own_lm_head


def test_checkpoint_head_survives_quarot_alignment(draft, monkeypatch):
    model, _ = draft
    monkeypatch.setattr(
        kimi_k3_dspark,
        "ParallelLMHead",
        lambda vocab_size, hidden_size, **kwargs: nn.Linear(hidden_size, vocab_size, bias=False),
    )
    checkpoint_head = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    model.load_weights(iter([("lm_head.weight", checkpoint_head)]))
    original_head = model.lm_head
    # Embedding ownership is independent; isolate the head and projection.
    model.has_own_embed_tokens = True
    projection_before = model.model.context_proj.weight.detach().clone()
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    monkeypatch.setattr(qwen3_dspark, "get_rotation_path", lambda config: "rotation.safetensors")
    monkeypatch.setattr(qwen3_dspark, "get_rotation_matrix", lambda path: rotation)
    target_loader = MagicMock()
    monkeypatch.setattr(qwen3_dspark, "load_quarot_target_layer", target_loader)
    config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace()))

    model.post_process(config)

    assert model.lm_head is original_head
    torch.testing.assert_close(model.lm_head.weight, checkpoint_head)
    torch.testing.assert_close(model.model.context_proj.weight, projection_before @ rotation)
    target_loader.assert_not_called()
