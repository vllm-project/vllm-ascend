# SPDX-License-Identifier: Apache-2.0
# UTs for vllm_ascend/models/step5/vl.py -- CPU only.
#
# AutoWeightsLoader is patched in the module namespace so the two-stream
# load_weights routing (language model weights first, vision-tower weights
# accumulated and loaded in a second pass) is exercised without building
# the multimodal model.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.models.step5.vl as step5_vl
from vllm_ascend.models.step5.vl import AscendStep5ForConditionalGeneration


class _RecordingLoader:
    instances: list["_RecordingLoader"] = []

    def __init__(self, model):
        self.model = model
        self.calls = []
        type(self).instances.append(self)

    def load_weights(self, weights, mapper=None):
        names = [name for name, _ in weights]
        self.calls.append((names, mapper))
        return set(names)


@pytest.fixture
def loader_cls(monkeypatch):
    _RecordingLoader.instances = []
    monkeypatch.setattr(step5_vl, "AutoWeightsLoader", _RecordingLoader)
    return _RecordingLoader


def _model_with_mapper(map_name):
    model = object.__new__(AscendStep5ForConditionalGeneration)
    model.hf_to_vllm_mapper = SimpleNamespace(_map_name=map_name)
    return model


def test_load_weights_splits_language_stream_and_tower(loader_cls):
    map_name = MagicMock(side_effect=lambda name: name)
    model = _model_with_mapper(map_name)
    weights = [
        ("language_model.layers.0.self_attn.qkv_proj.weight", torch.zeros(1)),
        ("vision.blocks.0.fc1.weight", torch.zeros(1)),
        ("language_model.model.norm.weight", torch.zeros(1)),
        ("multi_modal_projector.weight", torch.zeros(1)),
    ]

    loaded = model.load_weights(iter(weights))

    # one loader, two passes: the language stream first, the tower second
    assert len(loader_cls.instances) == 1
    loader = loader_cls.instances[0]
    assert loader.model is model
    assert len(loader.calls) == 2
    for _, mapper in loader.calls:
        assert mapper is model.hf_to_vllm_mapper

    assert loader.calls[0][0] == [
        "language_model.layers.0.self_attn.qkv_proj.weight",
        "language_model.model.norm.weight",
    ]
    assert loader.calls[1][0] == [
        "vision.blocks.0.fc1.weight",
        "multi_modal_projector.weight",
    ]
    assert loaded == {
        "language_model.layers.0.self_attn.qkv_proj.weight",
        "language_model.model.norm.weight",
        "vision.blocks.0.fc1.weight",
        "multi_modal_projector.weight",
    }
    # routing is decided on the mapped name, once per weight
    assert map_name.call_count == 4


def test_load_weights_mapped_prefix_decides_the_stream(loader_cls):
    # checkpoint "model.language_model.*" maps to "language_model.model.*":
    # still routed to the language stream even though the raw prefix differs.
    mapping = {"model.language_model.foo.weight": "language_model.model.foo.weight"}
    model = _model_with_mapper(lambda name: mapping.get(name, name))
    weights = [
        ("model.language_model.foo.weight", torch.zeros(1)),
        ("model.vision.bar.weight", torch.zeros(1)),
    ]

    loaded = model.load_weights(iter(weights))

    loader = loader_cls.instances[0]
    assert len(loader.calls) == 2
    assert loader.calls[0][0] == ["model.language_model.foo.weight"]
    assert loader.calls[1][0] == ["model.vision.bar.weight"]
    assert loaded == {"model.language_model.foo.weight", "model.vision.bar.weight"}


def test_load_weights_without_tower_weights_single_pass(loader_cls):
    model = _model_with_mapper(lambda name: name)
    weights = [
        ("language_model.a.weight", torch.zeros(1)),
        ("language_model.b.weight", torch.zeros(1)),
    ]

    loaded = model.load_weights(iter(weights))

    assert len(loader_cls.instances) == 1
    loader = loader_cls.instances[0]
    assert len(loader.calls) == 1
    assert loader.calls[0][0] == ["language_model.a.weight", "language_model.b.weight"]
    assert loaded == {"language_model.a.weight", "language_model.b.weight"}
