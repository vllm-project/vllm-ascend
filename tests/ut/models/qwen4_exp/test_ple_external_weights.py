# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.models.qwen4_exp.ple_external_weights import ExternalPLEWeightsMixin


class BaseModel(torch.nn.Module):
    def is_unused_checkpoint_weight(self, name):
        return name == "parent.unused"


class Model(ExternalPLEWeightsMixin, BaseModel):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Identity()


def test_filter_is_active_only_after_attachment_and_preserves_parent_rule():
    model = Model()
    assert not model.filter_unused_checkpoint_weights
    assert model.is_unused_checkpoint_weight("parent.unused")
    assert not model.is_unused_checkpoint_weight("ple.shard_0.weight")
    model.embedding.ple_external_checkpoint_names = frozenset({"ple.shard_0.weight"})
    assert model.filter_unused_checkpoint_weights
    assert model.is_unused_checkpoint_weight("ple.shard_0.weight")
    assert not model.is_unused_checkpoint_weight("ple.shard_1.weight")
    assert model.is_unused_checkpoint_weight("parent.unused")


def test_attached_table_requires_pre_materialization_filter(monkeypatch):
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    model = Model()
    model.embedding.ple_external_checkpoint_names = frozenset({"ple.shard_0.weight"})

    class OldSource:
        __dataclass_fields__ = {}

    monkeypatch.setattr(DefaultModelLoader, "Source", OldSource)
    with pytest.raises(RuntimeError, match="per-tensor checkpoint filtering"):
        _ = model.filter_unused_checkpoint_weights


def test_attached_names_filter_mixed_shard_before_tensor_materialization(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import DefaultModelLoader, weight_utils

    external = "ple.shard_0.weight"
    save_file({external: torch.ones(2), "model.weight": torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    model = Model()
    model.embedding.ple_external_checkpoint_names = frozenset({external})
    seen = []
    original = weight_utils.safe_open

    class TrackingOpen:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def keys(self):
            return self.inner.keys()

        def get_tensor(self, name):
            seen.append(name)
            return self.inner.get_tensor(name)

    monkeypatch.setattr(weight_utils, "safe_open", TrackingOpen)
    loader = DefaultModelLoader(LoadConfig(safetensors_load_strategy="lazy"))
    result = dict(loader.get_all_weights(SimpleNamespace(model=str(tmp_path), revision=None), model))
    assert list(result) == ["model.weight"]
    assert seen == ["model.weight"]
