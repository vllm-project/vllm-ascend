import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

PATCH = Path(__file__).parents[3] / "vllm_ascend/patch/worker/patch_external_weight_filter.py"


def load_patch(native=False):
    class Source:
        __dataclass_fields__ = {"filter_unused_per_tensor": None} if native else {}

        def __init__(
            self,
            model_or_path,
            revision,
            subfolder=None,
            prefix="",
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
            is_unused_weight=None,
        ):
            self.model_or_path, self.revision, self.subfolder, self.prefix = model_or_path, revision, subfolder, prefix
            self.fall_back_to_pt, self.allow_patterns_overrides = fall_back_to_pt, allow_patterns_overrides
            self.is_unused_weight = is_unused_weight

    class Loader:
        def get_all_weights(self, config, model):
            yield ("original", 1)

    loader_class: Any = Loader
    loader_class.Source = Source
    default: Any = types.ModuleType("vllm.model_executor.model_loader.default_loader")
    default.DefaultModelLoader = Loader
    weights: Any = types.ModuleType("vllm.model_executor.model_loader.weight_utils")
    weights._natural_sort_key = str
    weights.should_skip_weight = lambda name, ids: False
    calls = []

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def keys(self):
            return ["ple", "normal"]

        def get_tensor(self, name):
            calls.append(name)
            return name

    weights.safe_open = lambda *args, **kwargs: Handle()
    package: Any = types.ModuleType("vllm.model_executor.model_loader")
    package.default_loader, package.weight_utils = default, weights
    modules = {
        "vllm.model_executor.model_loader": package,
        "vllm.model_executor.model_loader.default_loader": default,
        "vllm.model_executor.model_loader.weight_utils": weights,
    }
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("external_filter_test", PATCH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return Loader, calls


def make_loader(cls):
    obj = cls()
    obj.load_config = SimpleNamespace(
        load_format="auto", safetensors_load_strategy="lazy", model_loader_extra_config={}
    )
    obj._encoder_only_lm_prefixes = None
    obj.local_expert_ids = None
    obj.counter_before_loading_weights = 0
    obj._prepare_weights = lambda *args: ("folder", ["mixed.safetensors"], True, None)
    return obj


def test_mixed_file_skips_before_materialization():
    cls, calls = load_patch()
    model = SimpleNamespace(
        filter_unused_checkpoint_weights=True, is_unused_checkpoint_weight=lambda name: name == "ple"
    )
    assert list(make_loader(cls).get_all_weights(SimpleNamespace(model="x", revision=None), model)) == [
        ("normal", "normal")
    ]
    assert calls == ["normal"]


def test_ordinary_and_native_loading_use_original():
    for native in (False, True):
        cls, calls = load_patch(native)
        model = SimpleNamespace(filter_unused_checkpoint_weights=native)
        assert list(make_loader(cls).get_all_weights(None, model)) == [("original", 1)]
        assert not calls
        assert getattr(cls, "_ascend_external_weight_filter", False) is not native


def test_unsupported_strategy_rejected():
    cls, _ = load_patch()
    obj = make_loader(cls)
    obj.load_config.safetensors_load_strategy = "eager"
    model = SimpleNamespace(filter_unused_checkpoint_weights=True, is_unused_checkpoint_weight=lambda name: False)
    with pytest.raises(ValueError, match="lazy"):
        list(obj.get_all_weights(None, model))


def test_real_vllm_loader_mixed_safetensors(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    default = pytest.importorskip("vllm.model_executor.model_loader.default_loader")
    weights = pytest.importorskip("vllm.model_executor.model_loader.weight_utils")
    cls = default.DefaultModelLoader
    if "filter_unused_per_tensor" in cls.Source.__dataclass_fields__:
        pytest.skip("native filtering is exercised by upstream vLLM")
    original = cls.get_all_weights
    monkeypatch.setattr(cls, "get_all_weights", original)
    monkeypatch.setattr(cls, "_ascend_external_weight_filter", False, raising=False)
    spec = importlib.util.spec_from_file_location("real_external_filter_test", PATCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = str(tmp_path / "mixed.safetensors")
    safetensors.save_file({"ple": torch.ones(2), "normal": torch.zeros(2)}, path)
    obj = object.__new__(cls)
    obj.load_config = SimpleNamespace(
        load_format="auto", safetensors_load_strategy="lazy", model_loader_extra_config={}
    )
    obj._encoder_only_lm_prefixes = None
    obj.local_expert_ids = None
    obj.counter_before_loading_weights = 0
    obj._prepare_weights = lambda *args: (str(tmp_path), [path], True, None)
    real_open = weights.safe_open
    calls = []

    class TrackingOpen:
        def __init__(self, *args, **kwargs):
            self.handle = real_open(*args, **kwargs)

        def __enter__(self):
            self.inner = self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def keys(self):
            return self.inner.keys()

        def get_tensor(self, name):
            calls.append(name)
            return self.inner.get_tensor(name)

    monkeypatch.setattr(weights, "safe_open", TrackingOpen)
    model = SimpleNamespace(
        filter_unused_checkpoint_weights=True, is_unused_checkpoint_weight=lambda name: name == "ple"
    )
    result = list(obj.get_all_weights(SimpleNamespace(model="model", revision=None), model))
    assert [name for name, _ in result] == ["normal"]
    assert calls == ["normal"]
    assert torch.equal(result[0][1], torch.zeros(2))
