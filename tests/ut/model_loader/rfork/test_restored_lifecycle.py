# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import weakref
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from .rfork_test_support import _load_module, _stub


@pytest.fixture
def mtp_model(rfork_helpers, tensor_runtime, monkeypatch):
    class WeightsMapper:
        def __init__(self, *, orig_to_new_prefix):
            self.orig_to_new_prefix = orig_to_new_prefix

        def _map_name(self, name):
            for prefix, replacement in self.orig_to_new_prefix.items():
                if name.startswith(prefix):
                    return replacement + name[len(prefix) :] if replacement is not None else None
            return name

        def apply_list(self, names):
            return [mapped for name in names if (mapped := self._map_name(name)) is not None]

        def apply(self, weights):
            return ((mapped, tensor) for name, tensor in weights if (mapped := self._map_name(name)) is not None)

    class DeepSeekMTP(torch.nn.Module):
        def load_weights(self, weights):
            return {name for name, _ in weights}

    _stub(monkeypatch, "vllm.compilation.decorators", support_torch_compile=lambda cls: cls)
    _stub(monkeypatch, "vllm.config", VllmConfig=object)
    _stub(monkeypatch, "vllm.model_executor.models.deepseek_mtp", DeepSeekMTP=DeepSeekMTP)
    _stub(monkeypatch, "vllm.model_executor.models.deepseek_v2", GlmMoeDsaForCausalLM=torch.nn.Module)
    _stub(monkeypatch, "vllm.model_executor.models.utils", AutoWeightsLoader=object, WeightsMapper=WeightsMapper)
    _stub(monkeypatch, "vllm.sequence", IntermediateTensors=object)
    _stub(monkeypatch, "vllm_ascend.utils", is_rot_weight_used=lambda _config: False)
    module = _load_module(monkeypatch, "_rfork_test_mtp_model", "../../models/deepseek_mtp.py")
    monkeypatch.setattr(rfork_helpers.mtp_module, "AscendDeepSeekMTP", module.AscendDeepSeekMTP)
    model = module.AscendDeepSeekMTP.__new__(module.AscendDeepSeekMTP)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=3)
    model.quant_config = None
    model.model = torch.nn.Module()
    model.model.mtp_start_layer_idx = 3
    layer = torch.nn.Module()
    layer.shared_head = torch.nn.Module()
    layer.shared_head.head = torch.nn.Linear(2, 2, bias=False)
    model.model.layers = torch.nn.ModuleDict({"3": layer})
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_tensor_on_transfer_device", lambda _tensor: True)
    return SimpleNamespace(model=model, mapper=WeightsMapper)


@pytest.mark.parametrize(
    "head_name, has_own_head",
    [
        ("model.layers.3.shared_head.head.weight", True),
        ("lm_head.weight", False),
        ("layers.3.shared_head.head.weight", False),
        ("model.language_model.layers.3.shared_head.head.weight", False),
        (None, False),
    ],
)
@pytest.mark.parametrize("processed_layout", [False, True])
def test_mtp_checkpoint_preparation_preserves_registered_structure(
    rfork_helpers, tensor_runtime, mtp_model, monkeypatch, head_name, has_own_head, processed_layout
):
    model = mtp_model.model
    layout = tensor_runtime.tensor_layout
    head = model.model.layers["3"].shared_head.head
    original = head.weight.detach().clone()
    assert not hasattr(model, "lm_head")
    names = {head_name} if head_name is not None else set()
    unprepared_digest = layout.build_structural_digest(layout.collect_transferable_tensors(model, processed_layout))

    monkeypatch.setattr(rfork_helpers.loader, "_get_checkpoint_weight_names", lambda *args: names)
    rfork_helpers.loader._prepare_rfork_model_state(model, SimpleNamespace(), SimpleNamespace())
    before = layout.collect_transferable_tensors(model, processed_layout)
    digest = layout.build_structural_digest(before)
    loaded = model.load_weights((name, original) for name in names)
    after = layout.collect_transferable_tensors(model, processed_layout)

    assert loaded == names
    assert model.has_own_lm_head is has_own_head
    assert hasattr(model, "lm_head") is has_own_head
    if has_own_head:
        assert model.lm_head is head
    expected = "lm_head.weight" if has_own_head else "model.layers.3.shared_head.head.weight"
    assert [name for name, _ in before] == [expected]
    assert [name for name, _ in after] == [expected]
    assert (digest != unprepared_digest) is has_own_head
    assert layout.build_structural_digest(after) == digest
    torch.testing.assert_close(head.weight, original)


def test_mtp_preparation_applies_same_checkpoint_mappers_as_loading(rfork_helpers, mtp_model, monkeypatch):
    model = mtp_model.model
    mapper = mtp_model.mapper(
        orig_to_new_prefix={"checkpoint_head.": "model.layers.3.shared_head.head.", "checkpoint_rot.": "rot."}
    )
    model.quant_config = SimpleNamespace(get_cache_scale_mapper=lambda: mapper)
    raw_names = {"checkpoint_head.weight", "checkpoint_rot.weight"}

    monkeypatch.setattr(rfork_helpers.loader, "_get_checkpoint_weight_names", lambda *args: raw_names)
    rfork_helpers.loader._prepare_rfork_model_state(model, SimpleNamespace(), SimpleNamespace())
    assert model.has_own_lm_head is True
    assert model.lm_head is model.model.layers["3"].shared_head.head
    loaded = model.load_weights((name, torch.ones(2)) for name in raw_names)

    assert loaded == {"model.layers.3.shared_head.head.weight", "model.layers.3.rot.weight"}
    assert model.has_own_lm_head is True
    model.load_weights([])
    assert model.has_own_lm_head is False


@pytest.mark.parametrize(
    ("model_sleep", "vllm_sleep", "weight_transfer", "expected"),
    [
        (True, False, False, "sleep mode"),
        (False, True, False, "sleep mode"),
        (False, False, True, "online weight transfer (weight_transfer_config)"),
        (False, False, False, None),
        (True, False, True, "sleep mode"),
    ],
)
def test_mutable_weights_bypass_reason_reports_sleep_and_weight_transfer(
    rfork_helpers, model_sleep, vllm_sleep, weight_transfer, expected
):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=vllm_sleep),
        weight_transfer_config=object() if weight_transfer else None,
    )
    model_config = SimpleNamespace(enable_sleep_mode=model_sleep)

    assert rfork_helpers.safety.mutable_weights_bypass_reason(vllm_config, model_config) == expected


def _add_dynamo_hook(hooks, owner):
    handle = RemovableHandle(hooks)
    hooks[handle.id] = owner.bytecode_hook
    owner._bytecode_hook_handle = handle
    return handle.id


def test_fallback_reset_restores_ascend_globals_and_only_removes_new_dynamo_hooks(rfork_helpers):
    class HookOwner(Module):
        def bytecode_hook(self, *args, **kwargs):
            pass

    adaptor = rfork_helpers.adaptor_module.VllmEplbAdaptor
    baseline_layer = object()
    baseline_registry = [baseline_layer]
    adaptor._registered_moe_layers = baseline_registry
    rfork_helpers.routed_experts_module.AscendRoutedExperts.moe_counter = 17

    baseline_caches = (object(), object(), object())
    (
        rfork_helpers.rotary_module._cos_sin_cache,
        rfork_helpers.rotary_module._cos_cache,
        rfork_helpers.rotary_module._sin_cache,
    ) = baseline_caches

    baseline_owner = HookOwner()
    baseline_hook_id = _add_dynamo_hook(rfork_helpers.dynamo_hooks, baseline_owner)
    vllm_config = SimpleNamespace(compilation_config=SimpleNamespace())
    snapshot = rfork_helpers.loader._snapshot_process_global_model_state(vllm_config)

    baseline_registry.append(object())
    adaptor._registered_moe_layers = [object()]
    rfork_helpers.routed_experts_module.AscendRoutedExperts.moe_counter = 999
    dirty_caches = (object(), object(), object())
    (
        rfork_helpers.rotary_module._cos_sin_cache,
        rfork_helpers.rotary_module._cos_cache,
        rfork_helpers.rotary_module._sin_cache,
    ) = dirty_caches

    discarded_owner = HookOwner()
    discarded_hook_id = _add_dynamo_hook(rfork_helpers.dynamo_hooks, discarded_owner)
    assert discarded_hook_id not in snapshot.dynamo_bytecode_hook_ids

    rfork_helpers.loader._reset_process_global_model_state(vllm_config, snapshot=snapshot)

    assert adaptor._registered_moe_layers is baseline_registry
    assert adaptor._registered_moe_layers == [baseline_layer]
    assert rfork_helpers.routed_experts_module.AscendRoutedExperts.moe_counter == 17
    assert (
        rfork_helpers.rotary_module._cos_sin_cache,
        rfork_helpers.rotary_module._cos_cache,
        rfork_helpers.rotary_module._sin_cache,
    ) == baseline_caches
    assert baseline_hook_id in rfork_helpers.dynamo_hooks
    assert rfork_helpers.dynamo_hooks[baseline_hook_id].__self__ is baseline_owner
    assert discarded_hook_id not in rfork_helpers.dynamo_hooks


@pytest.fixture
def checkpoint_loader(monkeypatch, rfork_helpers):
    @dataclass
    class CheckpointSource:
        model_or_path: str
        revision: str | None
        subfolder: str | None = None
        prefix: str = ""
        fall_back_to_pt: bool = True
        allow_patterns_overrides: list[str] | None = None

    calls = []
    selections: dict[str, tuple[list[str], bool]] = {}
    configs = []

    class DefaultModelLoader:
        Source = CheckpointSource

        def __init__(self, load_config):
            configs.append(load_config)

        def _prepare_weights(self, *args):
            calls.append(args)
            return "unused", *selections[args[0]]

    _stub(monkeypatch, "vllm.model_executor.model_loader.default_loader", DefaultModelLoader=DefaultModelLoader)
    return SimpleNamespace(Source=CheckpointSource, calls=calls, selections=selections, configs=configs)


def test_checkpoint_names_follow_selected_sources_and_prefixes(rfork_helpers, checkpoint_loader, tmp_path):
    primary = tmp_path / "selected.safetensors"
    secondary = tmp_path / "secondary.safetensors"
    save_file({"model.layers.3.shared_head.head.weight": torch.ones(2)}, primary)
    save_file({"head.weight": torch.zeros(2)}, secondary)
    save_file({"unselected.weight": torch.ones(2)}, tmp_path / "stale.safetensors")
    checkpoint_loader.selections.update(primary=([str(primary)], True), secondary=([str(secondary)], True))
    model = Module()
    model.fall_back_to_pt_during_load = False
    model.allow_patterns_overrides = ["selected*.safetensors"]
    model.secondary_weights = [
        checkpoint_loader.Source("secondary", "secondary-revision", "subfolder", "aux.", False, ["secondary*"])
    ]
    load_config = SimpleNamespace(load_format="rfork", model_loader_extra_config={"rfork_scheduler_url": "planner"})

    names = rfork_helpers.loader._get_checkpoint_weight_names(
        model, SimpleNamespace(model="primary", revision="primary-revision"), load_config
    )

    assert names == {"model.layers.3.shared_head.head.weight", "aux.head.weight"}
    assert checkpoint_loader.calls == [
        ("primary", None, "primary-revision", False, ["selected*.safetensors"]),
        ("secondary", "subfolder", "secondary-revision", False, ["secondary*"]),
    ]
    assert checkpoint_loader.configs[0].load_format == "auto"
    assert checkpoint_loader.configs[0].model_loader_extra_config == {}
    assert load_config.load_format == "rfork"
    assert load_config.model_loader_extra_config == {"rfork_scheduler_url": "planner"}


@pytest.mark.parametrize("metadata", ["pt", "corrupt", "empty", "missing"])
def test_checkpoint_names_reject_unavailable_metadata(rfork_helpers, checkpoint_loader, tmp_path, metadata):
    path = tmp_path / "model.safetensors"
    if metadata == "corrupt":
        path.write_bytes(b"invalid safetensors")
    elif metadata == "empty":
        save_file({}, path)
    checkpoint_loader.selections["model"] = ([str(path)], metadata != "pt")
    config = SimpleNamespace(load_format="rfork", model_loader_extra_config={})

    with pytest.raises((OSError, ValueError)):
        rfork_helpers.loader._get_checkpoint_weight_names(
            Module(), SimpleNamespace(model="model", revision=None), config
        )


@pytest.mark.parametrize("processed_layout", [False, True])
@pytest.mark.parametrize(
    "outcome", ["transfer", "shared", "local", "transfer_error", "metadata_error", "state_error", "other_model"]
)
def test_model_state_preparation_precedes_registration_and_guards_fallback(
    rfork_helpers, monkeypatch, processed_layout, outcome
):
    module = rfork_helpers.loader
    # Pytest's log capture would retain fallback exceptions and their model-owning tracebacks.
    monkeypatch.setattr(module.logger, "propagate", False)
    events = []
    names = {"model.layers.3.shared_head.head.weight"}
    fallback_model = Module()
    model_ref: Callable[[], Module | None] = lambda: None

    class Model(Module):
        quant_config = None

        def _maybe_set_own_lm_head(self, weight_names):
            assert weight_names == names
            events.append("prepare")
            if outcome == "state_error":
                raise ValueError("head state unavailable")
            self.has_own_lm_head = True

    class OtherModel(Module):
        def _maybe_set_own_lm_head(self, weight_names):
            raise AssertionError("RFork must not prepare unrelated model types")

    monkeypatch.setattr(rfork_helpers.mtp_module, "AscendDeepSeekMTP", Model)

    def initialize(**kwargs):
        nonlocal model_ref
        model = OtherModel() if outcome == "other_model" else Model()
        model_ref = weakref.ref(model)
        events.append("initialize")
        return model

    def default_load(*args):
        # Fallback must release the discarded model before allocating its replacement.
        assert model_ref() is None
        return record("default", fallback_model)

    def metadata(*args):
        assert outcome != "other_model"
        events.append("metadata")
        if outcome == "metadata_error":
            raise ValueError("checkpoint metadata unavailable")
        return names

    def record(event, result=None):
        events.append(event)
        if event in ("shared", "process", "register") and outcome != "other_model":
            live_model = model_ref()
            assert isinstance(live_model, Model)
            assert live_model.has_own_lm_head is True
        return result

    session = SimpleNamespace(
        can_reuse_shared_weights=lambda *args: record("shared", outcome == "shared"),
        register_destination=lambda *args: record("register", True),
        acquire_seed=lambda: record("acquire", outcome != "local"),
        transfer_from_seed=lambda *args: record("transfer", outcome != "transfer_error"),
        log_transferred_model_layout=lambda *args: record("digest", "prepared-digest"),
        prepare_for_fallback=lambda: record("cleanup", SimpleNamespace(can_schedule_seed=True)),
    )
    loader = module.RForkModelLoader.__new__(module.RForkModelLoader)
    loader.load_config = SimpleNamespace(device=None)
    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda *args: session)
    monkeypatch.setattr(loader, "_get_target_registered_blocks", lambda *args: [(1, 2)])
    monkeypatch.setattr(loader, "_requires_processed_layout_transfer", lambda *args: processed_layout)
    monkeypatch.setattr(module, "_get_checkpoint_weight_names", metadata)
    monkeypatch.setattr(module, "initialize_model", initialize)
    monkeypatch.setattr(module, "process_weights_after_loading", lambda *args: record("process"))
    monkeypatch.setattr(module, "_rfork_skip_unquantized_moe_post_load_processing", lambda *args: nullcontext())
    monkeypatch.setattr(module, "_start_rfork_seed_service", lambda *args, **kwargs: record("seed", True))
    monkeypatch.setattr(module, "_load_with_default_loader", default_load)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None), raising=False
    )
    monkeypatch.setattr(
        module,
        "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    model_config = SimpleNamespace(dtype=torch.float32)
    vllm_config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"), model_config=model_config)

    result = loader.load_model(vllm_config, model_config)

    prefix = ["initialize"] + ([] if outcome == "other_model" else ["metadata"])
    if outcome not in ("other_model", "metadata_error"):
        prefix.append("prepare")
    if outcome in ("metadata_error", "state_error"):
        assert result is fallback_model
        assert events == prefix + ["cleanup", "default"]
    elif outcome == "shared":
        assert result is model_ref()
        assert events == prefix + ["shared"]
    else:
        assert result is (fallback_model if outcome in ("local", "transfer_error") else model_ref())
        expected = prefix + ["shared"] + (["process"] if processed_layout else []) + ["register", "acquire"]
        if outcome in ("local", "transfer_error"):
            expected += ([] if outcome == "local" else ["transfer"]) + ["cleanup", "default", "seed"]
        else:
            expected += ["transfer"] + ([] if processed_layout else ["process"]) + ["digest", "seed"]
        assert events == expected
