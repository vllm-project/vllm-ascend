# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import gc
import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch.nn import Module


@pytest.fixture
def device_layout(tensor_runtime, monkeypatch):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda tensor: True)
    return layout


def _model() -> Module:
    model = Module()
    model.layer = Module()
    model.layer.weight = torch.nn.Parameter(torch.ones(4, 4))
    model.layer.impl = SimpleNamespace(weight=model.layer.weight)
    return model


def _untransferred(layout, model, baseline) -> list[str]:
    return layout.find_untransferred_post_load_tensors(model, baseline, layout.collect_transferable_tensors(model))


def test_post_load_guard_reports_private_derived_tensors(device_layout):
    model = _model()
    baseline = device_layout.snapshot_model_state_tensors(model)

    model.layer.impl._cache = {"weight": model.layer.weight.detach().t().contiguous()}

    assert _untransferred(device_layout, model, baseline) == ["layer.impl._cache['weight']"]


def test_post_load_guard_accepts_transferred_aliases_existing_state_and_workspaces(device_layout):
    model = _model()
    model.layer._init_workspace = torch.zeros(2)
    baseline = device_layout.snapshot_model_state_tensors(model)

    weight = model.layer.weight.detach()
    model.layer._row_view = weight[1:]
    model.layer.impl._transposed_view = weight.t()
    model.layer.impl._flashmla_writer_buffers = (torch.empty(3), torch.arange(3))
    model.layer.impl._released = torch.empty(0)

    assert _untransferred(device_layout, model, baseline) == []


def test_post_load_guard_reports_host_state_and_nested_helpers(tensor_runtime):
    layout = tensor_runtime.tensor_layout

    @dataclass
    class Part:
        full_tensor: torch.Tensor

    model = Module()
    baseline = layout.snapshot_model_state_tensors(model)

    model.host_scale = torch.ones(1)
    model.switch = SimpleNamespace(_states=((model, Part(torch.ones(2))),))

    assert _untransferred(layout, model, baseline) == ["host_scale", "switch._states[0][1].full_tensor"]


def test_post_load_snapshot_does_not_retain_replaced_tensors(device_layout):
    model = Module()
    model.cache = torch.ones(2)
    replaced = weakref.ref(model.cache)
    baseline = device_layout.snapshot_model_state_tensors(model)

    model.cache = torch.ones(2)
    gc.collect()

    assert replaced() is None
    assert _untransferred(device_layout, model, baseline) == []


class _PrologImpl:
    def __init__(self, events: list[str]) -> None:
        self._prolog_v3_weights = None
        self.events = events

    def _refresh_prolog_v3_weights(self) -> None:
        self.events.append("refresh_prolog")


def _load(rfork_helpers, monkeypatch, model: Module, process, events: list[str]):
    module = rfork_helpers.loader
    # Pytest's log capture would retain fallback exceptions and their model-owning tracebacks.
    monkeypatch.setattr(module.logger, "propagate", False)
    fallback_model = Module()

    def record(event, result=None):
        events.append(event)
        return result

    session = SimpleNamespace(
        loading=nullcontext,
        register_receiver_weights=lambda *args: record("register", True),
        acquire_seed_lease=lambda: record("acquire", True),
        read_weights_from_seed=lambda *args: record("transfer", True),
        validate_transferred_model_layout=lambda *args: record("validate"),
        prepare_for_fallback=lambda: record("cleanup", True),
    )
    loader = module.RForkModelLoader.__new__(module.RForkModelLoader)
    loader.load_config = SimpleNamespace(device=None)
    monkeypatch.setattr(loader, "_get_session", lambda *args: session)
    monkeypatch.setattr(module, "initialize_model", lambda **kwargs: model)
    monkeypatch.setattr(module, "process_weights_after_loading", lambda *args: process())
    monkeypatch.setattr(module, "_schedule_seed_publication", lambda *args, **kwargs: record("seed", True))
    monkeypatch.setattr(module, "_load_with_default_loader", lambda *args: record("default", fallback_model))
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None), raising=False
    )
    monkeypatch.setattr(
        module,
        "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    model_config = SimpleNamespace(dtype=torch.float32, quantization=None)
    vllm_config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"), model_config=model_config)
    return loader.load_model(vllm_config, model_config), fallback_model


def test_receiver_falls_back_before_registration_when_post_load_state_is_untransferred(rfork_helpers, monkeypatch):
    events: list[str] = []
    model = Module()
    model.weight = torch.nn.Parameter(torch.ones(2))

    def process():
        model._derived = model.weight.detach() * 2

    result, fallback_model = _load(rfork_helpers, monkeypatch, model, process, events)

    assert result is fallback_model
    assert events == ["cleanup", "default", "seed"]


def test_receiver_refreshes_prolog_v3_cache_after_transfer(rfork_helpers, monkeypatch):
    events: list[str] = []
    model = Module()
    model.attn = Module()
    model.attn.impl = _PrologImpl(events)
    model.idle = Module()
    model.idle.impl = _PrologImpl(events)

    def process():
        # Built from uninitialized weights; exempt from the guard because it is refreshed after the transfer.
        model.attn.impl._prolog_v3_weights = {"weight_dq": torch.ones(2)}

    result, _ = _load(rfork_helpers, monkeypatch, model, process, events)

    assert result is model
    assert events == ["register", "acquire", "transfer", "refresh_prolog", "validate", "seed"]
