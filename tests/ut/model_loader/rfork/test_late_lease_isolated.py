# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.model_loader.rfork.test_lease_release import runtime as runtime


@pytest.fixture
def loader_runtime(request, monkeypatch):
    """Execute real load_model/rollback logic on CPU with external model construction injected."""
    lease_runtime = request.getfixturevalue("runtime")

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, attr = name.rpartition(".")
        if parent in sys.modules:
            monkeypatch.setattr(sys.modules[parent], attr, module, raising=False)
        return module

    class BaseLoader:
        def __init__(self, load_config):
            self.load_config = load_config

    stub("vllm", __path__=[])
    stub("vllm.config", __path__=[], ModelConfig=SimpleNamespace, VllmConfig=SimpleNamespace)
    stub("vllm.config.load", LoadConfig=SimpleNamespace)
    stub("vllm.distributed", __path__=[], get_tensor_model_parallel_rank=lambda: 0)
    stub("vllm.distributed.parallel_state", get_ep_group=Mock(), get_pp_group=Mock())
    stub("vllm.model_executor", __path__=[])
    model_loader = stub(
        "vllm.model_executor.model_loader", __path__=[], register_model_loader=lambda name: lambda cls: cls
    )
    stub("vllm.model_executor.model_loader.base_loader", BaseModelLoader=BaseLoader)
    stub("vllm.model_executor.model_loader.utils", initialize_model=Mock(), process_weights_after_loading=Mock())
    stub("vllm.utils.torch_utils", set_default_torch_dtype=lambda dtype: nullcontext())
    stub("vllm.model_executor.layers", __path__=[])
    rope = stub("vllm.model_executor.layers.rotary_embedding", _ROPE_DICT={})
    stub(
        "vllm_ascend.ascend_config",
        get_ascend_config=lambda: SimpleNamespace(
            eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)
        ),
    )
    stub("vllm_ascend.device.hardware_profile", get_current_hardware_profile=Mock())
    monkeypatch.setattr(
        sys.modules["vllm_ascend.model_loader.rfork.identity"], "build_compatibility_fingerprint", Mock(), raising=False
    )
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/model_loader/rfork/rfork_loader.py"
    name = "vllm_ascend.model_loader.rfork.rfork_loader"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    events = []
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda: events.append("sync"), empty_cache=Mock()), raising=False
    )
    monkeypatch.setattr(module, "_rfork_pre_transfer_weight_processing", lambda model: nullcontext())
    monkeypatch.setattr(module, "_rfork_skip_unquantized_moe_post_load_processing", lambda model: nullcontext())
    load_config = SimpleNamespace(device=None, model_loader_extra_config={}, load_format="rfork")
    loader = module.RForkModelLoader(load_config)
    config = SimpleNamespace(dtype=torch.float32, quantization=None, model="model")
    existing = torch.nn.Module()
    vc = SimpleNamespace(
        device_config=SimpleNamespace(device="cpu"),
        parallel_config=SimpleNamespace(enable_eplb=False),
        compilation_config=SimpleNamespace(
            static_forward_context={"existing": existing}, static_all_moe_layers=[existing]
        ),
    )
    rope._ROPE_DICT["existing"] = existing
    session = Mock(identity=lease_runtime.identity)
    session.acquire_seed.side_effect = lambda: events.append("acquire") or True
    session.transfer_from_seed.side_effect = lambda *args: events.append("transfer") or True
    session.prepare_for_fallback.side_effect = lambda: events.append("cleanup") or True
    session.start_seed_service.side_effect = lambda *args: events.append("publish") or True
    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda *args: session)
    monkeypatch.setattr(loader, "_get_target_registered_blocks", lambda *args: [])
    model = torch.nn.Module()
    fallback = torch.nn.Module()

    def initialize(**kwargs):
        events.append("initialize")
        vc.compilation_config.static_forward_context["new"] = model
        vc.compilation_config.static_all_moe_layers.append(model)
        rope._ROPE_DICT["new"] = model
        return model

    monkeypatch.setattr(module, "initialize_model", initialize)
    monkeypatch.setattr(module, "process_weights_after_loading", lambda *args: events.append("layout"))
    monkeypatch.setattr(
        model_loader, "get_model", lambda **kwargs: events.append("fallback") or fallback, raising=False
    )
    return SimpleNamespace(
        module=module,
        loader=loader,
        config=config,
        vc=vc,
        session=session,
        events=events,
        rope=rope._ROPE_DICT,
        existing=existing,
        model=model,
        fallback=fallback,
    )


@pytest.mark.parametrize("processed", [False, True])
def test_real_loader_acquires_only_after_preparation(loader_runtime, monkeypatch, processed):
    r = loader_runtime
    monkeypatch.setattr(r.loader, "_requires_processed_layout_transfer", lambda config: processed)
    assert r.loader.load_model(r.vc, r.config) is r.model
    expected = (
        ["initialize", "layout", "sync", "acquire", "transfer", "publish"]
        if processed
        else ["initialize", "acquire", "transfer", "layout", "publish"]
    )
    assert r.events == expected


@pytest.mark.parametrize("failure", ["seed_miss", "initialize", "layout", "transfer"])
def test_real_loader_failure_rolls_back_only_attempt_state(loader_runtime, monkeypatch, failure):
    r = loader_runtime
    monkeypatch.setattr(r.loader, "_requires_processed_layout_transfer", lambda config: True)
    if failure == "seed_miss":
        r.session.acquire_seed.side_effect = lambda: r.events.append("acquire") or False
    elif failure == "initialize":
        original = r.module.initialize_model

        def fail_initialize(**kwargs):
            original(**kwargs)
            raise RuntimeError("partial construction failed")

        monkeypatch.setattr(r.module, "initialize_model", fail_initialize)
    elif failure == "layout":
        monkeypatch.setattr(r.module, "process_weights_after_loading", Mock(side_effect=RuntimeError("layout failed")))
    else:
        r.session.transfer_from_seed.side_effect = lambda *args: r.events.append("transfer") or False
    assert r.loader.load_model(r.vc, r.config) is r.fallback
    assert r.vc.compilation_config.static_forward_context == {"existing": r.existing}
    assert r.vc.compilation_config.static_all_moe_layers == [r.existing]
    assert r.rope == {"existing": r.existing}
    assert r.events.index("cleanup") < r.events.index("fallback")
    if failure in ("initialize", "layout"):
        r.session.acquire_seed.assert_not_called()
    if failure != "transfer":
        r.session.transfer_from_seed.assert_not_called()


def test_snapshot_restores_original_registry_references_after_replacement(loader_runtime):
    r = loader_runtime
    compilation = r.vc.compilation_config
    original_context = compilation.static_forward_context
    original_layers = compilation.static_all_moe_layers
    snapshot = r.module._snapshot_process_global_model_state(r.vc)
    original_context["partial"] = object()
    original_layers.append(object())
    compilation.static_forward_context = {"replacement": object()}
    compilation.static_all_moe_layers = [object()]
    r.module._reset_process_global_model_state(r.vc, snapshot=snapshot)
    assert compilation.static_forward_context is original_context
    assert compilation.static_all_moe_layers is original_layers
    assert original_context == {"existing": r.existing}
    assert original_layers == [r.existing]


def test_snapshot_restores_deleted_replaced_and_shared_baseline_objects(loader_runtime):
    r = loader_runtime
    snapshot = r.module._snapshot_process_global_model_state(r.vc)
    r.vc.compilation_config.static_forward_context.clear()
    r.vc.compilation_config.static_forward_context["partial"] = object()
    r.vc.compilation_config.static_all_moe_layers[:] = [SimpleNamespace()]
    r.rope["existing"] = object()
    # The discarded draft references an existing target module; rollback must preserve it.
    discarded = torch.nn.Module()
    discarded.add_module("shared", r.existing)
    r.module._reset_process_global_model_state(r.vc, discarded, snapshot)
    assert r.vc.compilation_config.static_forward_context == {"existing": r.existing}
    assert r.vc.compilation_config.static_all_moe_layers == [r.existing]
    assert r.rope == {"existing": r.existing}
