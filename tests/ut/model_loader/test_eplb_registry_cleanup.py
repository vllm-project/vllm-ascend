import gc
from types import SimpleNamespace
from unittest.mock import patch
from weakref import ref

import pytest
from torch import nn

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.model_loader.netloader.netloader import ModelNetLoaderElastic
from vllm_ascend.model_loader.rfork.rfork_loader import (
    _reset_process_global_model_state,
    _snapshot_process_global_model_state,
)


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(VllmEplbAdaptor, "_registered_moe_layers", [])


def _config():
    return SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={}, static_all_moe_layers=[]))


def test_netloader_rollback_restores_snapshot_after_dead_ref_pruning():
    config = _config()
    target = nn.Module()
    old = nn.Module()
    VllmEplbAdaptor.register_layer(target)
    VllmEplbAdaptor.register_layer(old)
    with patch.object(ModelNetLoaderElastic, "_get_npu_memory_usage", return_value=None):
        context = ModelNetLoaderElastic._create_fallback_cleanup_context(config, "npu")
    old_ref = ref(old)
    del old
    gc.collect()
    failed_model = nn.Module()
    draft = nn.Module()
    failed_model.add_module("draft", draft)
    VllmEplbAdaptor.register_layer(draft)
    assert old_ref() is None
    assert len(VllmEplbAdaptor._registered_moe_layers) == 2
    with (
        patch.object(ModelNetLoaderElastic, "_cleanup_compilation_hooks"),
        patch.object(ModelNetLoaderElastic, "_remove_new_static_forward_context_keys"),
    ):
        failed_model_ref = ModelNetLoaderElastic._release_failed_model_references(failed_model, config, context)
    del failed_model, draft
    gc.collect()
    assert failed_model_ref() is None
    assert VllmEplbAdaptor.get_registered_layers() == [target]


def test_rfork_snapshot_does_not_own_model_layers():
    layer = nn.Module()
    VllmEplbAdaptor.register_layer(layer)
    layer_ref = ref(layer)
    snapshot = _snapshot_process_global_model_state(_config())
    del layer
    gc.collect()
    assert layer_ref() is None
    assert snapshot.ascend_moe_layers is not None


def test_rfork_reset_preserves_live_layers_outside_failed_model():
    target = nn.Module()
    failed_model = nn.Module()
    stale = nn.Module()
    failed_model.add_module("stale", stale)
    VllmEplbAdaptor.register_layer(target)
    VllmEplbAdaptor.register_layer(stale)
    _reset_process_global_model_state(_config(), model=failed_model)
    assert VllmEplbAdaptor.get_registered_layers() == [target]


def test_rfork_snapshot_restore_maintains_registry_identity_and_order():
    first, second, draft = nn.Module(), nn.Module(), nn.Module()
    VllmEplbAdaptor.register_layer(first)
    VllmEplbAdaptor.register_layer(second)
    registry = VllmEplbAdaptor._registered_moe_layers
    config = _config()
    snapshot = _snapshot_process_global_model_state(config)
    VllmEplbAdaptor.register_layer(draft)
    _reset_process_global_model_state(config, snapshot=snapshot)
    assert VllmEplbAdaptor._registered_moe_layers is registry
    assert VllmEplbAdaptor.get_registered_layers() == [first, second]
