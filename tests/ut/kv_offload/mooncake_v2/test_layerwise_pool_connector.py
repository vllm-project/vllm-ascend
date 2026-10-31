# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import signal
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata, KVConnectorRole

from vllm_ascend.core.kv_cache_placement import build_layer_cache_bundles
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import layerwise_pool_connector as facade
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.layerwise_pool_connector import LayerwisePoolConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import (
    LayerwisePoolConnectorMetadata,
    PoolSaveRequest,
)


@pytest.mark.parametrize("fails", [False, True])
def test_backend_init_restores_native_termination_handlers_even_when_python_table_is_unchanged(fails):
    libc = ctypes.CDLL(None)
    libc.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]
    libc.signal.restype = ctypes.c_void_p
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    received = []

    def framework_handler(signum, frame):
        received.append(signum)

    def backend(*args, **kwargs):
        for signum in previous:
            # Mirror the SDK's C-level replacement without changing the Python
            # table. Ignoring the signal keeps the regression safe when broken.
            libc.signal(signum, ctypes.c_void_p(1))
            assert signal.getsignal(signum) is framework_handler
        if fails:
            raise ValueError("SDK initialization failed")
        return "backend"

    try:
        for signum in previous:
            signal.signal(signum, framework_handler)
        if fails:
            with pytest.raises(ValueError, match="SDK initialization failed"):
                facade._create_backend(backend, None, {})
        else:
            assert facade._create_backend(backend, None, {}) == "backend"
        for signum in previous:
            signal.raise_signal(signum)
        assert received == list(previous)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def test_backend_init_rejects_background_start_before_native_resources_are_created():
    backend = MagicMock()
    errors = []

    def initialize():
        try:
            facade._create_backend(backend, None, {})
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=initialize)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(errors) == 1 and "main thread" in errors[0]
    backend.assert_not_called()


def test_v2_pre_forward_binding_prepares_writes_before_deferred_start_load(monkeypatch):
    worker = MagicMock()
    profile = object()

    def init_worker(connector):
        connector.connector_worker = worker
        connector._handshake = SimpleNamespace(profile=profile)

    monkeypatch.setattr(LayerwisePoolConnector, "_init_worker", init_worker)
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=True, is_kv_consumer=False))
    connector = LayerwisePoolConnector(config, KVConnectorRole.WORKER, MagicMock())
    metadata = LayerwisePoolConnectorMetadata()
    metadata.released_sources = {"cancelled-source"}
    request = PoolSaveRequest("transfer", "attempt", "digest", 0, 4, ((1,),), (("hash",),))
    metadata.saves["request"] = request
    connector.bind_connector_metadata(metadata)
    worker.prepare_save_request.assert_called_once_with("request", request, profile)
    worker.release_source_requests.assert_called_once_with({"cancelled-source"})
    worker.start_load_kv.assert_not_called()
    connector.start_load_kv(None)
    connector.wait_for_save()
    assert [call[0] for call in worker.mock_calls] == [
        "release_source_requests",
        "prepare_save_request",
        "start_load_kv",
        "wait_for_save",
    ]
    connector.clear_connector_metadata()
    connector.on_kv_cache_written("model.layers.0.attn")
    worker.save_kv_layer.assert_not_called()


def test_layer_save_reuses_attention_bundle_including_auxiliary_cache(monkeypatch):
    worker = MagicMock()

    def init_worker(connector):
        connector.connector_worker = worker

    monkeypatch.setattr(LayerwisePoolConnector, "_init_worker", init_worker)
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=True, is_kv_consumer=False))
    connector = LayerwisePoolConnector(config, KVConnectorRole.WORKER, MagicMock())
    connector.bind_connector_metadata(LayerwisePoolConnectorMetadata())
    connector._bundles = {"model.layers.0.attn": ("model.layers.0.attn", "model.layers.0.indexer")}
    connector.save_kv_layer("model.layers.0.attn", None, None)
    assert [call.args[0] for call in worker.save_kv_layer.call_args_list] == [
        "model.layers.0.attn",
        "model.layers.0.indexer",
    ]
    assert LayerwisePoolConnector.requires_piecewise_for_cudagraph({})
    with pytest.raises(TypeError, match="another connector"):
        connector.bind_connector_metadata(KVConnectorMetadata())


def test_target_scatter_never_submits_same_index_draft_cache(monkeypatch):
    worker = MagicMock()
    monkeypatch.setattr(LayerwisePoolConnector, "_init_worker", lambda self: setattr(self, "connector_worker", worker))
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=True, is_kv_consumer=False))
    connector = LayerwisePoolConnector(config, KVConnectorRole.WORKER, MagicMock())
    connector.bind_connector_metadata(LayerwisePoolConnectorMetadata())
    target = "model.layers.0.attn"
    auxiliary = "model.layers.0.indexer.k_cache"
    draft = "model.mtp.layers.0.attn"
    draft_auxiliary = "model.mtp.layers.0.indexer.k_cache"
    connector._bundles = build_layer_cache_bundles(dict.fromkeys((target, auxiliary, draft, draft_auxiliary), object()))
    connector.on_kv_cache_written(target)
    assert [call.args[0] for call in worker.save_kv_layer.call_args_list] == [target, auxiliary]
    worker.save_kv_layer.reset_mock()
    connector.save_kv_layer(draft, None, None)
    assert [call.args[0] for call in worker.save_kv_layer.call_args_list] == [draft, draft_auxiliary]


def test_cache_bundles_keep_explicit_non_numeric_draft_layers_independent():
    target, draft = "model.layers.0.attn", "draft.attn"
    assert build_layer_cache_bundles(dict.fromkeys((target, draft), object()), {draft}) == {
        target: (target,),
        draft: (draft,),
    }


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_graph_requirement_uses_platform_derived_role(monkeypatch, role):
    from vllm_ascend.platform import NPUPlatform

    # A stale or manually supplied marker must not override the actual role.
    extra = {"_pool_pd_role": "kv_consumer" if role == "kv_producer" else "kv_producer"}
    transfer = SimpleNamespace(
        kv_connector="LayerwisePoolConnector",
        kv_role=role,
        kv_connector_module_path=None,
        kv_connector_extra_config=extra,
    )
    monkeypatch.setattr("vllm_ascend.platform._get_reduced_cg_cap", lambda _: None)
    NPUPlatform.apply_config_platform_defaults(SimpleNamespace(kv_transfer_config=transfer))
    assert extra["_pool_pd_role"] == role
    assert LayerwisePoolConnector.requires_piecewise_for_cudagraph(extra) == (role == "kv_producer")


def test_producer_rejects_decode_graph_exemption_when_platform_hook_was_bypassed(monkeypatch):
    init_worker = MagicMock()
    monkeypatch.setattr(LayerwisePoolConnector, "_init_worker", init_worker)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=True, is_kv_consumer=False, kv_connector_extra_config={"_pool_pd_role": "kv_consumer"}
        )
    )
    with pytest.raises(ValueError, match="graph exemption"):
        LayerwisePoolConnector(config, KVConnectorRole.WORKER, MagicMock())
    init_worker.assert_not_called()


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_vllm_config_preserves_full_decode_graph_only_for_bulk_reader(monkeypatch, role):
    from vllm.config import CompilationConfig, DeviceConfig, KVTransferConfig, VllmConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    from vllm_ascend.distributed.kv_transfer import register_connector
    from vllm_ascend.platform import NPUPlatform

    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    register_connector()
    monkeypatch.setattr("vllm.platforms.current_platform", NPUPlatform())
    monkeypatch.setattr("vllm_ascend.platform._get_reduced_cg_cap", lambda _: None)
    # No model/device execution: run the real configuration pipeline and its
    # connector graph gate, with explicit CPU device to skip NPU setup.
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        kv_transfer_config=KVTransferConfig(kv_connector="LayerwisePoolConnector", kv_role=role),
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
    )
    expected = CUDAGraphMode.PIECEWISE if role == "kv_producer" else CUDAGraphMode.FULL_DECODE_ONLY
    assert config.compilation_config.cudagraph_mode == expected


def test_kvpp_registration_preserves_loaded_proposer_draft_identity(monkeypatch):
    target, draft = "model.layers.0.attn", "draft.attn"
    worker = MagicMock(tp_size=1, tp_rank=0, specs=dict.fromkeys((target, draft), object()))
    monkeypatch.setattr(LayerwisePoolConnector, "_init_worker", lambda self: setattr(self, "connector_worker", worker))
    monkeypatch.setattr(facade, "KVPPConfig", SimpleNamespace(from_vllm_config=lambda _: SimpleNamespace(size=2)))
    monkeypatch.setattr(facade, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(facade, "map_kvpp_layers_to_owners", lambda *_: {target: 1})
    monkeypatch.setattr(facade, "find_draft_layers", lambda *_: {draft})
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=True, is_kv_consumer=False))
    connector = LayerwisePoolConnector(config, KVConnectorRole.WORKER, MagicMock())
    caches = dict.fromkeys((target, draft), object())
    connector.register_kv_caches(caches)
    worker.register_kv_caches.assert_called_once_with(caches, owned_layers={draft})
    assert connector._bundles[draft] == (draft,)


@pytest.mark.parametrize("draft_prefix", ["model.layers.4", "model.mtp.layers.0"])
def test_kvpp_draft_indexer_follows_draft_attention_completion(monkeypatch, draft_prefix):
    target = "model.layers.0.self_attn.attn"
    draft = f"{draft_prefix}.self_attn.attn"
    indexer = f"{draft_prefix}.self_attn.indexer.k_cache"
    specs = dict.fromkeys((target, draft, indexer), object())
    worker = MagicMock(tp_size=2, tp_rank=0, specs=specs)
    monkeypatch.setattr(LayerwisePoolConnector, "_init_worker", lambda self: setattr(self, "connector_worker", worker))
    monkeypatch.setattr(facade, "KVPPConfig", SimpleNamespace(from_vllm_config=lambda _: SimpleNamespace(size=2)))
    monkeypatch.setattr(facade, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(facade, "map_kvpp_layers_to_owners", lambda *_: {target: 1})
    monkeypatch.setattr(facade, "find_draft_layers", lambda *_: {draft, indexer})
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=True, is_kv_consumer=False))
    connector = LayerwisePoolConnector(config, KVConnectorRole.WORKER, MagicMock())
    connector.register_kv_caches(specs)
    worker.register_kv_caches.assert_called_once_with(specs, owned_layers={draft, indexer})
    connector.bind_connector_metadata(LayerwisePoolConnectorMetadata())
    connector.save_kv_layer(target, None, None)
    assert [call.args[0] for call in worker.save_kv_layer.call_args_list] == [target]
    worker.save_kv_layer.reset_mock()
    connector.save_kv_layer(draft, None, None)
    assert [call.args[0] for call in worker.save_kv_layer.call_args_list] == [draft, indexer]
    # Save completion grouping must not alter KVPP's independent allocations.
    allocated = build_layer_cache_bundles(specs, {draft, indexer})
    assert allocated[draft] == (draft,) and allocated[indexer] == (indexer,)
