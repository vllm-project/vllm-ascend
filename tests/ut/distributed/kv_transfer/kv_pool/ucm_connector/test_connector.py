# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock, sentinel

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from vllm_ascend.distributed.kv_transfer.kv_pool.ucm_connector import connector as module


@pytest.fixture
def connector():
    subject = module.UCMConnectorV1.__new__(module.UCMConnectorV1)
    subject._ucm_engine = SimpleNamespace()
    return subject


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_initialization_preserves_config_role_and_cache(monkeypatch, role):
    base_init = MagicMock(return_value=None)
    factory = MagicMock()
    monkeypatch.setattr(module.KVConnectorBase_V1, "__init__", base_init)
    monkeypatch.setattr(module, "UCMConnector", factory)
    config = SimpleNamespace(kv_transfer_config=sentinel.transfer_config)
    subject = module.UCMConnectorV1(config, role, sentinel.cache)
    base_init.assert_called_once_with(vllm_config=config, role=role, kv_cache_config=sentinel.cache)
    factory.assert_called_once_with(config, role, sentinel.cache)
    assert subject._ucm_engine is factory.return_value


def test_initialization_requires_transfer_config(monkeypatch):
    monkeypatch.setattr(module.KVConnectorBase_V1, "__init__", MagicMock(return_value=None))
    factory = MagicMock()
    monkeypatch.setattr(module, "UCMConnector", factory)
    with pytest.raises(AssertionError):
        module.UCMConnectorV1(SimpleNamespace(kv_transfer_config=None), KVConnectorRole.WORKER, None)
    factory.assert_not_called()


@pytest.mark.parametrize(
    "name,args,kwargs,returns_value",
    [
        ("has_connector_metadata", (), {}, True),
        ("register_kv_caches", (sentinel.caches,), {}, False),
        ("start_load_kv", (sentinel.context,), {"step": 3}, False),
        ("wait_for_layer_load", ("layer.1",), {}, False),
        ("save_kv_layer", ("layer.1", sentinel.tensor, sentinel.metadata), {"step": 3}, False),
        ("wait_for_save", (), {}, False),
        ("clear_connector_metadata", (), {}, False),
        ("bind_connector_metadata", (sentinel.metadata,), {}, False),
        ("get_block_ids_with_load_errors", (), {}, True),
        ("get_num_new_matched_tokens", (sentinel.request, 32), {}, True),
        ("update_state_after_alloc", (sentinel.request, sentinel.blocks, 16), {}, False),
        ("build_connector_meta", (sentinel.output,), {}, True),
        ("request_finished", (sentinel.request, [1, 2]), {}, True),
        ("request_finished_all_groups", (sentinel.request, ([1], [2])), {}, True),
        ("get_finished", ({"r1"},), {}, True),
        ("update_connector_output", (sentinel.output,), {}, False),
    ],
)
def test_delegates_explicit_protocol_arguments_and_results(connector, name, args, kwargs, returns_value):
    call = MagicMock(return_value=sentinel.result)
    setattr(connector._ucm_engine, name, call)
    result = getattr(connector, name)(*args, **kwargs)
    assert result is (sentinel.result if returns_value else None)
    call.assert_called_once_with(*args, **kwargs)


@pytest.mark.parametrize(
    "name,args,returns_value",
    [
        ("shutdown", (), False),
        ("set_host_xfer_buffer_ops", (sentinel.copy,), False),
        ("handle_preemptions", (sentinel.metadata,), False),
        ("build_connector_worker_meta", (), True),
        ("take_events", (), True),
        ("get_kv_connector_stats", (), True),
        ("get_kv_connector_kv_cache_events", (), True),
        ("get_handshake_metadata", (), True),
        ("set_xfer_handshake_metadata", ({1: sentinel.metadata},), False),
        ("get_finished_count", (), True),
        ("reset_cache", (), True),
    ],
)
def test_reserved_hooks_reach_inner_connector(connector, name, args, returns_value):
    call = MagicMock(return_value=sentinel.result)
    connector._ucm_engine.connector = SimpleNamespace(**{name: call})
    result = getattr(connector, name)(*args)
    assert result is (sentinel.result if returns_value else None)
    call.assert_called_once_with(*args)


def test_dispatcher_implementation_takes_precedence_over_inner(connector):
    class Dispatcher:
        def reset_cache(self):
            return True

    dispatcher = Dispatcher()
    dispatcher.connector = MagicMock()
    connector._ucm_engine = dispatcher
    assert connector.reset_cache() is True
    dispatcher.connector.reset_cache.assert_not_called()


def test_missing_and_noncallable_hooks_have_optional_defaults(connector):
    assert connector.build_connector_worker_meta() is None
    assert connector.take_events() == ()
    connector._ucm_engine.reset_cache = False
    assert connector.reset_cache() is None


def test_hook_exception_is_propagated(connector):
    connector._ucm_engine.reset_cache = MagicMock(side_effect=RuntimeError("SDK reset failed"))
    with pytest.raises(RuntimeError, match="SDK reset failed"):
        connector.reset_cache()


def test_stats_and_prometheus_factories_forward_arguments(monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr(module, "UCMConnector", factory)
    assert module.UCMConnectorV1.build_kv_connector_stats() is factory.build_kv_connector_stats.return_value
    factory.build_kv_connector_stats.assert_called_once_with(None)
    args = (sentinel.config, {int: str}, ["model"], {0: ["m"]})
    assert module.UCMConnectorV1.build_prom_metrics(*args) is factory.build_prom_metrics.return_value
    factory.build_prom_metrics.assert_called_once_with(*args)
