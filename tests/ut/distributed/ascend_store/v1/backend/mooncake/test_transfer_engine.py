"""v1 registration ownership, rollback, and isolation from legacy connectors."""

import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from tests.ut.distributed.ascend_store.v1.backend.mooncake.fixtures import _make_mooncake_registration_backend
from tests.ut.distributed.ascend_store.v1.infrastructure.resources_fixtures import run_backend_resource_lifecycle
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import BackendSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.mooncake import MooncakeBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.mooncake import backend as mooncake_adapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.mooncake.transfer_engine import (
    RegisteredTransferEngine,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.registration import (
    BufferRegistrationError,
    BufferReleaseError,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE


@pytest.fixture
def registered_engine():
    native = MagicMock()
    native.initialize.return_value = 0
    native.register_memory.return_value = 0
    native.unregister_memory.return_value = 0
    manager = RegisteredTransferEngine()
    with patch.object(sys.modules["mooncake.engine"], "TransferEngine", return_value=native, create=True):
        assert manager.get_transfer_engine("host", None) is native
    return manager, native


def test_shared_region_releases_only_after_last_reference(registered_engine):
    manager, native = registered_engine
    first = manager.acquire_registration([100], [10])
    second = manager.acquire_registration([100], [10])
    first.close()
    native.unregister_memory.assert_not_called()
    second.close()
    second.close()
    native.register_memory.assert_called_once_with(100, 10)
    native.unregister_memory.assert_called_once_with(100)


def test_concurrent_clients_share_one_native_registration(registered_engine):
    manager, native = registered_engine
    with ThreadPoolExecutor(max_workers=4) as executor:
        registrations = list(executor.map(lambda _: manager.acquire_registration([100], [10]), range(16)))
        list(executor.map(lambda registration: registration.close(), registrations))
    native.register_memory.assert_called_once_with(100, 10)
    native.unregister_memory.assert_called_once_with(100)


@pytest.mark.parametrize("legacy_first", (True, False))
def test_legacy_registration_survives_v1_release(legacy_first):
    manager = RegisteredTransferEngine()
    legacy = GlobalTE()
    native, legacy_native = MagicMock(), MagicMock()
    for engine in (native, legacy_native):
        engine.initialize.return_value = 0
        engine.register_memory.return_value = 0
        engine.unregister_memory.return_value = 0
    engines = [legacy_native, native] if legacy_first else [native, legacy_native]
    with patch.object(sys.modules["mooncake.engine"], "TransferEngine", side_effect=engines, create=True):
        if legacy_first:
            legacy.get_transfer_engine("host", None)
            legacy.register_buffer([100], [10])
        manager.get_transfer_engine("host", None)
        registration = manager.acquire_registration([100], [10])
        if not legacy_first:
            legacy.get_transfer_engine("host", None)
            legacy.register_buffer([100], [10])
    registration.close()
    legacy.register_buffer([200], [20])
    native.unregister_memory.assert_called_once_with(100)
    legacy_native.register_memory.assert_called_once_with(100, 10)
    legacy_native.unregister_memory.assert_not_called()
    assert legacy.is_register_buffer
    assert not hasattr(legacy, "acquire_registration")


@pytest.mark.parametrize("failure", ("return_code", "initialize_exception", "constructor_exception"))
def test_initialization_failure_preserves_upstream_no_retry_policy(failure):
    manager = RegisteredTransferEngine()
    native = MagicMock()
    native.initialize.return_value = -7
    if failure == "initialize_exception":
        native.initialize.side_effect = RuntimeError("native init failed")
    factory = MagicMock(return_value=native)
    if failure == "constructor_exception":
        factory.side_effect = RuntimeError("native construction failed")
    with patch.object(sys.modules["mooncake.engine"], "TransferEngine", factory, create=True):
        with pytest.raises(RuntimeError):
            manager.get_transfer_engine("host", None)
        with pytest.raises(RuntimeError, match="previously failed"):
            manager.get_transfer_engine("host", None)
    factory.assert_called_once_with()
    assert manager._engine_owner.transfer_engine is None


def test_partial_registration_failure_rolls_back_prefix(registered_engine):
    manager, native = registered_engine
    native.register_memory.side_effect = [0, -7]
    with pytest.raises(RuntimeError, match="registration failed"):
        manager.acquire_registration([100, 200], [10, 20])
    native.unregister_memory.assert_called_once_with(100)


def test_partial_rollback_preserves_retryable_owner(registered_engine):
    manager, native = registered_engine
    native.register_memory.side_effect = [0, -7]
    native.unregister_memory.side_effect = [-9, 0]
    with pytest.raises(BufferRegistrationError, match="rollback is incomplete") as error:
        manager.acquire_registration([100, 200], [10, 20])
    registration = error.value.registration
    assert [(region.address, region.size) for region in registration.regions] == [(100, 10)]
    registration.close()
    assert registration.released
    assert native.unregister_memory.call_args_list == [call(100), call(100)]


def test_failed_release_keeps_reference_for_retry(registered_engine):
    manager, native = registered_engine
    registration = manager.acquire_registration([100], [10])
    native.unregister_memory.side_effect = [-9, 0]
    with pytest.raises(BufferReleaseError):
        registration.close()
    second = manager.acquire_registration([100], [10])
    registration.close()
    assert native.unregister_memory.call_count == 1
    second.close()
    assert native.unregister_memory.call_count == 2
    native.register_memory.assert_called_once_with(100, 10)


def test_conflicting_size_rolls_back_only_new_reference(registered_engine):
    manager, native = registered_engine
    existing = manager.acquire_registration([100], [10])
    with pytest.raises(ValueError, match="registered with size 10"):
        manager.acquire_registration([100, 100], [10, 20])
    native.unregister_memory.assert_not_called()
    existing.close()
    native.unregister_memory.assert_called_once_with(100)


@pytest.mark.parametrize("addresses,sizes", (([100], []), ([-1], [10]), ([100], [0])))
def test_invalid_regions_do_not_reach_native_registration(registered_engine, addresses, sizes):
    manager, native = registered_engine
    with pytest.raises(ValueError):
        manager.acquire_registration(addresses, sizes)
    native.register_memory.assert_not_called()


def test_backend_setup_and_registration_use_same_v1_engine(registered_engine, monkeypatch):
    manager, native = registered_engine
    monkeypatch.setattr(mooncake_adapter, "registered_engine", manager)
    native.get_rpc_port.return_value = 1234
    registrations = []
    for _ in range(2):
        store = MagicMock()
        backend = _make_mooncake_registration_backend(store)
        backend._use_store_independent_te = False
        backend._contribute_memory = True
        backend.config = SimpleNamespace(
            metadata_server="metadata",
            global_segment_size=1024,
            local_buffer_size=64,
            protocol="ascend",
            device_name="",
            master_server_address="master",
        )
        backend._setup_native_store(store, "host", {})
        registrations.append(backend.register_buffer([100], [10]))
        assert store.setup.call_args.kwargs["engine"] is native.get_engine.return_value
        assert store.setup.call_args.kwargs["local_hostname"] == "host:1234"
    registrations[0].close()
    native.unregister_memory.assert_not_called()
    registrations[1].close()
    native.register_memory.assert_called_once_with(100, 10)
    native.unregister_memory.assert_called_once_with(100)


def test_resources_release_v1_engine_before_closing_store(registered_engine, monkeypatch):
    manager, native = registered_engine
    monkeypatch.setattr(mooncake_adapter, "registered_engine", manager)
    store = MagicMock()
    store.close.return_value = 0
    backend = _make_mooncake_registration_backend(store)
    backend._use_store_independent_te = False
    lifecycle = MagicMock()
    lifecycle.attach_mock(native.register_memory, "register")
    lifecycle.attach_mock(native.unregister_memory, "unregister")
    lifecycle.attach_mock(store.close, "close")
    run_backend_resource_lifecycle(backend, BackendSpec("mooncake", MooncakeBackend, None, True))
    assert [entry[0] for entry in lifecycle.mock_calls] == ["register", "unregister", "close"]
