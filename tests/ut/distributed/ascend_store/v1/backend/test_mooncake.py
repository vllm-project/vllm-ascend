"""Mooncake native result normalization, registration rollback, and client close."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from tests.ut.distributed.ascend_store.v1.worker.resource_fixtures import (
    run_backend_resource_lifecycle,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    BufferRegistrationError,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    mooncake as mooncake_adapter,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.mooncake import (
    MooncakeBackend,
)


def _make_mooncake_registration_backend(store: MagicMock) -> MooncakeBackend:
    backend = MooncakeBackend.__new__(MooncakeBackend)
    backend._store = store
    backend._use_store_independent_te = True
    backend._use_fabric_mem = False
    backend._initialized = True
    backend._lazy_init = False
    backend._closed = False
    return backend


def test_named_mooncake_worker_unregisters_store_regions_before_client_close() -> None:
    store = MagicMock()
    backend = _make_mooncake_registration_backend(store)
    store.register_buffer.return_value = 0
    store.unregister_buffer.return_value = 0
    store.close.return_value = 0
    lifecycle = MagicMock()
    lifecycle.attach_mock(store.register_buffer, "register")
    lifecycle.attach_mock(store.unregister_buffer, "unregister")
    lifecycle.attach_mock(store.close, "close")
    backend_spec = BackendSpec("mooncake", MooncakeBackend, None, True)

    run_backend_resource_lifecycle(backend, backend_spec)

    assert [entry[0] for entry in lifecycle.mock_calls] == ["register", "unregister", "close"]


def test_v1_mooncake_adapter_preserves_native_bulk_store_contract() -> None:
    store = MagicMock()
    store.batch_put_from_multi_buffers.return_value = [0]
    backend = _make_mooncake_registration_backend(store)
    backend.config = SimpleNamespace(
        preferred_segment=True,
        prefer_alloc_in_same_node=False,
    )
    backend._local_segment = "segment-0"

    assert backend.store(["key"], [[100]], [[10]]) == (0,)

    keys, addresses, sizes, replicate_config = store.batch_put_from_multi_buffers.call_args.args
    assert (keys, addresses, sizes) == (["key"], [[100]], [[10]])
    assert replicate_config.preferred_segment == "segment-0"
    assert not replicate_config.prefer_alloc_in_same_node


def test_mooncake_store_registration_failure_rolls_back_owned_prefix() -> None:
    store = MagicMock()
    store.register_buffer.side_effect = [0, -7]
    store.unregister_buffer.return_value = 0
    backend = _make_mooncake_registration_backend(store)

    with pytest.raises(RuntimeError, match="registration failed"):
        backend.register_buffer([100, 200], [10, 20])

    assert store.register_buffer.call_args_list == [call(100, 10), call(200, 20)]
    store.unregister_buffer.assert_called_once_with(100)


def test_mooncake_registration_rollback_failure_returns_retryable_owner() -> None:
    store = MagicMock()
    store.register_buffer.side_effect = [0, -7]
    store.unregister_buffer.side_effect = [-9, 0]
    backend = _make_mooncake_registration_backend(store)

    with pytest.raises(BufferRegistrationError, match="rollback is incomplete") as error:
        backend.register_buffer([100, 200], [10, 20])

    assert tuple((region.address, region.size) for region in error.value.registration.regions) == ((100, 10),)
    error.value.registration.close()
    assert store.unregister_buffer.call_args_list == [call(100), call(100)]


def test_mooncake_global_registration_preserves_incomplete_shared_owner() -> None:
    class SharedRegistrationError(RuntimeError):
        def __init__(self, operation_error, rollback_error, registration) -> None:
            self.operation_error = operation_error
            self.rollback_error = rollback_error
            self.registration = registration

    store = MagicMock()
    backend = _make_mooncake_registration_backend(store)
    backend._use_store_independent_te = False
    shared_registration = MagicMock()
    operation_error = RuntimeError("registration failed")
    rollback_error = RuntimeError("rollback failed")
    shared_manager = MagicMock()
    shared_manager.acquire_registration.side_effect = SharedRegistrationError(
        operation_error,
        rollback_error,
        shared_registration,
    )

    with (
        patch.object(mooncake_adapter, "global_te", shared_manager),
        patch.object(mooncake_adapter, "RegistrationAcquisitionError", SharedRegistrationError),
        pytest.raises(BufferRegistrationError, match="rollback is incomplete") as error,
    ):
        backend.register_buffer([100], [10])

    assert error.value.registration is shared_registration


def test_mooncake_close_failure_keeps_client_for_retry() -> None:
    store = MagicMock()
    store.close.side_effect = [-7, 0]
    backend = _make_mooncake_registration_backend(store)

    with pytest.raises(RuntimeError, match="close failed"):
        backend.close()

    assert backend._store is store
    assert not backend._closed
    backend.close()
    assert backend._store is None
    assert backend._closed


def test_mooncake_setup_failure_closes_constructed_client() -> None:
    store = MagicMock()
    store.setup.return_value = -7
    store.close.return_value = 0
    backend = _make_mooncake_registration_backend(store)
    backend.parallel_config = SimpleNamespace()
    backend.device_id = 0
    backend._contribute_memory = True
    backend._use_fabric_mem = True
    backend._use_store_independent_te = False
    backend.config = SimpleNamespace(
        metadata_server="metadata",
        global_segment_size=1024,
        local_buffer_size=1024,
        protocol="ascend",
        device_name="",
        master_server_address="master",
        enable_ssd_offload=False,
        ssd_offload_path="",
        tenant_id="default",
    )

    with (
        patch("mooncake.store.MooncakeDistributedStore", return_value=store, create=True),
        patch.object(MooncakeBackend, "set_device"),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.mooncake.get_ip",
            return_value="127.0.0.1",
        ),
        pytest.raises(RuntimeError, match="initialization failed"),
    ):
        backend._setup_store()

    store.close.assert_called_once_with()
