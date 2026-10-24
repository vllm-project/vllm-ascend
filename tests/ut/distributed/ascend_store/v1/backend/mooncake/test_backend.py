"""Mooncake native result normalization, registration rollback, and client close."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, call, patch

import pytest

from tests.ut.distributed.ascend_store.v1.backend.mooncake.fixtures import _make_mooncake_registration_backend
from tests.ut.distributed.ascend_store.v1.helpers import FakeBackend, FakeLoadStartGate, begin_step, make_worker
from tests.ut.distributed.ascend_store.v1.infrastructure.resources_fixtures import (
    run_backend_resource_lifecycle,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    BufferRegistrationError,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.mooncake import (
    MooncakeBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.mooncake import (
    backend as mooncake_adapter,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.results import BackendResultError
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.messages import (
    LoadCommand,
    RangeStoreCommand,
)

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import RegistrationAcquisitionError


def _load_global_te_type():
    source = Path(mooncake_adapter.__file__).resolve().parents[5] / "utils" / "mooncake_transfer_engine.py"
    spec = importlib.util.spec_from_file_location("_ascend_store_test_global_te", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GlobalTE


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
    backend.config = SimpleNamespace(preferred_segment=True, prefer_alloc_in_same_node=False)
    backend._local_segment = "segment-0"

    assert backend.store(["key"], [[100]], [[10]]) == (0,)

    keys, addresses, sizes, replicate_config = store.batch_put_from_multi_buffers.call_args.args
    assert (keys, addresses, sizes) == (["key"], [[100]], [[10]])
    assert replicate_config.preferred_segment == "segment-0"
    assert not replicate_config.prefer_alloc_in_same_node


def test_mooncake_creates_sessions_with_native_commit_and_release():
    store = MagicMock()
    backend = _make_mooncake_registration_backend(store)
    backend.config = SimpleNamespace(preferred_segment=False, prefer_alloc_in_same_node=False)
    store.batch_get_session_start.return_value = [0]
    store.batch_get_session_end.return_value = 0
    store.batch_put_session_start.return_value = [0]
    store.batch_put_session_end.return_value = [0]

    sessions = backend.create_sessions()
    assert sessions.start_load(["read"], [64]) == (0,)
    assert sessions.start_store(["write"], [64]) == (0,)
    assert sessions.commit_store(["write"]) == (0,)
    assert not sessions.pending_store_keys
    sessions.close()
    store.batch_get_session_end.assert_called_once_with(["read"])
    store.batch_put_session_end.assert_called_once_with(["write"])
    store.batch_put_session_revoke.assert_not_called()


def test_mooncake_session_factory_rejects_missing_native_support():
    store = MagicMock()
    backend = _make_mooncake_registration_backend(store)
    store.batch_get_session_start = None
    with pytest.raises(RuntimeError, match="Missing methods: batch_get_session_start"):
        backend.create_sessions()
    store.batch_put_session_start.assert_not_called()


@pytest.mark.parametrize(
    "operation,args",
    (("set_device", ()), ("create_sessions", ()), ("exists", (["key"],)), ("load", (["key"], [[100]], [[10]]))),
)
def test_closed_mooncake_rejects_reads_and_device_setup(operation, args) -> None:
    store = MagicMock()
    store.close.return_value = 0
    backend = _make_mooncake_registration_backend(store)
    backend._lazy_init = True
    backend.close()
    store.reset_mock()

    with pytest.raises(RuntimeError, match="Backend is closed"):
        getattr(backend, operation)(*args)

    assert store.mock_calls == []


@pytest.mark.parametrize("direction", ("store", "load"))
@pytest.mark.parametrize(
    "native_results,expected",
    (
        ([524288, 24, 0], (0, 0, 0)),
        ([524288, -707, -7], (0, -707, -7)),
        ([0, 8, 23], (0, 0, 0)),
        ([24, 25, 0], (0, 0, 0)),
    ),
)
def test_mooncake_range_copy_normalizes_native_byte_counts(direction, native_results, expected):
    store = MagicMock()
    method = (
        store.batch_put_from_multi_buffer_ranges if direction == "store" else store.batch_get_into_multi_buffer_ranges
    )
    method.return_value = native_results
    backend = _make_mooncake_registration_backend(store)
    copy = backend.copy_key_range_store if direction == "store" else backend.copy_key_range_load
    args = (
        ["large", "split", "empty"],
        [[100, 200], [300, 400], []],
        [[262144, 262144], [8, 16], []],
        [[0, 262144], [8, 32], []],
    )
    assert copy(*args) == expected
    method.assert_called_once_with(*args)


@pytest.mark.parametrize("direction", ("store", "load"))
@pytest.mark.parametrize("native_results", (None, [], [32, 32], [True], [32.0]))
def test_mooncake_range_copy_rejects_malformed_results(direction, native_results) -> None:
    store = MagicMock()
    method = (
        store.batch_put_from_multi_buffer_ranges if direction == "store" else store.batch_get_into_multi_buffer_ranges
    )
    method.return_value = native_results
    backend = _make_mooncake_registration_backend(store)
    copy = backend.copy_key_range_store if direction == "store" else backend.copy_key_range_load

    with pytest.raises(BackendResultError):
        copy(["key"], [[100]], [[32]], [[0]])


@pytest.mark.parametrize("code", (0, -707, 32))
def test_mooncake_range_sessions_preserve_native_status_codes(code) -> None:
    store = MagicMock()
    backend = _make_mooncake_registration_backend(store)
    backend.config = SimpleNamespace(preferred_segment=False, prefer_alloc_in_same_node=False)
    for method in (
        store.batch_put_session_start,
        store.batch_put_session_end,
        store.batch_put_session_revoke,
        store.batch_get_session_start,
    ):
        method.return_value = [code]
    store.batch_get_session_end.return_value = code

    assert backend.start_key_range_store(["key"], [32]) == (code,)
    assert backend.commit_key_range_store(["key"]) == (code,)
    assert backend.revoke_key_range_store(["key"]) == (code,)
    assert backend.start_key_range_load(["key"]) == (code,)
    if code == 0:
        backend.finish_key_range_load(["key"])
    else:
        with pytest.raises(RuntimeError, match="batch_get_session_end failed"):
            backend.finish_key_range_load(["key"])


@pytest.mark.parametrize("failed_copy", (None, "store", "load"))
@pytest.mark.parametrize("successful_bytes", (None, 0, 1))
def test_mooncake_native_byte_counts_reach_layerwise_store_and_load(failed_copy, successful_bytes) -> None:
    store = MagicMock()
    backend = _make_mooncake_registration_backend(store)
    backend.device_id = 0
    backend.config = SimpleNamespace(preferred_segment=False, prefer_alloc_in_same_node=False)
    store.batch_put_session_start.side_effect = lambda keys, sizes, config: [0] * len(keys)
    store.batch_put_session_end.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_session_revoke.side_effect = lambda keys: [0] * len(keys)
    store.batch_get_session_start.side_effect = lambda keys: [0] * len(keys)
    store.batch_get_session_end.return_value = 0
    store.batch_put_from_multi_buffer_ranges.side_effect = lambda keys, addresses, sizes, offsets: [
        -707 if failed_copy == "store" else (sum(row) if successful_bytes is None else successful_bytes)
        for row in sizes
    ]
    store.batch_get_into_multi_buffer_ranges.side_effect = lambda keys, addresses, sizes, offsets: [
        -707 if failed_copy == "load" else (sum(row) if successful_bytes is None else successful_bytes) for row in sizes
    ]
    worker, resources, _ = make_worker(
        cast(FakeBackend, backend), layerwise=True, start_gate_factory=lambda: FakeLoadStartGate(opened=True)
    )
    try:
        command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17)
        begin_step(worker, store=(command,))
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
        worker.finish_step()
        assert store.batch_put_from_multi_buffer_ranges.call_count == 2
        assert store.batch_put_session_end.call_count == (0 if failed_copy == "store" else 1)
        assert store.batch_put_session_revoke.call_count == (1 if failed_copy == "store" else 0)
        assert worker.take_released_store_job_ids() == {17}
        worker.end_step()

        if failed_copy != "store":
            begin_step(worker, load=(LoadCommand("request", 0, 4, ((1,),), (b"a",)),))
            worker.start_load()
            worker.wait_for_layer_load("layers.0.group.0")
            worker.wait_for_layer_load("layers.1.group.0")
            result = worker.collect_load_result()
            assert bool(result.failed_locations) == (failed_copy == "load")
            assert store.batch_get_into_multi_buffer_ranges.call_count == 2
            store.batch_get_session_end.assert_called_once()
            worker.end_step()
    finally:
        worker.close()
    assert resources.closed


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
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.mooncake.backend.get_ip",
            return_value="127.0.0.1",
        ),
        pytest.raises(RuntimeError, match="initialization failed"),
    ):
        backend._setup_store()

    store.close.assert_called_once_with()


class TestGlobalTransferEngineRegistration(unittest.TestCase):
    def _make_global_te(self):
        manager = _load_global_te_type()()
        manager.transfer_engine = MagicMock()
        manager.transfer_engine.register_memory.return_value = 0
        manager.transfer_engine.unregister_memory.return_value = 0
        return manager

    def test_shared_region_is_natively_registered_once_and_reference_counted(self):
        manager = self._make_global_te()

        first = manager.acquire_registration([100], [10])
        second = manager.acquire_registration([100], [10])
        first.close()
        manager.transfer_engine.unregister_memory.assert_not_called()
        second.close()

        manager.transfer_engine.register_memory.assert_called_once_with(100, 10)
        manager.transfer_engine.unregister_memory.assert_called_once_with(100)
        self.assertFalse(manager.is_register_buffer)

    def test_classic_registration_retains_one_process_lifetime_lease(self):
        manager = self._make_global_te()

        self.assertIsNone(manager.register_buffer([100], [10]))
        self.assertIsNone(manager.register_buffer([200], [20]))

        manager.transfer_engine.register_memory.assert_called_once_with(100, 10)
        manager.transfer_engine.unregister_memory.assert_not_called()
        self.assertTrue(manager.is_register_buffer)

    def test_classic_registration_protects_region_after_v1_lease_closes(self):
        manager = self._make_global_te()

        v1_registration = manager.acquire_registration([100], [10])
        manager.register_buffer([100], [10])
        v1_registration.close()

        manager.transfer_engine.register_memory.assert_called_once_with(100, 10)
        manager.transfer_engine.unregister_memory.assert_not_called()
        self.assertTrue(manager.is_register_buffer)

    def test_failed_initialization_does_not_publish_global_engine(self):
        manager = _load_global_te_type()()
        failed_engine = MagicMock()
        failed_engine.initialize.return_value = -7
        initialized_engine = MagicMock()
        initialized_engine.initialize.return_value = 0
        engine_factory = MagicMock(side_effect=(failed_engine, initialized_engine))

        with patch.object(sys.modules["mooncake.engine"], "TransferEngine", engine_factory, create=True):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                manager.get_transfer_engine("host", None)
            self.assertIsNone(manager.transfer_engine)
            self.assertIs(manager.get_transfer_engine("host", None), initialized_engine)

        self.assertEqual(engine_factory.call_count, 2)
        failed_engine.initialize.assert_called_once_with("host", "P2PHANDSHAKE", "ascend", "")
        initialized_engine.initialize.assert_called_once_with("host", "P2PHANDSHAKE", "ascend", "")

    def test_partial_registration_failure_rolls_back_owned_prefix(self):
        manager = self._make_global_te()
        manager.transfer_engine.register_memory.side_effect = [0, -7]

        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            manager.acquire_registration([100, 200], [10, 20])

        manager.transfer_engine.unregister_memory.assert_called_once_with(100)
        self.assertFalse(manager.is_register_buffer)

    def test_partial_rollback_failure_returns_retryable_registration(self):
        manager = self._make_global_te()
        manager.transfer_engine.register_memory.side_effect = [0, -7]
        manager.transfer_engine.unregister_memory.side_effect = [-9, 0]

        with self.assertRaisesRegex(RuntimeError, "rollback is incomplete") as context:
            manager.acquire_registration([100, 200], [10, 20])

        # Isolated loading has a distinct exception class; narrow the known rollback-failure contract.
        registration_error = cast("RegistrationAcquisitionError", context.exception)
        self.assertEqual(
            tuple((region.address, region.size) for region in registration_error.registration.regions),
            ((100, 10),),
        )
        registration_error.registration.close()
        self.assertFalse(manager.is_register_buffer)

    def test_classic_partial_rollback_failure_retains_process_owner(self):
        manager = self._make_global_te()
        manager.transfer_engine.register_memory.side_effect = [0, -7]
        manager.transfer_engine.unregister_memory.return_value = -9

        with self.assertRaisesRegex(RuntimeError, "rollback is incomplete"):
            manager.register_buffer([100, 200], [10, 20])
        manager.register_buffer([300], [30])

        self.assertTrue(manager.is_register_buffer)
        self.assertEqual(manager.transfer_engine.register_memory.call_count, 2)
