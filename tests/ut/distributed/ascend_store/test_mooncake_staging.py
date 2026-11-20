# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, call

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_staging
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import BatchResultShapeError
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.buffer import BufferDescriptor, MemoryLocation


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(mooncake_staging, "_HOST_STAGING_BYTES", 16)
    host = ctypes.create_string_buffer(b"...............................")
    device = ctypes.create_string_buffer(8)
    host_ptr = ctypes.addressof(host)
    device_ptr = ctypes.addressof(device)
    sdk = Mock()
    sdk.register_buffer.return_value = 0
    sdk.unregister_buffer.return_value = 0
    bulk = mooncake_staging.MooncakeHostStaging(sdk)
    try:
        bulk.register(
            [
                BufferDescriptor(address=host_ptr, size_bytes=ctypes.sizeof(host), memory_location=MemoryLocation.HOST),
                BufferDescriptor(address=device_ptr, size_bytes=ctypes.sizeof(device)),
            ]
        )
        assert sdk.register_buffer.call_args_list == [
            call(bulk._send.base, 16),
            call(bulk._recv.base, 16),
            call(device_ptr, ctypes.sizeof(device)),
        ]
        yield bulk, sdk, host, device
    finally:
        bulk.unregister()
        bulk.close()
    assert sdk.unregister_buffer.call_args_list == [call(device_ptr), call(bulk._recv.base), call(bulk._send.base)]
    sdk.close.assert_not_called()


def test_host_put_packs_payload_without_copying_gaps_and_reuses_bounded_buffer(registered):
    bulk, sdk, host, _ = registered
    host.raw = b"abcdefghijklmnopqrstuvwx0123456."
    host_ptr = ctypes.addressof(host)
    seen = []

    def save(keys, addrs, sizes, config):
        seen.append((keys, [b"".join(ctypes.string_at(a, n) for a, n in zip(p, s)) for p, s in zip(addrs, sizes)]))
        return [0] * len(keys)

    sdk.batch_put_from_multi_buffers.side_effect = save
    assert bulk.put(
        ["a", "b"],
        [[host_ptr, host_ptr + 8], [host_ptr + 18]],
        [[4, 6], [10]],
        None,
        memory_location=MemoryLocation.HOST,
    )
    assert seen == [(["a"], [b"abcdijklmn"]), (["b"], [b"stuvwx0123"])]


def test_get_preserves_key_order_and_does_not_publish_short_host_read(registered):
    bulk, sdk, host, device = registered
    host_ptr = ctypes.addressof(host)
    device_ptr = ctypes.addressof(device)

    def load(keys, addrs, sizes):
        for ptrs, lengths in zip(addrs, sizes):
            for ptr, length in zip(ptrs, lengths):
                ctypes.memset(ptr, ord("k"), length)
        return [4 if key == "short" else sum(parts) for key, parts in zip(keys, sizes)]

    sdk.batch_get_into_multi_buffers.side_effect = load
    assert bulk.get(
        ["short", "good"],
        [[host_ptr], [host_ptr + 8]],
        [[8], [8]],
        memory_location=MemoryLocation.HOST,
    ) == [-1, 0]
    assert bulk.get(["index"], [[device_ptr]], [[8]], memory_location=MemoryLocation.DEVICE) == [0]
    assert host.raw[:16] == b"........kkkkkkkk"
    assert device.raw == b"kkkkkkkk"


def test_get_rejects_malformed_sdk_results(registered):
    bulk, sdk, host, _ = registered
    sdk.batch_get_into_multi_buffers.return_value = []
    with pytest.raises(BatchResultShapeError, match="0 results for 1"):
        bulk.get(["bad"], [[ctypes.addressof(host)]], [[8]], memory_location=MemoryLocation.HOST)
    assert host.raw[:8] == b"........"


@pytest.mark.parametrize(("offset", "size", "message"), [(28, 8, "outside"), (0, 18, "capacity")])
def test_invalid_host_geometry_never_reaches_sdk(registered, offset, size, message):
    bulk, sdk, host, _ = registered
    with pytest.raises(ValueError, match=message):
        bulk.get(["bad"], [[ctypes.addressof(host) + offset]], [[size]], memory_location=MemoryLocation.HOST)
    sdk.batch_get_into_multi_buffers.assert_not_called()


@pytest.mark.parametrize("outer_timeout", ["15", "10"])
def test_mooncake_mixed_config_prevents_early_outer_timeout(monkeypatch, tmp_path, outer_timeout):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

    config_path = tmp_path / "mooncake.json"
    config_path.write_text('{"protocol": "ascend"}', encoding="utf-8")
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ASCEND_GLOBAL_RESOURCE_CONFIG", "{}")
    monkeypatch.setenv("MC_TRANSFER_TIMEOUT", outer_timeout)
    monkeypatch.setenv("ASCEND_TRANSFER_TIMEOUT", "5000")
    monkeypatch.delenv("ASCEND_ENABLE_USE_FABRIC_MEM", raising=False)
    if outer_timeout == "15":
        MooncakeBackend.validate_hisparse_colocated_config()
    else:
        with pytest.raises(ValueError, match="MC_TRANSFER_TIMEOUT"):
            MooncakeBackend.validate_hisparse_colocated_config()


def test_mooncake_partial_registration_rolls_back_device_and_staging(monkeypatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

    monkeypatch.setattr(mooncake_staging, "_HOST_STAGING_BYTES", 16)
    backend = MooncakeBackend.__new__(MooncakeBackend)
    sdk = Mock()
    sdk.register_buffer.side_effect = [0, 0, 0, -600]
    sdk.unregister_buffer.return_value = 0
    backend.store = sdk
    backend._store_initialized = backend._use_store_independent_te = True
    backend._host_staging = None

    with pytest.raises(RuntimeError, match="registration failed.*-600"):
        backend.register_buffer(
            [
                BufferDescriptor(address=100, size_bytes=32, memory_location=MemoryLocation.HOST),
                BufferDescriptor(address=200, size_bytes=64),
                BufferDescriptor(address=300, size_bytes=128),
            ]
        )

    send_ptr = sdk.register_buffer.call_args_list[0].args[0]
    recv_ptr = sdk.register_buffer.call_args_list[1].args[0]
    assert sdk.register_buffer.call_args_list == [call(send_ptr, 16), call(recv_ptr, 16), call(200, 64), call(300, 128)]
    assert sdk.unregister_buffer.call_args_list == [call(200), call(recv_ptr), call(send_ptr)]
    sdk.close.assert_called_once_with()
    assert backend._host_staging is None and backend.store is None
    assert not backend._store_initialized


def test_backend_routes_both_memory_kinds_through_the_same_get_and_put(registered, monkeypatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

    staging, sdk, host, device = registered
    backend = MooncakeBackend.__new__(MooncakeBackend)
    backend._host_staging = staging
    monkeypatch.setattr(backend, "_build_replicate_config", Mock(return_value=object()))
    sdk.batch_put_from_multi_buffers.return_value = [0]
    sdk.batch_get_into_multi_buffers.return_value = [8]
    for memory_location, buffer in ((MemoryLocation.HOST, host), (MemoryLocation.DEVICE, device)):
        address = ctypes.addressof(buffer)
        assert backend.put(["key"], [[address]], [[8]], memory_location=memory_location) == [True]
        put_address = sdk.batch_put_from_multi_buffers.call_args.args[1][0][0]
        assert (put_address == address) is (memory_location is MemoryLocation.DEVICE)
        assert backend.get(["key"], [[address]], [[8]], memory_location=memory_location) == [0]
        get_address = sdk.batch_get_into_multi_buffers.call_args.args[1][0][0]
        assert (get_address == address) is (memory_location is MemoryLocation.DEVICE)


def test_explicit_device_batch_keeps_the_device_only_backend_path(monkeypatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

    backend = MooncakeBackend.__new__(MooncakeBackend)
    backend._host_staging = None
    backend._lazy_init = False
    backend._store_initialized = True
    backend.store = Mock()
    monkeypatch.setattr(backend, "_build_replicate_config", Mock(return_value=object()))
    backend.store.batch_put_from_multi_buffers.return_value = [0]
    backend.store.batch_get_into_multi_buffers.return_value = [8]
    assert backend.put(["key"], [[100]], [[8]], memory_location=MemoryLocation.DEVICE) == [True]
    assert backend.get(["key"], [[100]], [[8]], memory_location=MemoryLocation.DEVICE) == [0]
    for operation in (backend.put, backend.get):
        with pytest.raises(RuntimeError, match="Host buffers have not been registered"):
            operation(["key"], [[100]], [[8]], memory_location=MemoryLocation.HOST)


def test_host_get_completes_while_host_put_waits_in_sdk(registered):
    bulk, sdk, host, _ = registered
    entered, release = threading.Event(), threading.Event()
    host_ptr = ctypes.addressof(host)

    def put(keys, addrs, sizes, config):
        assert ctypes.string_at(addrs[0][0], 8) == b"........"
        entered.set()
        assert release.wait(5)
        assert ctypes.string_at(addrs[0][0], 8) == b"........"
        return [0]

    def get(keys, addrs, sizes):
        ctypes.memset(addrs[0][0], ord("k"), 8)
        return [8]

    sdk.batch_put_from_multi_buffers.side_effect = put
    sdk.batch_get_into_multi_buffers.side_effect = get
    with ThreadPoolExecutor(2) as executor:
        sending = executor.submit(bulk.put, ["send"], [[host_ptr]], [[8]], None, memory_location=MemoryLocation.HOST)
        try:
            assert entered.wait(5)
            receiving = executor.submit(
                bulk.get, ["recv"], [[host_ptr + 8]], [[8]], memory_location=MemoryLocation.HOST
            )
            assert receiving.result(timeout=3) == [0]
            assert host.raw[:16] == b"........kkkkkkkk"
        finally:
            release.set()
        assert sending.result(timeout=3) == [True]
    assert bulk._send.base != bulk._recv.base


@pytest.mark.parametrize("operation", ["get", "put"])
@pytest.mark.parametrize("parts", [[16], [8, 8], [17], [8, 9]])
def test_host_key_capacity_is_checked_before_sdk(registered, operation, parts):
    bulk, sdk, host, _ = registered
    ptr = ctypes.addressof(host)
    addrs = [ptr + sum(parts[:i]) for i in range(len(parts))]
    sdk.batch_get_into_multi_buffers.return_value = [sum(parts)]
    sdk.batch_put_from_multi_buffers.return_value = [0]

    def transfer():
        if operation == "get":
            return bulk.get(["boundary"], [addrs], [parts], memory_location=MemoryLocation.HOST)
        return bulk.put(["boundary"], [addrs], [parts], None, memory_location=MemoryLocation.HOST)

    if sum(parts) <= 16:
        assert transfer() == ([0] if operation == "get" else [True])
    else:
        with pytest.raises(ValueError, match="staging capacity"):
            transfer()
        sdk.batch_get_into_multi_buffers.assert_not_called()
        sdk.batch_put_from_multi_buffers.assert_not_called()


def test_invalid_later_host_key_does_not_issue_earlier_batch(registered):
    bulk, sdk, host, _ = registered
    ptr = ctypes.addressof(host)
    with pytest.raises(ValueError, match="staging capacity"):
        bulk.get(["first", "bad"], [[ptr], [ptr]], [[8], [17]], memory_location=MemoryLocation.HOST)
    sdk.batch_get_into_multi_buffers.assert_not_called()


def test_staging_pages_are_faulted_and_zeroed_before_registration(monkeypatch):
    monkeypatch.setattr(mooncake_staging, "_HOST_STAGING_BYTES", 16)
    sdk = Mock()
    sdk.unregister_buffer.return_value = 0
    bulk = mooncake_staging.MooncakeHostStaging(sdk)

    def register(ptr, size):
        assert ctypes.string_at(ptr, size) == bytes(size)
        return 0

    sdk.register_buffer.side_effect = register
    try:
        bulk.register([BufferDescriptor(address=100, size_bytes=32, memory_location=MemoryLocation.HOST)])
    finally:
        bulk.unregister()
        bulk.close()
    sdk.close.assert_not_called()


def test_backend_closes_store_before_releasing_failed_deregistration(monkeypatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

    monkeypatch.setattr(mooncake_staging, "_HOST_STAGING_BYTES", 16)
    backend = MooncakeBackend.__new__(MooncakeBackend)
    sdk = Mock()
    sdk.register_buffer.return_value = 0
    sdk.unregister_buffer.return_value = -1
    bulk = mooncake_staging.MooncakeHostStaging(sdk)
    bulk.register([BufferDescriptor(address=100, size_bytes=32, memory_location=MemoryLocation.HOST)])
    backend._host_staging, backend.store = bulk, sdk
    backend._store_initialized = True

    def close_store():
        assert not bulk._send.buffer.closed and not bulk._recv.buffer.closed

    sdk.close.side_effect = close_store
    with pytest.raises(RuntimeError, match="could not unregister"):
        backend.close()
    sdk.close.assert_called_once_with()
    assert bulk._send.buffer.closed and bulk._recv.buffer.closed
    assert backend.store is None and backend._host_staging is None
    backend.close()
    sdk.close.assert_called_once_with()


def test_failed_store_shutdown_keeps_mappings_and_retries_only_outstanding_registration(monkeypatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

    monkeypatch.setattr(mooncake_staging, "_HOST_STAGING_BYTES", 16)
    backend = MooncakeBackend.__new__(MooncakeBackend)
    sdk = Mock()
    sdk.register_buffer.return_value = 0
    sdk.unregister_buffer.side_effect = [-1, 0, 0]
    sdk.close.side_effect = [RuntimeError("Store close failed"), None]
    staging = mooncake_staging.MooncakeHostStaging(sdk)
    staging.register([BufferDescriptor(address=100, size_bytes=32, memory_location=MemoryLocation.HOST)])
    backend._host_staging, backend.store = staging, sdk

    with pytest.raises(RuntimeError, match="Store close failed"):
        backend.close()
    assert not staging._send.buffer.closed and not staging._recv.buffer.closed
    backend.close()
    assert sdk.unregister_buffer.call_args_list == [
        call(staging._recv.base),
        call(staging._send.base),
        call(staging._recv.base),
    ]
    assert staging._send.buffer.closed and staging._recv.buffer.closed
