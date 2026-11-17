# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, call

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_bulk
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import BatchResultShapeError


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(mooncake_bulk, "_HOST_STAGING_BYTES", 16)
    host = ctypes.create_string_buffer(b"...............................")
    device = ctypes.create_string_buffer(8)
    host_ptr = ctypes.addressof(host)
    device_ptr = ctypes.addressof(device)
    sdk = Mock()
    sdk.register_buffer.return_value = 0
    sdk.unregister_buffer.return_value = 0
    bulk = mooncake_bulk.MooncakeMixedBulk(sdk, {host_ptr: ctypes.sizeof(host)})
    try:
        bulk.register([host_ptr, device_ptr], [ctypes.sizeof(host), ctypes.sizeof(device)])
        assert sdk.register_buffer.call_args_list == [
            call(bulk._send.base, 16),
            call(bulk._recv.base, 16),
            call(device_ptr, ctypes.sizeof(device)),
        ]
        yield bulk, sdk, host, device
    finally:
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
    assert bulk.put(["a", "b"], [[host_ptr, host_ptr + 8], [host_ptr + 18]], [[4, 6], [10]], None, host=True)
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
        host=True,
    ) == [-1, 0]
    assert bulk.get(["index"], [[device_ptr]], [[8]], host=False) == [0]
    assert host.raw[:16] == b"........kkkkkkkk"
    assert device.raw == b"kkkkkkkk"


def test_get_rejects_malformed_sdk_results(registered):
    bulk, sdk, host, _ = registered
    sdk.batch_get_into_multi_buffers.return_value = []
    with pytest.raises(BatchResultShapeError, match="0 results for 1"):
        bulk.get(["bad"], [[ctypes.addressof(host)]], [[8]], host=True)
    assert host.raw[:8] == b"........"


@pytest.mark.parametrize(("offset", "size", "message"), [(28, 8, "outside"), (0, 18, "capacity")])
def test_invalid_host_geometry_never_reaches_sdk(registered, offset, size, message):
    bulk, sdk, host, _ = registered
    with pytest.raises(ValueError, match=message):
        bulk.get(["bad"], [[ctypes.addressof(host) + offset]], [[size]], host=True)
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
        MooncakeBackend.validate_mixed_bulk_config()
    else:
        with pytest.raises(ValueError, match="MC_TRANSFER_TIMEOUT"):
            MooncakeBackend.validate_mixed_bulk_config()


def test_mooncake_partial_registration_rolls_back_device_and_staging(monkeypatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

    monkeypatch.setattr(mooncake_bulk, "_HOST_STAGING_BYTES", 16)
    backend = MooncakeBackend.__new__(MooncakeBackend)
    sdk = Mock()
    sdk.register_buffer.side_effect = [0, 0, 0, -600]
    sdk.unregister_buffer.return_value = 0
    backend.store = sdk
    backend._store_initialized = backend._use_store_independent_te = True
    backend._mixed_bulk = None

    with pytest.raises(RuntimeError, match="registration failed.*-600"):
        backend.register_bulk_buffers([100, 200, 300], [32, 64, 128], host_ranges={100: 32})

    send_ptr = sdk.register_buffer.call_args_list[0].args[0]
    recv_ptr = sdk.register_buffer.call_args_list[1].args[0]
    assert sdk.register_buffer.call_args_list == [call(send_ptr, 16), call(recv_ptr, 16), call(200, 64), call(300, 128)]
    assert sdk.unregister_buffer.call_args_list == [call(200), call(recv_ptr), call(send_ptr)]
    sdk.close.assert_called_once_with()
    assert backend._mixed_bulk is None and backend.store is None
    assert not backend._store_initialized


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
        sending = executor.submit(bulk.put, ["send"], [[host_ptr]], [[8]], None, host=True)
        try:
            assert entered.wait(5)
            receiving = executor.submit(bulk.get, ["recv"], [[host_ptr + 8]], [[8]], host=True)
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
            return bulk.get(["boundary"], [addrs], [parts], host=True)
        return bulk.put(["boundary"], [addrs], [parts], None, host=True)

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
        bulk.get(["first", "bad"], [[ptr], [ptr]], [[8], [17]], host=True)
    sdk.batch_get_into_multi_buffers.assert_not_called()


def test_staging_pages_are_faulted_and_zeroed_before_registration(monkeypatch):
    monkeypatch.setattr(mooncake_bulk, "_HOST_STAGING_BYTES", 16)
    sdk = Mock()
    sdk.unregister_buffer.return_value = 0
    bulk = mooncake_bulk.MooncakeMixedBulk(sdk, {100: 32})

    def register(ptr, size):
        assert ctypes.string_at(ptr, size) == bytes(size)
        return 0

    sdk.register_buffer.side_effect = register
    try:
        bulk.register([100], [32])
    finally:
        bulk.close()
    sdk.close.assert_not_called()


def test_backend_closes_store_before_releasing_failed_deregistration(monkeypatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

    monkeypatch.setattr(mooncake_bulk, "_HOST_STAGING_BYTES", 16)
    backend = MooncakeBackend.__new__(MooncakeBackend)
    sdk = Mock()
    sdk.register_buffer.return_value = 0
    sdk.unregister_buffer.return_value = -1
    bulk = mooncake_bulk.MooncakeMixedBulk(sdk, {100: 32})
    bulk.register([100], [32])
    backend._mixed_bulk, backend.store = bulk, sdk
    backend._store_initialized = True

    def close_store():
        assert not bulk._send.buffer.closed and not bulk._recv.buffer.closed

    sdk.close.side_effect = close_store
    with pytest.raises(RuntimeError, match="could not unregister"):
        backend.close()
    sdk.close.assert_called_once_with()
    assert bulk._send.buffer.closed and bulk._recv.buffer.closed
    assert backend.store is None and backend._mixed_bulk is None
    backend.close()
    sdk.close.assert_called_once_with()
