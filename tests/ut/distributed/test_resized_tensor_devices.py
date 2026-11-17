# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import ctypes
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, get_ident
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.ec_transfer import memcache_resized_tensor_store as store_module
from vllm_ascend.distributed.ec_transfer.memcache_resized_tensor_store import (
    MemcacheResizedTensorStore,
    ResizedTensorLoadRequest,
)
from vllm_ascend.distributed.ec_transfer.similarity_cache_config import MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MmcDirect


def _store():
    backend = MagicMock()
    backend.exists.return_value = [0]
    backend.put_from.return_value = 0
    return MemcacheResizedTensorStore(backend, 2, "test-model", MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE)


def test_receive_buffer_is_cpu_and_is_reused():
    store = _store()
    try:
        buffer = store._ensure_recv_buffer()
        assert buffer.device.type == "cpu"
        assert store._ensure_recv_buffer() is buffer
        store._backend.register_buffer.assert_called_once_with([buffer.data_ptr()], [buffer.nbytes])
    finally:
        store.shutdown()


def test_cpu_receive_loads_real_bytes_with_host_direction():
    store = _store()
    current = torch.zeros(16, 12)
    expected = torch.arange(current.numel(), dtype=current.dtype).reshape_as(current)
    recv = torch.empty(current.nbytes, dtype=torch.uint8)
    outputs = [None]

    def copy_into(keys, addresses, sizes, direction):
        assert direction == MmcDirect.COPY_G2H.value
        ctypes.memmove(addresses[0], expected.data_ptr(), sizes[0])
        return [0]

    store._backend.batch_get_into_buffers.side_effect = copy_into
    try:
        with patch.object(store, "get_transfer_direction", side_effect=AssertionError("helper called")):
            store._load_batch_into_buffer(
                [ResizedTensorLoadRequest(0, "candidate", current.nbytes)], outputs, recv, current
            )
        assert torch.equal(outputs[0], expected)
        recv.zero_()
        assert torch.equal(outputs[0], expected), "outputs must survive receive-buffer reuse"
    finally:
        store.shutdown()


def test_put_uses_host_tensor_and_fixed_host_direction():
    store = _store()
    host = torch.ones(16, 12)
    tensor = MagicMock()
    tensor.dtype = host.dtype
    tensor.shape = host.shape
    tensor.detach.return_value.to.return_value = host
    try:
        with patch.object(store, "get_transfer_direction", side_effect=AssertionError("helper called")):
            assert store.put("image", tensor, (1, 4, 4))
        tensor.detach.return_value.to.assert_called_once_with(device="cpu")
        assert store._backend.put_from.call_args.args[1:] == (host.data_ptr(), host.nbytes, MmcDirect.COPY_H2G.value)
        store._backend.unregister_buffer.assert_called_once_with([host.data_ptr()], [host.nbytes])
    finally:
        store.shutdown()


@pytest.mark.parametrize(
    "device,to_store,direction",
    [
        ("cpu", True, MmcDirect.COPY_H2G.value),
        ("cpu", False, MmcDirect.COPY_G2H.value),
        ("npu", True, MmcDirect.COPY_L2G.value),
        ("npu", False, MmcDirect.COPY_G2L.value),
    ],
)
def test_retained_get_transfer_direction(device, to_store, direction):
    assert MemcacheResizedTensorStore.get_transfer_direction(
        SimpleNamespace(type=device), to_store=to_store
    ) == direction


def test_get_transfer_direction_rejects_unsupported_devices():
    with pytest.raises(ValueError, match="Unsupported EC resized tensor device"):
        MemcacheResizedTensorStore.get_transfer_direction(torch.device("meta"), to_store=True)


def test_parallel_put_serializes_registered_pointer_lifetimes():
    store = _store()
    tensor = torch.ones(16, 12)
    first_in_copy = Event()
    release_first = Event()
    second_entered_put = Event()
    second_registered = Event()
    state_lock = Lock()
    active = set()
    overlap = []

    def register(ptrs, sizes):
        with state_lock:
            overlap.append(bool(active))
            active.add(ptrs[0])
            if len(overlap) == 2:
                second_registered.set()

    def unregister(ptrs, sizes):
        with state_lock:
            active.discard(ptrs[0])

    def copy(key, *args):
        if key.startswith("resized_first_"):
            first_in_copy.set()
            assert release_first.wait(5)
        return 0

    def second_put():
        second_entered_put.set()
        return store.put("second", tensor, (1, 4, 4))

    store._backend.register_buffer.side_effect = register
    store._backend.unregister_buffer.side_effect = unregister
    store._backend.put_from.side_effect = copy
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(store.put, "first", tensor, (1, 4, 4))
            assert first_in_copy.wait(5)
            second = executor.submit(second_put)
            assert second_entered_put.wait(5)
            # Give an unprotected second registration time to overlap the
            # first transfer; with serialization it must remain blocked.
            overlapped = second_registered.wait(0.2)
            release_first.set()
            assert first.result(timeout=5)
            assert second.result(timeout=5)
        assert not overlapped
        assert overlap == [False, False]
        assert not active
    finally:
        release_first.set()
        store.shutdown()


def test_put_failure_unregisters_buffer():
    store = _store()
    store._backend.put_from.side_effect = RuntimeError("transfer failed")
    tensor = torch.ones(16, 12)
    try:
        with pytest.raises(RuntimeError, match="transfer failed"):
            store.put("image", tensor, (1, 4, 4))
        store._backend.unregister_buffer.assert_called_once_with([tensor.data_ptr()], [tensor.nbytes])
    finally:
        store.shutdown()


def test_oversized_receive_uses_cpu_buffer_and_unregisters_it():
    store = _store()
    current = torch.zeros(2048)
    requests = [[ResizedTensorLoadRequest(0, "candidate", current.nbytes)]]
    try:
        with (
            patch.object(store_module.torch, "empty", wraps=torch.empty) as allocate,
            patch.object(store, "_load_batch_into_buffer"),
            patch.object(store, "calculate_ssim_results", return_value=[]),
        ):
            assert store.select_ssim_from_oversized_batches(requests, current, 0.99) == []
        assert allocate.call_args.kwargs["device"] == "cpu"
        assert store._backend.register_buffer.call_args == store._backend.unregister_buffer.call_args
        store._backend.unregister_buffer.assert_called_once()
    finally:
        store.shutdown()


def test_failed_clone_unregisters_temporary_cpu_buffer():
    store = _store()
    current = torch.zeros(16, 12)
    recv = MagicMock()
    recv.device = SimpleNamespace(type="cpu")
    recv.data_ptr.return_value = 4096
    recv.nbytes = current.nbytes
    clone = recv.__getitem__.return_value.view.return_value.reshape.return_value.clone
    clone.side_effect = RuntimeError("clone failed")
    operations = []
    store._backend.batch_get_into_buffers.side_effect = lambda *args: operations.append("get") or [0]
    store._backend.unregister_buffer.side_effect = lambda *args: operations.append("unregister")
    try:
        with (
            patch.object(store_module.torch, "empty", return_value=recv),
            pytest.raises(RuntimeError, match="clone failed"),
        ):
            store.select_ssim_from_oversized_batches(
                [[ResizedTensorLoadRequest(0, "candidate", current.nbytes)]], current, 0.99
            )
        assert operations == ["get", "unregister"]
    finally:
        store.shutdown()


def test_device_ssim_scoring_stays_on_caller_thread():
    store = _store()
    # Meta tensors check dispatch without running accelerator arithmetic.
    current = torch.ones(16, 12, device="meta")
    count = store_module._PARALLEL_SIMILARITY_COMPUTE_THRESHOLD + 1
    requests = [
        ResizedTensorLoadRequest(
            i,
            store.get_resized_key(f"candidate-{i}", (1, 4, 4), current.shape, current.dtype),
            current.nbytes,
        )
        for i in range(count)
    ]
    caller = get_ident()
    worker_threads = []
    try:
        with patch.object(store, "_score_candidate", side_effect=lambda *args: worker_threads.append(get_ident())):
            assert store.calculate_ssim_results(current, [current] * count, requests) == []
        assert worker_threads == [caller] * count
    finally:
        store.shutdown()
