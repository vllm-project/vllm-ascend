import ctypes
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import EmbeddingStoreError

from tests.ut.distributed.ec_transfer.test_mooncake_memory import _make_pool
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import store_client as module
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_client import AscendMooncakeEmbeddingStoreClient


@pytest.fixture(autouse=True)
def mock_npu_control_calls():
    # These are CPU tensor tests, including when collected on an NPU host.
    with (
        patch.object(module.torch.npu, "set_device"),
        patch.object(module.torch.npu, "current_stream") as current_stream,
    ):
        yield current_stream


def _client(capacity=1024, **kwargs):
    pool = _make_pool(capacity)
    store = MagicMock()
    store.register_buffer.return_value = 0
    client = AscendMooncakeEmbeddingStoreClient(store, pool, **kwargs)
    client.bind_staging_buffer(pool.tensor)
    return client, pool, store


def _get(client, objects):
    spec = SimpleNamespace(shape=(2, 4), dtype="torch.float32", nbytes=32)
    source = torch.arange(8, dtype=torch.float32).view(2, 4)
    metadata = module._encode_mooncake_tensor_metadata(source)
    blob = metadata + source.numpy().tobytes()

    def receive(keys, addresses, sizes):
        for address in addresses:
            ctypes.memmove(address, blob, len(blob))
        return [len(blob)] * len(keys)

    client.store.batch_get_into.side_effect = receive
    empty = torch.empty
    with patch.object(module.torch, "empty", side_effect=lambda *a, **kw: empty(*a, **{**kw, "device": "cpu"})):
        loaded = client._load_chunk(dict.fromkeys(objects, spec), SimpleNamespace(type="npu"))
    return loaded, source


def test_failed_direct_send_poisons_and_retains_get_lease():
    client, pool, _ = _client(256)
    send: Future[None] = Future()
    client.on_get_payloads = lambda payloads: [send]
    loaded, _ = _get(client, ["key"])
    send.set_exception(RuntimeError("native write status unknown"))
    assert client._poisoned
    assert pool._allocator._free == []
    assert client._unsafe_owners
    assert torch.equal(loaded["key"], torch.arange(8, dtype=torch.float32).view(2, 4))


def test_get_when_pool_busy_does_not_block_or_submit_io():
    client, pool, store = _client(256)
    held = pool.reserve_bytes([256], torch.device("cpu"))
    loaded, _ = _get(client, ["key"])
    assert loaded == {}
    store.batch_get_into.assert_not_called()
    pool.release(held)


@pytest.mark.parametrize("result", [[-704], [0]])
def test_safe_get_miss_releases_entire_lease(result):
    client, pool, store = _client(256)
    store.batch_get_into.return_value = result
    with patch.object(module.logger, "info") as log:
        loaded = client._load_chunk({"key": SimpleNamespace(nbytes=32)}, SimpleNamespace(type="npu"))
    log.assert_not_called()
    assert loaded == {}
    assert pool._allocator._free == [(0, 256)]
    assert not client._poisoned


def test_unknown_get_completion_keeps_lease_and_poisons_pool():
    client, pool, store = _client(256)
    store.batch_get_into.return_value = [-1]
    with pytest.raises(EmbeddingStoreError, match="unconfirmed"):
        client._load_chunk({"key": SimpleNamespace(nbytes=32)}, SimpleNamespace(type="npu"))
    assert client._poisoned
    assert pool._allocator._free == []
    with pytest.raises(RuntimeError, match="poisoned"):
        pool.reserve_bytes([1], torch.device("cpu"))


def test_corrupt_get_header_releases_without_poison():
    client, pool, store = _client(256)
    with (
        patch.object(
            module, "_validate_mooncake_tensor_metadata", side_effect=module.EmbeddingStoreOperationError("bad header")
        ),
        patch.object(module.logger, "info") as log,
    ):
        loaded, _ = _get(client, ["key"])
    log.assert_not_called()
    assert loaded == {}
    assert not client._poisoned
    assert pool._allocator._free == [(0, 256)]


@pytest.mark.parametrize("result", [0, -705, -1])
def test_prepared_put_only_uses_fixed_pool_and_retains_unknown_completion(result):
    client, pool, store = _client(256)
    lease = pool.reserve_bytes([64], torch.device("cpu"))
    address = lease.tensors[0].data_ptr()
    store.batch_put_from_multi_buffers.return_value = [result]
    if result == 0:
        client.put_prepared_tensor("key", [address, address + 24], [24, 32], [lease])
    else:
        with pytest.raises(EmbeddingStoreError):
            client.put_prepared_tensor("key", [address, address + 24], [24, 32], [lease])
    assert client._poisoned is (result == -1)
    if result != -1:
        pool.release(lease)
        assert pool._allocator._free == [(0, 256)]
    store.unregister_buffer.assert_not_called()


def test_prepared_put_batches_objects_in_one_native_call():
    client, pool, store = _client(512)
    lease = pool.reserve_bytes([64, 64], torch.device("cpu"))
    first, second = lease.tensors
    items = [
        ("first", [first.data_ptr()], [first.nbytes], [lease]),
        ("second", [second.data_ptr()], [second.nbytes], [lease]),
    ]
    store.batch_put_from_multi_buffers.return_value = [0, -705]

    assert client.put_prepared_tensors(items) == [0, -705]

    store.batch_put_from_multi_buffers.assert_called_once_with(
        ["first", "second"],
        [[first.data_ptr()], [second.data_ptr()]],
        [[first.nbytes], [second.nbytes]],
        client.replicate_config,
    )
    assert not client._poisoned
    pool.release(lease)


def test_close_keeps_bound_store_and_pool_registered():
    client, pool, store = _client(use_store_independent_te=True)
    retained = len(module._PROCESS_LIFETIME_STORE_OWNERS)
    try:
        client.close()
        client.close()
        store.close.assert_not_called()
        store.unregister_buffer.assert_not_called()
        assert len(module._PROCESS_LIFETIME_STORE_OWNERS) == retained + 1
    finally:
        del module._PROCESS_LIFETIME_STORE_OWNERS[retained:]


@pytest.mark.parametrize("independent", [False, True])
def test_store_binds_once(independent):
    client, pool, store = _client(256, use_store_independent_te=independent)
    client.bind_staging_buffer(pool.tensor)
    if independent:
        store.register_buffer.assert_called_once_with(pool.tensor.data_ptr(), 256)
    else:
        store.register_buffer.assert_not_called()
    store.unregister_buffer.assert_not_called()


@pytest.mark.parametrize("read_bytes", [56, 1024])
def test_inherited_load_groups_reads_and_ascend_bounds_them_by_pool(read_bytes):
    client, pool, store = _client(256, read_buffer_bytes=read_bytes)
    spec = SimpleNamespace(shape=(2, 4), dtype="torch.float32", nbytes=32)
    source = torch.arange(8, dtype=torch.float32).view(2, 4)
    blob = module._encode_mooncake_tensor_metadata(source) + source.numpy().tobytes()

    def receive(keys, addresses, sizes):
        for address in addresses:
            ctypes.memmove(address, blob, len(blob))
        return [len(blob)] * len(keys)

    store.batch_get_into.side_effect = receive
    empty = torch.empty
    load_chunk = client._load_chunk
    with (
        patch.object(
            client,
            "_load_chunk",
            side_effect=lambda expected, device: load_chunk(expected, SimpleNamespace(type="npu")),
        ),
        patch.object(module.torch, "empty", side_effect=lambda *a, **kw: empty(*a, **{**kw, "device": "cpu"})),
    ):
        loaded = client.load_tensors({"first": spec, "second": spec}, "cpu")
    assert list(loaded) == ["first", "second"]
    assert all(torch.equal(tensor, source) for tensor in loaded.values())
    assert store.batch_get_into.call_count == 2
    assert all(
        not pool.tensor.data_ptr() <= tensor.data_ptr() < pool.tensor.data_ptr() + pool.capacity
        for tensor in loaded.values()
    )
    assert pool._allocator._free == [(0, pool.capacity)]
    store.unregister_buffer.assert_not_called()
    assert not client._poisoned


def test_model_get_dispatches_while_normal_transfer_waits_for_pool():
    from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
        ECMooncakeConnectorMetadata,
        ECMooncakePushSpec,
    )

    from tests.ut.distributed.ec_transfer.test_mooncake_worker import _make_lifecycle_worker
    from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_backend import AscendDeferredStoreBackend
    from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.worker import AscendProducerPushManager

    client, pool, store = _client(256)
    backend = AscendDeferredStoreBackend(client, "namespace", max_pending_items=4, max_pending_bytes=1024)
    worker = _make_lifecycle_worker()
    worker._output_store = backend
    worker._producer_memory = pool
    worker._buffer_device = "cpu"
    worker._producer_pushes = AscendProducerPushManager()
    worker._bind_push_source = lambda tensor, key: worker._producer_pushes.bind_source(key, tensor, None)
    worker._reserve_remote = lambda spec: [
        {
            "ready": True,
            "cached": False,
            "cancelled": False,
            "write": True,
            "nbytes": 32,
            "dst_session": "peer",
            "dst_ptr": 1234,
        }
    ]
    worker._notify_completions = MagicMock()
    worker._run_fanout = lambda tasks, on_submit: [task() for task in tasks]
    source = torch.arange(8, dtype=torch.float32).view(2, 4)
    blob = module._encode_mooncake_tensor_metadata(source) + source.numpy().tobytes()
    store.batch_is_exist.return_value = [1]

    def receive(keys, addresses, sizes):
        for address in addresses:
            ctypes.memmove(address, blob, len(blob))
        return [len(blob)] * len(keys)

    store.batch_get_into.side_effect = receive
    spec = SimpleNamespace(shape=(2, 4), dtype="torch.float32", nbytes=32)
    metadata = ECMooncakeConnectorMetadata(
        pushes=[ECMooncakePushSpec("hash", 32, (2, 4), "float32", "consumer", "transfer")],
        store_candidates={"hash": spec},
    )
    get_started, normal_waiting, sent = Event(), Event(), Event()
    original_wait = pool._space_available.wait

    def wait():
        normal_waiting.set()
        return original_wait()

    def normal_transfer():
        assert get_started.wait(3)
        staged = pool.stage_wait([torch.zeros(32, dtype=torch.uint8)])
        pool.release(staged)

    def write(*args):
        assert normal_waiting.wait(3)
        sent.set()

    worker._transfer.write.side_effect = write
    original_receive = store.batch_get_into.side_effect

    def receive_after_lease(*args):
        get_started.set()
        return original_receive(*args)

    store.batch_get_into.side_effect = receive_after_lease
    empty = torch.empty
    # Keep the inherited backend and worker orchestration; adapt CPU test devices
    # only at the client boundary, without replacing the global torch.device type.
    with (
        ThreadPoolExecutor(1) as control,
        ThreadPoolExecutor(1) as normal,
        ThreadPoolExecutor(1) as direct,
        patch.object(pool._space_available, "wait", side_effect=wait),
        patch.object(
            client,
            "load_tensors",
            side_effect=lambda expected, device: client._load_chunk(expected, SimpleNamespace(type="npu")),
        ),
        patch.object(module.torch, "empty", side_effect=lambda *a, **kw: empty(*a, **{**kw, "device": "cpu"})),
    ):
        worker._control_executor, worker._io_executor, worker._get_executor = control, normal, direct
        normal_future = normal.submit(normal_transfer)
        cache: dict[str, torch.Tensor] = {}
        try:
            worker.start_save_caches(metadata, cache)
            assert sent.wait(3)  # GET send completes even while normal staging waits.
            assert torch.equal(cache["hash"], source)
            record = worker._producer_pushes._records["transfer"]
            record.batch_future.result(timeout=3)
            normal_future.result(timeout=3)
            worker._transfer.write.assert_called_once_with(
                "peer", [store.batch_get_into.call_args.args[1][0] + 24], [1234], [32]
            )
            assert client.on_get_payloads is None
            assert pool._allocator._free == [(0, 256)]
        finally:
            get_started.set()
            if not normal_future.done():
                pool.poison()
    backend.shutdown()


def test_get_chunk_batches_direct_send_and_header_copy():
    client, pool, store = _client(512)
    sends: list[Future[None]] = [Future(), Future()]
    dispatched = []

    def dispatch(payloads):
        dispatched.append(payloads)
        return sends

    client.on_get_payloads = dispatch
    stack = torch.stack
    with patch.object(module.torch, "stack", wraps=stack) as gather_headers:
        loaded, source = _get(client, ["a", "b"])
    assert list(dispatched[0]) == ["a", "b"]
    assert all(torch.equal(tensor, source) for tensor in loaded.values())
    addresses = store.batch_get_into.call_args.args[1]
    assert [tensor.data_ptr() for tensor in dispatched[0].values()] == [
        address + module._MOONCAKE_TENSOR_HEADER.size for address in addresses
    ]
    assert pool.reserve_bytes([512], torch.device("cpu")) is None
    gather_headers.assert_called_once()
    assert len(gather_headers.call_args.args[0]) == 2
    sends[0].set_result(None)
    assert pool._allocator._free == []
    sends[1].set_result(None)
    assert pool._allocator._free == [(0, 512)]


@pytest.mark.parametrize("event_failure", [False, True])
def test_npu_cache_copy_uses_event_or_synchronizes_only_copy_stream(mock_npu_control_calls, event_failure):
    client, _, _ = _client(256)
    copy_stream = MagicMock()
    copy_event = MagicMock()
    if event_failure:
        copy_event.record.side_effect = RuntimeError("event unavailable")
    target = MagicMock()
    target.device.type = "npu"
    payload = MagicMock()
    compute_stream = mock_npu_control_calls.return_value
    with (
        patch.object(module.torch.npu, "Stream", return_value=copy_stream),
        patch.object(module.torch.npu, "stream", return_value=nullcontext()),
        patch.object(module.torch.npu, "Event", return_value=copy_event),
    ):
        result = client._copy_loaded_payloads(
            {"key": target},
            {"key": payload},
            torch.device("cpu"),
        )
    target.record_stream.assert_called_once_with(copy_stream)
    target.copy_.assert_called_once_with(payload, non_blocking=True)
    copy_event.record.assert_called_once_with(copy_stream)
    compute_stream.synchronize.assert_not_called()
    if event_failure:
        assert result is None
        copy_stream.synchronize.assert_called_once_with()
    else:
        assert result is copy_event
        compute_stream.wait_event.assert_called_once_with(copy_event)
        copy_stream.synchronize.assert_not_called()


def test_get_event_and_direct_send_both_hold_lease():
    client, pool, _ = _client(256)
    lease = pool.reserve_bytes([64], torch.device("cpu"))
    event_started, finish_event = Event(), Event()
    send: Future[None] = Future()
    copy_event = MagicMock()

    def synchronize():
        event_started.set()
        assert finish_event.wait(3)

    copy_event.synchronize.side_effect = synchronize
    client._defer_get_release(lease, copy_event, [send], [], torch.device("cpu"))
    try:
        assert event_started.wait(3)
        assert pool.reserve_bytes([256], torch.device("cpu")) is None
        finish_event.set()
        assert pool.reserve_bytes([256], torch.device("cpu")) is None
        send.set_result(None)
        client._get_completion_executor.submit(lambda: None).result(timeout=3)
        assert pool._allocator._free == [(0, 256)]
    finally:
        finish_event.set()
        if not send.done():
            send.set_result(None)
        client._get_completion_executor.shutdown(wait=True)


def test_get_event_failure_poisons_and_retains_lease():
    client, pool, _ = _client(256)
    lease = pool.reserve_bytes([64], torch.device("cpu"))
    copy_event = MagicMock()
    copy_event.synchronize.side_effect = RuntimeError("copy completion unknown")
    client._defer_get_release(lease, copy_event, [], [], torch.device("cpu"))
    client._get_completion_executor.submit(lambda: None).result(timeout=3)
    client._get_completion_executor.shutdown(wait=True)
    assert client._poisoned
    assert pool._allocator._free == []
