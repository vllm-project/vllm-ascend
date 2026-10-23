import ctypes
from concurrent.futures import Future, ThreadPoolExecutor
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


def test_get_without_sender_returns_lease_and_independent_cache_tensor():
    client, pool, store = _client()
    with patch.object(module.logger, "info") as log:
        loaded, source = _get(client, ["key"])
    log.assert_called_once_with("Loaded %d encoder output(s) from Mooncake Store", 1)
    assert torch.equal(loaded["key"], source)
    address = store.batch_get_into.call_args.args[1][0]
    assert not pool.tensor.data_ptr() <= loaded["key"].data_ptr() < pool.tensor.data_ptr() + pool.capacity
    assert address >= pool.tensor.data_ptr()
    assert pool._allocator._free == [(0, pool.capacity)]
    store.unregister_buffer.assert_not_called()


@pytest.mark.parametrize("send_first", [True, False])
def test_get_p2p_starts_before_cache_copy_finishes_and_both_hold_lease(send_first, mock_npu_control_calls):
    client, pool, store = _client(256)
    send = Future()
    copying, finish_copy = Event(), Event()
    payloads = []

    def dispatch(key, payload):
        assert key == "key"
        payloads.append(payload)
        return send

    def synchronize():
        assert payloads  # P2P was submitted before waiting for the cache copy.
        copying.set()
        assert finish_copy.wait(3)

    client.on_get_payload = dispatch
    mock_npu_control_calls.return_value.synchronize.side_effect = synchronize
    with ThreadPoolExecutor(1) as executor:
        loaded_future = executor.submit(_get, client, ["key"])
        try:
            assert copying.wait(3)
            address = store.batch_get_into.call_args.args[1][0]
            assert payloads[0].data_ptr() == address + module._MOONCAKE_TENSOR_HEADER.size
            assert pool.reserve_bytes([256], torch.device("cpu")) is None
            if send_first:
                send.set_result(None)
                assert pool.reserve_bytes([256], torch.device("cpu")) is None
            finish_copy.set()
            loaded, source = loaded_future.result(timeout=3)
            assert torch.equal(loaded["key"], source)
            if not send_first:
                assert pool.reserve_bytes([256], torch.device("cpu")) is None
                send.set_result(None)
            assert pool._allocator._free == [(0, 256)]
        finally:
            finish_copy.set()
            if not send.done():
                send.set_result(None)


def test_failed_direct_send_poisons_and_retains_get_lease():
    client, pool, _ = _client(256)
    send = Future()
    client.on_get_payload = lambda key, payload: send
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


def test_get_copy_sync_failure_retains_lease_and_target(mock_npu_control_calls):
    client, pool, _ = _client(256)
    send = Future()
    client.on_get_payload = lambda key, payload: send
    with (
        patch.object(mock_npu_control_calls.return_value, "synchronize", side_effect=RuntimeError("copy pending")),
        pytest.raises(EmbeddingStoreError, match="unconfirmed"),
    ):
        _get(client, ["key"])
    assert client._poisoned
    assert pool._allocator._free == []
    assert client._unsafe_owners
    send.set_result(None)
    assert pool._allocator._free == []


@pytest.mark.parametrize("independent", [False, True])
def test_store_bind_once_and_reject_out_of_pool_io(independent):
    client, pool, store = _client(256, use_store_independent_te=independent)
    client.bind_staging_buffer(pool.tensor)
    if independent:
        store.register_buffer.assert_called_once_with(pool.tensor.data_ptr(), 256)
    else:
        store.register_buffer.assert_not_called()
    start = pool.tensor.data_ptr()
    for addresses, sizes in [([start - 1], [24]), ([start + 250], [24]), ([start], [])]:
        with pytest.raises(module.EmbeddingStoreOperationError, match="fixed staging pool"):
            client.put_prepared_tensor("key", addresses, sizes, [])
    store.batch_put_from_multi_buffers.assert_not_called()
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
    assert not client._poisoned


def test_model_get_dispatches_while_normal_transfer_waits_for_pool(mock_npu_control_calls):

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
    worker._run_session_writes = lambda sessions, records, write: [
        write(session, items) for session, items in sessions.items()
    ]
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
    begin_normal, normal_waiting, sent = Event(), Event(), Event()
    original_wait = pool._space_available.wait

    def wait():
        normal_waiting.set()
        return original_wait()

    def normal_transfer():
        assert begin_normal.wait(3)
        staged = pool.stage_wait([torch.zeros(32, dtype=torch.uint8)])
        pool.release(staged)

    def write(*args):
        assert normal_waiting.wait(3)
        sent.set()

    def copy_sync():
        begin_normal.set()
        assert sent.wait(3)  # GET send completes even while normal staging waits.
        assert pool._allocator._free == []  # Copy still owns its read lease.

    worker._transfer.write.side_effect = write
    mock_npu_control_calls.return_value.synchronize.side_effect = copy_sync
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
        cache = {}
        try:
            worker.start_save_caches(metadata, cache)
            assert torch.equal(cache["hash"], source)
            record = worker._producer_pushes._records["transfer"]
            record.batch_future.result(timeout=3)
            normal_future.result(timeout=3)
            worker._transfer.write.assert_called_once_with(
                "peer", [store.batch_get_into.call_args.args[1][0] + 24], [1234], [32]
            )
            assert client.on_get_payload is None
            assert pool._allocator._free == [(0, 256)]
        finally:
            begin_normal.set()
            if not normal_future.done():
                pool.poison()
    backend.shutdown()


def test_get_chunk_waits_for_all_direct_sends():
    client, pool, _ = _client(512)
    sends = {key: Future() for key in ("a", "b")}
    client.on_get_payload = lambda key, payload: sends[key]
    loaded, source = _get(client, ["a", "b"])
    assert all(torch.equal(tensor, source) for tensor in loaded.values())
    sends["a"].set_result(None)
    assert pool._allocator._free == []
    sends["b"].set_result(None)
    assert pool._allocator._free == [(0, 512)]
