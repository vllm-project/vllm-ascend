import ctypes
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.mp.test_transfer_npu_ipc import _CPUMemoryAdapter
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreRecvingProcess,
    KVCacheStoreSendingProcess,
    KVCacheStoreSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import npu_ipc
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer import KVTransferProcess
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer_backend import TransferBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.worker import TransferRuntime


class MemoryBackend:
    requires_exists_before_put = True

    def __init__(self):
        self.values = {}
        self.writes = []

    def set_device(self):
        pass

    def register_buffer(self, pointers, lengths):
        self.registrations = list(zip(pointers, lengths))

    def exists(self, keys):
        return [int(key in self.values) for key in keys]

    def put(self, keys, addresses, sizes):
        for key, row, row_sizes in zip(keys, addresses, sizes):
            value = [ctypes.string_at(address, size) for address, size in zip(row, row_sizes)]
            self.values[key] = value
            self.writes.append((key, value))

    def get(self, keys, addresses, sizes):
        result = []
        for key, row, row_sizes in zip(keys, addresses, sizes):
            if key not in self.values:
                result.append(-1)
                continue
            for address, size, data in zip(row, row_sizes, self.values[key]):
                assert len(data) == size
                ctypes.memmove(address, data, size)
            result.append(0)
        return result

    def close(self):
        pass


def database():
    db = ChunkedTokenDatabase([KeyMetadata("test", 1, 2, 1, 3)], [2], None, hash_block_size=2)
    db.set_group_buffers({0: [100]}, {0: [2]}, {0: [2]})
    return db


def request(req_id="request"):
    return ReqMeta(req_id, 4, [[1, 3]], [b"a" * 32, b"b" * 32], can_save=True)


def registered_runtime(monkeypatch, config, backend, worker, caches, pointers, lengths):
    adapter = _CPUMemoryAdapter()
    export = npu_ipc.export_worker_kv_caches
    import_cache = npu_ipc.import_worker_kv_caches
    monkeypatch.setattr(npu_ipc, "export_worker_kv_caches", lambda values: export(values, adapter))
    monkeypatch.setattr(npu_ipc, "import_worker_kv_caches", lambda spec: import_cache(spec, adapter))
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer_backend.create_transfer_backend",
        return_value=backend,
    ):
        runtime = TransferRuntime(config)
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.TransferChannel"):
        parent = KVTransferProcess(config)
    parent.register_kv_caches(worker, caches, pointers, lengths)
    payload = parent.channel.call.call_args.args[1]
    runtime.execute("register", msgspec.msgpack.decode(msgspec.msgpack.encode(payload)))
    return runtime, parent


def run_transfer(runtime, parent, operation, req):
    parent.submit_request(operation, req)
    payload = parent.channel.submit.call_args.args[1]
    return runtime.submit(operation, msgspec.msgpack.decode(msgspec.msgpack.encode(payload))).result(2)


def test_child_handlers_match_thread_keys_and_roundtrip_buffer_contents(monkeypatch):
    caches = {"layer.0": torch.arange(8, dtype=torch.uint8).view(4, 2)}
    tensor = caches["layer.0"]
    db = database()
    db.set_group_buffers({0: [tensor.data_ptr()]}, {0: [2]}, {0: [2]})
    worker = SimpleNamespace(
        token_database=db,
        group_kv_caches_base_addr=db.group_kv_caches_base_addr,
        group_block_len={0: [2]},
        group_block_stride={0: [2]},
        group_kv_cache_families={0: "default"},
        group_num_layers={0: 1},
        group_layer_cache_entry_offsets={0: [0]},
        group_uses_align_state=[False],
    )
    config = dict(
        device_index=None,
        tp_rank=1,
        tp_size=2,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        enable_kv_events=True,
        lazy_init=False,
    )
    backend = MemoryBackend()
    config = {"backend": "mooncake", **config}
    runtime, parent = registered_runtime(
        monkeypatch, config, backend, worker, caches, [tensor.data_ptr()], [tensor.nbytes]
    )
    try:
        original_backend = MemoryBackend()
        original = KVCacheStoreSendingThread(original_backend, db, [2], 1, 2, 1, enable_kv_event=True)
        req = request()
        req.token_ids = [1, 2, 3, 4]
        req.original_block_size = 2
        original.add_stored_request(req.req_id)
        original.request_queue.put(req)
        original.request_queue.get_nowait()
        original._handle_request(req)

        result = run_transfer(runtime, parent, "store", req)
        assert result["finished"]
        assert backend.writes == original_backend.writes
        assert len(backend.writes) == 2
        assert "@pcp:2@dcp:1@head_or_tp_rank:1@pp_rank:3@" in backend.writes[0][0]
        assert result["events"] == original.get_kv_events()

        req.block_ids_by_group = [[0, 2]]
        req.load_spec = LoadSpec(0, 4, True, token_len=4)
        result = run_transfer(runtime, parent, "load", req)
        assert result["finished"] and result["invalid_blocks"] == []
        assert tensor[0].tolist() == tensor[1].tolist()
        assert tensor[2].tolist() == tensor[3].tolist()

        backend.values.clear()
        result = run_transfer(runtime, parent, "load", req)
        assert sorted(result["invalid_blocks"]) == [0, 2]
        with pytest.raises(ValueError, match="exceeds"):
            runtime.execute("get_ranges", (["key"], [[(0, 7, 2)]]))
    finally:
        runtime.close()
        parent.close()


def test_child_tp_mismatch_handler_reuses_worker_business_logic(monkeypatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

    tensor = torch.arange(16, dtype=torch.uint8).view(4, 2, 2)
    db = ChunkedTokenDatabase([KeyMetadata("test", 0, 0, 0, 0)], [2], None, hash_block_size=2)
    db.set_group_buffers({0: [tensor.data_ptr()]}, {0: [4]}, {0: [4]})
    worker = SimpleNamespace(
        token_database=db,
        group_kv_caches_base_addr=db.group_kv_caches_base_addr,
        group_block_len={0: [4]},
        group_block_stride={0: [4]},
        group_kv_cache_families={0: "default"},
        group_num_layers={0: 1},
        group_layer_cache_entry_offsets={0: [0]},
        group_uses_align_state=[False],
        tp_mismatch=True,
        block_size=2,
        num_sub_keys=2,
        sub_size_bytes=1,
    )
    config = dict(
        backend="mooncake",
        device_index=None,
        tp_rank=0,
        tp_size=2,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        enable_kv_events=False,
        lazy_init=False,
    )
    backend = MemoryBackend()
    runtime, parent = registered_runtime(
        monkeypatch, config, backend, worker, {"layer.0": tensor}, [tensor.data_ptr()], [tensor.nbytes]
    )
    try:
        original_backend = MemoryBackend()
        original_worker = KVPoolWorker.__new__(KVPoolWorker)
        original_worker.tp_mismatch = True
        original_worker.m_store = original_backend
        original_worker.token_database = db
        original_worker.group_kv_caches_base_addr = db.group_kv_caches_base_addr
        original_worker.group_block_len = worker.group_block_len
        original_worker.group_block_stride = worker.group_block_stride
        original_worker.block_size = worker.block_size
        original_worker.num_sub_keys = worker.num_sub_keys
        original_worker.sub_size_bytes = worker.sub_size_bytes
        original_worker.tp_rank = config["tp_rank"]
        original_worker.enable_kv_events = False
        original = KVCacheStoreSendingThread(original_backend, db, [2], 0, 2, 1, worker=original_worker)
        original_worker.kv_send_thread = original
        req = request()
        original.add_stored_request(req.req_id)
        original.request_queue.put(req)
        original.request_queue.get_nowait()
        original._handle_request(req)

        result = run_transfer(runtime, parent, "store", req)
        assert result["finished"]
        assert backend.writes == original_backend.writes

        req.block_ids_by_group = [[0, 2]]
        req.load_spec = LoadSpec(0, 4, True, token_len=4)
        result = run_transfer(runtime, parent, "load", req)
        assert result["finished"] and result["invalid_blocks"] == []
        assert tensor[0].tolist() == tensor[1].tolist()
        assert tensor[2].tolist() == tensor[3].tolist()
    finally:
        runtime.close()
        parent.close()


def test_child_handlers_preserve_hybrid_group_keys_and_buffers(monkeypatch):
    kv = torch.arange(8, dtype=torch.uint8).view(4, 2)
    state = torch.arange(16, 24, dtype=torch.uint8).view(4, 2)
    db = ChunkedTokenDatabase(
        [KeyMetadata("test", 0, 0, 0, 0, 0), KeyMetadata("test", 0, 0, 0, 0, 1)],
        [2, 2],
        None,
        hash_block_size=2,
    )
    group_addresses = {0: [kv.data_ptr()], 1: [state.data_ptr()]}
    group_lengths = {0: [2], 1: [2]}
    group_strides = {0: [2], 1: [2]}
    families = {0: "default", 1: "state"}
    db.set_group_buffers(
        group_addresses,
        group_lengths,
        group_strides,
        group_cache_families=families,
        group_num_layers={0: 1, 1: 1},
    )
    worker = SimpleNamespace(
        token_database=db,
        group_kv_caches_base_addr=group_addresses,
        group_block_len=group_lengths,
        group_block_stride=group_strides,
        group_kv_cache_families=families,
        group_num_layers={0: 1, 1: 1},
        group_layer_cache_entry_offsets={0: [0], 1: [0]},
        group_uses_align_state=[False, True],
        tp_mismatch=False,
    )
    config = dict(
        backend="mooncake",
        device_index=None,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        enable_kv_events=False,
        lazy_init=False,
    )
    backend = MemoryBackend()
    runtime, parent = registered_runtime(
        monkeypatch,
        config,
        backend,
        worker,
        {"layer.0.kv": kv, "layer.0.state": state},
        [kv.data_ptr(), state.data_ptr()],
        [kv.nbytes, state.nbytes],
    )
    try:
        original_backend = MemoryBackend()
        original = KVCacheStoreSendingThread(original_backend, db, [2, 2], 0, group_uses_align_state=[False, True])
        req = ReqMeta(
            "hybrid",
            4,
            [[1, 3], [0, 2]],
            [b"a" * 32, b"b" * 32],
            can_save=True,
            kv_cache_group_ids=[0, 1],
            skip_null_blocks_by_group=[False, True],
        )
        original.add_stored_request(req.req_id)
        original.request_queue.put(req)
        original.request_queue.get_nowait()
        original._handle_request(req)

        result = run_transfer(runtime, parent, "store", req)
        assert result["finished"]
        assert backend.writes == original_backend.writes
        assert any("@group:0@" in key for key, _ in backend.writes)
        assert any("@group:1@" in key for key, _ in backend.writes)

        req.block_ids_by_group = [[0, 2], [1, 3]]
        req.load_spec = LoadSpec(0, 4, True, token_len=4)
        result = run_transfer(runtime, parent, "load", req)
        assert result["finished"] and result["invalid_blocks"] == []
        assert kv[0].tolist() == kv[1].tolist() and kv[2].tolist() == kv[3].tolist()
        assert state[1].tolist() == [18, 19] and state[3].tolist() == state[2].tolist()
    finally:
        runtime.close()
        parent.close()


def process_endpoint(cls):
    process = MagicMock()
    process.channel.timeout = 1
    process.submit_request.side_effect = lambda *args: Future()
    endpoint = cls(MagicMock(), database(), [2], 0, process=process)
    return process, endpoint


def completed(future, *, invalid_blocks=()):
    future.set_result({"finished": True, "events": [], "invalid_blocks": list(invalid_blocks)})


def test_sending_completion_counts_all_submissions_and_ignores_preempted_generation():
    process, sender = process_endpoint(KVCacheStoreSendingProcess)
    futures: list[Future] = [Future(), Future(), Future()]
    process.submit_request.side_effect = futures
    req = request()
    for _ in range(2):
        sender.add_stored_request(req.req_id)
        sender.add_request(req)
    completed(futures[0])
    assert sender.get_and_clear_finished_requests() == set()
    sender.delete_finished_stored_request(req.req_id)
    sender.discard_finished_requests({req.req_id})
    sender.add_stored_request(req.req_id)
    sender.add_request(req)
    completed(futures[1])
    assert sender.get_stored_request_count(req.req_id) == 1
    completed(futures[2])
    sender.wait_for_pending()
    assert sender.get_and_clear_finished_requests() == {req.req_id}
    assert not sender._generations


def test_preempted_load_does_not_publish_stale_failure_or_completion():
    process, receiver = process_endpoint(KVCacheStoreRecvingProcess)
    future: Future = Future()
    process.submit_request.side_effect = [future]
    receiver.add_request(request())
    receiver.discard_finished_requests({"request"})
    completed(future, invalid_blocks=[1])
    assert receiver.get_and_clear_finished_requests() == set()
    assert not receiver._invalid_block_ids


def test_failed_async_transfer_is_raised_by_waiter():
    process, sender = process_endpoint(KVCacheStoreSendingProcess)
    future: Future = Future()
    process.submit_request.side_effect = [future]
    sender.add_request(request())
    future.set_exception(RuntimeError("child died"))
    with pytest.raises(RuntimeError, match="asynchronous transfer"):
        sender.wait_for_pending()


def test_registration_rollback_retains_only_failed_unregistrations():
    backend = MagicMock()
    backend.store.register_buffer.side_effect = [0, 0, -1]
    backend.store.unregister_buffer.side_effect = [-2, 0, 0]
    adapter = TransferBackend("memcache", backend, 0)
    with pytest.raises(RuntimeError, match="unregistration failed"):
        adapter.register_buffer([100, 200, 300], [10, 10, 10])
    assert adapter._registered == [(200, 10)]
    adapter.close()
    assert not adapter._registered
    assert backend.store.unregister_buffer.call_args_list[-1].args == (200, 10)


def test_backend_exists_before_put_capability_is_forwarded():
    backend = MagicMock(requires_exists_before_put=False)
    assert TransferBackend("yuanrong", backend, 0).requires_exists_before_put is False


def test_lazy_memcache_registration_keeps_backend_initialization_semantics():
    backend = MagicMock()
    backend._lazy_init = True
    backend._store_initialized = False
    backend.store = None
    adapter = TransferBackend("memcache", backend, 0)

    adapter.register_buffer([100], [10])

    backend.register_buffer.assert_called_once_with([100], [10])
    assert adapter._registered == [(100, 10)]

    backend._store_initialized = True
    backend.store = MagicMock()
    backend.store.unregister_buffer.return_value = 0
    adapter.close()
    backend.store.unregister_buffer.assert_called_once_with(100, 10)


def test_yuanrong_registration_uses_existing_backend_path():
    backend = MagicMock()
    adapter = TransferBackend("yuanrong", backend, 0)

    adapter.register_buffer([100], [10])

    backend.register_buffer.assert_called_once_with([100], [10])
    adapter.close()


@pytest.mark.parametrize(
    "use_hybrid,use_compress,tp_mismatch", [(True, False, False), (False, True, False), (False, False, True)]
)
def test_worker_selects_process_without_changing_ordinary_transfer_modes(use_hybrid, use_compress, tp_mismatch):
    case = unittest.TestCase()
    try:
        worker = make_worker(case)
        # The existing fixture's hf_config is a MagicMock; provide the actual
        # ordinary-model capability before exercising backend selection.
        worker.use_hybrid = use_hybrid
        worker.use_compress = use_compress
        worker.tp_mismatch = tp_mismatch
        worker.use_multiprocess = True
        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer_backend.requires_model_worker_backend",
                return_value=False,
            ),
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.KVTransferProcess") as factory,
        ):
            worker._init_backend(None, worker._extra_config)
            assert worker.m_store is factory.return_value
            assert factory.call_args.args[0]["tp_rank"] == worker.tp_rank
            assert factory.call_args.args[0]["kv_role"] == "kv_producer"
            assert factory.call_args.args[0]["lazy_init"] is use_compress
    finally:
        case.doCleanups()


def test_backend_requiring_distributed_state_keeps_existing_thread_path():
    case = unittest.TestCase()
    try:
        worker = make_worker(case)
        worker.use_compress = False
        worker.use_multiprocess = True
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer_backend.requires_model_worker_backend",
            return_value=True,
        ):
            worker._init_backend(None, worker._extra_config)
        assert worker.transfer_process is None
    finally:
        case.doCleanups()


def test_layerwise_mode_keeps_existing_thread_path():
    case = unittest.TestCase()
    try:
        worker = make_worker(case, use_layerwise=True, extra_config={"use_multiprocess": True})
        assert worker.transfer_process is None
    finally:
        case.doCleanups()


def test_shutdown_accepts_scheduler_role_without_a_worker():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector

    scheduler = AscendStoreConnector.__new__(AscendStoreConnector)
    scheduler.shutdown()
    worker = AscendStoreConnector.__new__(AscendStoreConnector)
    worker.connector_worker = MagicMock()
    worker.shutdown()
    worker.connector_worker.close.assert_called_once()


@pytest.mark.parametrize("failed", [False, True])
def test_source_event_is_retained_until_completion_or_child_reaping(failed):
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.TransferChannel"):
        parent = KVTransferProcess({})
    parent.cache = MagicMock()
    parent._device_uuid = "device-0"
    req = request()
    req.current_event = MagicMock()
    req.current_event.ipc_handle.return_value = b"source-event"
    future: Future = Future()
    parent.channel.submit.return_value = future
    parent.submit_request("store", req)
    assert parent._events[future] is req.current_event
    payload = parent.channel.submit.call_args.args[1]
    assert payload["current_event"].handle == b"source-event"
    if failed:
        future.set_exception(RuntimeError("child stopped responding"))
        assert parent._events[future] is req.current_event
    else:
        future.set_result({})
        assert not parent._events
    parent.channel.process.poll.return_value = 0
    parent.close()
    assert not parent._events
