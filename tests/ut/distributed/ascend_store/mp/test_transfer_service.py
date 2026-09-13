import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.mp.test_transfer_npu_ipc import _CPUMemoryAdapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerBlockRange,
    LayerTransferTask,
    LoadSpec,
    ReqMeta,
    SharedBlockData,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import npu_ipc
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.service import TransferService
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer import KVTransferProcess


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


class GVAMemoryBackend(MemoryBackend):
    def __init__(self):
        super().__init__()
        self.store = self
        self.copies = []
        self.finishes = []

    def batch_copy(self, gvas, addresses, sizes, direction):
        for gva, address, size in zip(gvas, addresses, sizes):
            source, target = (address, gva) if direction == 0 else (gva, address)
            ctypes.memmove(target, source, size)
        self.copies.append((gvas, addresses, sizes, direction))
        return 0

    def batch_write_finish(self, keys, results):
        self.finishes.append((keys, results))
        return [0] * len(keys)

    def batch_remove_lease(self, keys):
        return 0


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
    monkeypatch.setattr(
        npu_ipc,
        "import_worker_kv_caches",
        lambda spec, *, device_index=None: import_cache(spec, adapter, device_index=device_index),
    )
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer_backend.create_transfer_backend",
        return_value=backend,
    ):
        runtime = TransferService(config)
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.TransferProcess"):
        parent = KVTransferProcess(config)
    parent.register_kv_caches(worker, caches, pointers, lengths)
    payload = parent.client.call.call_args.args[1]
    runtime.execute("register", msgspec.msgpack.decode(msgspec.msgpack.encode(payload)))
    return runtime, parent


def run_transfer(runtime, parent, operation, req):
    parent.submit_request(operation, req)
    payload = parent.client.submit.call_args.args[1]
    return runtime.submit(operation, msgspec.msgpack.decode(msgspec.msgpack.encode(payload))).result(2)


def layerwise_worker(db, *, use_gva, use_key_major_ranges=False, num_layers=1):
    group_ids = sorted(db.group_kv_caches_base_addr)
    save_events = [MagicMock() for _ in range(num_layers)]
    return SimpleNamespace(
        token_database=db,
        group_kv_caches_base_addr=db.group_kv_caches_base_addr,
        group_block_len=db.group_block_len,
        group_block_stride=db.group_block_stride,
        group_kv_cache_families={group_id: "default" for group_id in group_ids},
        group_num_layers={group_id: num_layers for group_id in group_ids},
        group_layer_cache_entry_offsets={
            group_id: [0, len(db.group_kv_caches_base_addr[group_id])] for group_id in group_ids
        },
        group_uses_align_state=[False] * len(group_ids),
        use_layerwise=True,
        use_layerwise_transfer=use_gva,
        use_block_key_layerwise=use_key_major_ranges,
        block_size=db.block_size[0],
        num_layers=num_layers,
        sync_save_events=save_events,
        page_size_bytes=sum(db.group_block_len[0]),
        consumer_is_to_put=False,
        h2d_stagger_us=0,
        layerwise_max_transfer_blocks=0,
        layerwise_max_transfer_bytes=0,
        tp_mismatch=False,
    )


def run_layer_transfer(runtime, parent, operation, tasks):
    parent.submit_layer_request(operation, tasks, tasks[0].layer_id)
    payload = parent.client.submit.call_args.args[1]
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
        global_rank=0,
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
        assert [(operation, keys) for operation, _duration, keys in result["operations"]] == [("load_get", 2)]
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


def test_child_tp_mismatch_handler_roundtrips_strided_slices(monkeypatch):
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
        num_sub_keys=2,
    )
    config = dict(
        backend="mooncake",
        device_index=None,
        global_rank=0,
        tp_rank=0,
        tp_size=2,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        enable_kv_events=True,
        lazy_init=False,
    )
    backend = MemoryBackend()
    runtime, parent = registered_runtime(
        monkeypatch, config, backend, worker, {"layer.0": tensor}, [tensor.data_ptr()], [tensor.nbytes]
    )
    try:
        register_payload = parent.client.call.call_args.args[1]
        assert register_payload["tp_mismatch"] == {"num_sub_keys": 2}

        req = request()
        req.token_ids = [1, 2, 3, 4]
        req.original_block_size = 2
        result = run_transfer(runtime, parent, "store", req)
        assert result["finished"]
        assert len(backend.writes) == 4
        assert {key.split("@head_or_tp_rank:")[1].split("@", 1)[0] for key, _ in backend.writes} == {"0", "1"}
        assert len(result["events"]) == 2

        result = run_transfer(runtime, parent, "store", req)
        assert result["finished"] and result["events"] == []
        assert len(backend.writes) == 4

        req.block_ids_by_group = [[0, 2]]
        req.load_spec = LoadSpec(0, 4, True, token_len=4)
        result = run_transfer(runtime, parent, "load", req)
        assert result["finished"] and result["invalid_blocks"] == []
        assert [(operation, keys) for operation, _duration, keys in result["operations"]] == [("load_get", 4)]
        assert tensor[0].tolist() == tensor[1].tolist()
        assert tensor[2].tolist() == tensor[3].tolist()

        backend.values.clear()
        result = run_transfer(runtime, parent, "load", req)
        assert sorted(result["invalid_blocks"]) == [0, 2]
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
        global_rank=0,
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


def test_child_key_layer_handlers_roundtrip_buffer_contents(monkeypatch):
    tensor = torch.arange(8, dtype=torch.uint8).view(4, 2)
    db = database()
    db.set_group_buffers(
        {0: [tensor.data_ptr()]},
        {0: [2]},
        {0: [2]},
        group_num_layers={0: 1},
        group_layer_cache_entry_offsets={0: [0, 1]},
    )
    worker = layerwise_worker(db, use_gva=False)
    config = dict(
        backend="mooncake",
        device_index=None,
        global_rank=0,
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
        monkeypatch, config, backend, worker, {"layer.0": tensor}, [tensor.data_ptr()], [tensor.nbytes]
    )
    req = request()
    req.is_last_chunk = True
    task = LayerTransferTask(0, [LayerBlockRange(req, 0, 2)])
    expected = torch.zeros_like(tensor)
    expected[[1, 3]] = tensor[[1, 3]]
    try:
        result = run_layer_transfer(runtime, parent, "store", [task])
        assert result["finished_req_ids"] == [req.req_id]
        assert len(backend.writes) == 2
        tensor.zero_()
        result = run_layer_transfer(runtime, parent, "load", [task])
        assert result["finished_req_ids"] == [req.req_id]
        assert tensor.tolist() == expected.tolist()
    finally:
        runtime.close()
        parent.close()


def test_child_gva_layer_handlers_roundtrip_multiple_cache_groups(monkeypatch):
    caches = {
        "layer.0.kv": torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8),
        "layer.0.state": torch.tensor([[5, 6], [7, 8]], dtype=torch.uint8),
    }
    remotes = [torch.zeros_like(cache) for cache in caches.values()]
    db = ChunkedTokenDatabase(
        [KeyMetadata("test", 0, 0, 0, 0, 0), KeyMetadata("test", 0, 0, 0, 0, 1)],
        [2, 2],
        None,
        hash_block_size=2,
    )
    addresses = {group_id: [cache.data_ptr()] for group_id, cache in enumerate(caches.values())}
    db.set_group_buffers(
        addresses,
        {0: [2], 1: [2]},
        {0: [2], 1: [2]},
        group_num_layers={0: 1, 1: 1},
        group_layer_cache_entry_offsets={0: [0, 1], 1: [0, 1]},
    )
    worker = layerwise_worker(db, use_gva=True)
    config = dict(
        backend="memcache",
        device_index=None,
        global_rank=0,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        enable_kv_events=False,
        lazy_init=False,
    )
    backend = GVAMemoryBackend()
    pointers = [cache.data_ptr() for cache in caches.values()]
    lengths = [cache.nbytes for cache in caches.values()]
    runtime, parent = registered_runtime(monkeypatch, config, backend, worker, caches, pointers, lengths)
    tasks = []
    for group_id, remote in enumerate(remotes):
        shared = SharedBlockData(
            block_ids_arr=torch.tensor([0, 1]).numpy(),
            block_gvas_arr=torch.tensor([remote.data_ptr(), remote.data_ptr() + 2]).numpy(),
            req_ids=[f"request-{group_id}"],
            is_last_chunks=[True],
            save_keys=[f"key-{group_id}"],
            load_keys=[f"key-{group_id}"],
        )
        tasks.append(
            LayerTransferTask(
                0,
                [],
                shared_block_data=shared,
                group_id=group_id,
                write_finish_keys=["key-0", "key-1"] if group_id == 1 else [],
            )
        )
    try:
        result = run_layer_transfer(runtime, parent, "store", tasks)
        assert set(result["finished_req_ids"]) == {"request-0", "request-1"}
        assert [remote.tolist() for remote in remotes] == [cache.tolist() for cache in caches.values()]
        assert backend.finishes == [(["key-0", "key-1"], [0, 0])]
        for cache in caches.values():
            cache.zero_()
        result = run_layer_transfer(runtime, parent, "load", tasks)
        assert set(result["finished_req_ids"]) == {"request-0", "request-1"}
        assert [cache.tolist() for cache in caches.values()] == [remote.tolist() for remote in remotes]
        assert [len(copy[0]) for copy in backend.copies] == [4, 4]
        assert [copy[-1] for copy in backend.copies] == [0, 1]
    finally:
        runtime.close()
        parent.close()


def two_layer_key_runtime(monkeypatch):
    tensor = torch.arange(8, dtype=torch.uint8).view(4, 2)
    db = database()
    db.set_group_buffers(
        {0: [tensor.data_ptr()]},
        {0: [2]},
        {0: [2]},
        group_num_layers={0: 2},
        group_layer_cache_entry_offsets={0: [0, 1]},
    )
    worker = layerwise_worker(db, use_gva=False, num_layers=2)
    config = dict(
        backend="mooncake",
        device_index=None,
        global_rank=0,
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
        monkeypatch, config, backend, worker, {"layer.0": tensor}, [tensor.data_ptr()], [tensor.nbytes]
    )
    return runtime, parent, backend


def test_hot_layer_stores_keep_parent_events_out_of_the_child(monkeypatch):
    runtime, parent, backend = two_layer_key_runtime(monkeypatch)
    req = request()
    req.is_last_chunk = True
    task = LayerTransferTask(1, [LayerBlockRange(req, 0, 2)])
    try:
        for round_id in range(2):
            if round_id > 0:
                # A hot request stores its new suffix blocks, which the pool
                # has not seen yet.
                backend.values.clear()
            result = run_layer_transfer(runtime, parent, "store", [task])
            assert result["finished_req_ids"] == [req.req_id]
            payload = parent.client.submit.call_args.args[1]
            assert "current_event" not in payload

        # Layerwise ordering stays in the model Worker. The child neither
        # imports a parent event nor retains one in its synchronous handlers.
        assert runtime.sender.sync_save_events == [None, None]
    finally:
        runtime.close()
        parent.close()


def test_mooncake_range_tasks_roundtrip_through_child_handlers(monkeypatch):
    tensor = torch.arange(8, dtype=torch.uint8).view(4, 2)
    db = database()
    db.set_group_buffers(
        {0: [tensor.data_ptr()]},
        {0: [2]},
        {0: [2]},
        group_num_layers={0: 1},
        group_layer_cache_entry_offsets={0: [0, 1]},
    )
    worker = layerwise_worker(db, use_gva=False, use_key_major_ranges=True)
    config = dict(
        backend="mooncake",
        device_index=None,
        global_rank=0,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        put_step=1,
        kv_role="kv_producer",
        enable_kv_events=False,
        lazy_init=False,
    )
    backend = MagicMock()
    backend.batch_copy_put.return_value = [2]
    backend.batch_commit.return_value = [0]
    backend.batch_copy_get.return_value = [2]
    runtime, parent = registered_runtime(
        monkeypatch,
        config,
        backend,
        worker,
        {"layer.0": tensor},
        [tensor.data_ptr()],
        [tensor.nbytes],
    )
    shared = SharedBlockData(
        block_ids_arr=torch.tensor([1]).numpy(),
        block_gvas_arr=None,
        block_keys=["key"],
        req_ids=["request"],
        is_last_chunks=[True],
    )
    task = LayerTransferTask(0, [], shared_block_data=shared, use_key_major_ranges=True)
    try:
        store_result = run_layer_transfer(runtime, parent, "store", [task])
        assert store_result["committed_keys"] == ["key"]
        assert store_result["revoked_keys"] == []
        backend.batch_copy_put.assert_called_once()
        backend.batch_commit.assert_called_once_with(["key"])

        load_result = run_layer_transfer(runtime, parent, "load", [task])
        assert load_result["invalid_blocks"] == []
        assert load_result["load_aborted"] is False
        backend.batch_copy_get.assert_called_once()

        backend.batch_copy_get.return_value = [-1]
        failed_load = run_layer_transfer(runtime, parent, "load", [task])
        assert failed_load["invalid_blocks"] == [1]
    finally:
        runtime.close()
        parent.close()
