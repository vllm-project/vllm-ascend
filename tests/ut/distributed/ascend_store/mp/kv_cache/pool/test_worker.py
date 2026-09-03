import queue
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.pool.worker import (
    MPKVPoolWorker,
    _MPTransferThreadMixin,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata, ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.npu_ipc import (
    KVCacheStorageSpec,
    NPUEventSpec,
    WorkerKVCacheSpec,
    export_worker_kv_caches,
    import_worker_kv_caches,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

# isort: on


class _CPUMemoryAdapter:
    def __init__(self):
        self.storages: list[torch.Tensor] = []

    def export_storage(self, storage: torch.Tensor) -> KVCacheStorageSpec:
        index = len(self.storages)
        self.storages.append(storage)
        return KVCacheStorageSpec(
            size_bytes=storage.untyped_storage().nbytes(),
            device_type="cpu",
            device_uuid="cpu",
            handle_type="test_cpu",
            handle_version=1,
            handle=index.to_bytes(4),
        )

    def import_storage(self, spec: KVCacheStorageSpec) -> tuple[torch.Tensor, int]:
        return self.storages[int.from_bytes(spec.handle)], 3


class _Store:
    def __init__(self, error: Exception | None = None):
        self.error = error

    def set_device(self) -> None:
        if self.error is not None:
            raise self.error


class _TestTransferThread(_MPTransferThreadMixin, threading.Thread):
    def __init__(self, store: _Store | None = None):
        super().__init__(daemon=True, name="test-transfer")
        self.m_store = store or _Store()
        self.ready_event = threading.Event()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.handled: list[Any] = []
        self._fatal_error: BaseException | None = None

    @staticmethod
    def _set_os_thread_name() -> None:
        return None

    def _handle_request(self, request: Any) -> None:
        self.handled.append(request)
        self.request_queue.task_done()


def _make_vllm_config(tp_size: int = 1, rank: int = 0) -> MagicMock:
    config = MagicMock()

    hf_config = MagicMock(spec=[])
    config.model_config.model = "org/llama-7b"
    config.model_config.hf_text_config = hf_config
    config.model_config.hf_config = hf_config
    config.model_config.use_mla = False
    config.model_config.max_model_len = 1024
    config.model_config.get_num_layers.return_value = 2
    config.model_config.get_total_num_kv_heads.return_value = tp_size

    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_index = 0
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.rank = rank
    config.parallel_config.world_size = tp_size
    config.parallel_config.tensor_parallel_size = tp_size
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1

    config.kv_transfer_config.kv_role = "kv_producer"
    config.kv_transfer_config.engine_id = "engine-0"
    config.kv_transfer_config.kv_connector = "AscendStoreConnector"
    config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}
    config.cache_config.block_size = 16
    config.cache_config.prefix_match_unit = None
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.speculative_config = None
    config.kv_events_config = None
    return config


def _make_worker(exists_result: list[int], tp_size: int = 1, rank: int = 0) -> MPKVPoolWorker:
    store = MagicMock()
    store.exists.return_value = exists_result
    return MPKVPoolWorker(_make_vllm_config(tp_size, rank), store=store, rank=rank)


def test_mp_worker_reuses_original_lookup_implementation() -> None:
    assert MPKVPoolWorker.lookup_scheduler is KVPoolWorker.lookup_scheduler


def test_mp_worker_reuses_original_cache_registration() -> None:
    assert MPKVPoolWorker.register_kv_caches is KVPoolWorker.register_kv_caches


def test_mp_worker_reuses_original_get_finished() -> None:
    assert MPKVPoolWorker.get_finished is KVPoolWorker.get_finished


def test_mp_worker_reuses_original_worker_metadata() -> None:
    assert MPKVPoolWorker.build_connector_worker_meta is KVPoolWorker.build_connector_worker_meta


def test_mp_worker_reuses_original_kv_events() -> None:
    assert MPKVPoolWorker.get_kv_events is KVPoolWorker.get_kv_events


def test_mp_worker_reuses_original_retrieve_methods() -> None:
    assert MPKVPoolWorker.get_block_ids_with_load_errors is KVPoolWorker.get_block_ids_with_load_errors


def test_mp_transfer_thread_drains_accepted_requests_before_stopping() -> None:
    thread = _TestTransferThread()
    thread.start()
    assert thread.ready_event.wait(timeout=1)

    thread.add_request("first")
    thread.add_request("second")
    thread.stop()

    assert thread.handled == ["first", "second"]
    assert not thread.is_alive()
    with pytest.raises(RuntimeError, match="no longer accepts requests"):
        thread.add_request("late")


def test_mp_transfer_thread_reports_device_setup_failure_without_blocking_startup() -> None:
    thread = _TestTransferThread(_Store(RuntimeError("device unavailable")))
    thread.start()

    assert thread.ready_event.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert isinstance(thread._fatal_error, RuntimeError)


def test_mp_worker_uses_registered_rank() -> None:
    worker = _make_worker([1, 1, 1, 1], tp_size=2, rank=1)

    assert worker.tp_rank == 1
    assert worker.pp_rank == 0


def test_mp_worker_initializes_parent_cpu_state() -> None:
    worker = _make_worker([1, 1])

    assert worker.device_index is None
    assert worker.kv_send_thread is None
    assert worker.kv_recv_thread is None
    assert worker.physical_layer_to_group_layers == {}


def test_mp_worker_maps_cache_once_and_releases_it_on_close() -> None:
    adapter = _CPUMemoryAdapter()
    exported = export_worker_kv_caches({"layer.0": torch.arange(8)}, adapter)
    importer = MagicMock(side_effect=lambda spec: import_worker_kv_caches(spec, adapter))
    store = MagicMock()
    worker = MPKVPoolWorker(_make_vllm_config(), store=store, cache_importer=importer)

    worker.configure_kv_caches(exported.spec)
    worker.configure_kv_caches(exported.spec)

    assert worker.kv_cache_spec == exported.spec
    assert torch.equal(worker.kv_caches["layer.0"][0], torch.arange(8))
    assert worker.device_index == 3
    importer.assert_called_once_with(exported.spec)
    store.register_buffer.assert_called_once()

    worker.close()

    assert worker.kv_cache_spec is None
    assert worker.kv_caches == {}
    assert worker.device_index is None
    store.unregister_buffer.assert_called_once_with()
    store.close.assert_not_called()


def test_mp_worker_rejects_a_different_cache_mapping() -> None:
    adapter = _CPUMemoryAdapter()
    first = export_worker_kv_caches({"layer.0": torch.zeros(8)}, adapter)
    conflicting = export_worker_kv_caches({"layer.0": torch.ones(8)}, adapter)
    worker = MPKVPoolWorker(
        _make_vllm_config(),
        store=MagicMock(),
        cache_importer=lambda spec: import_worker_kv_caches(spec, adapter),
    )

    worker.configure_kv_caches(first.spec)

    with pytest.raises(RuntimeError, match="different specification"):
        worker.configure_kv_caches(conflicting.spec)
    worker.close()


@pytest.mark.parametrize(("exists_result", "expected"), [([1, 1], 32), ([1, 0], 16), ([0, 1], 0)])
def test_mp_worker_single_tp(exists_result: list[int], expected: int) -> None:
    worker = _make_worker(exists_result)
    result = worker.lookup_scheduler(32, ["01" * 32, "02" * 32], use_layerwise=False)
    assert result == expected


@pytest.mark.parametrize(("exists_result", "expected"), [([1, 1, 1, 1], 32), ([1, 1, 1, 0], 16), ([1, 1, 0, 1], 0)])
def test_mp_worker_requires_all_tp_ranks(exists_result: list[int], expected: int) -> None:
    worker = _make_worker(exists_result, tp_size=2)
    result = worker.lookup_scheduler(32, ["01" * 32, "02" * 32], use_layerwise=False)
    assert result == expected


def test_mp_worker_returns_miss_when_store_fails() -> None:
    store = MagicMock()
    store.exists.side_effect = RuntimeError("store unavailable")
    worker = MPKVPoolWorker(_make_vllm_config(), store=store)

    result = worker.lookup_scheduler(32, ["01" * 32, "02" * 32], use_layerwise=False)
    assert result == 0


def test_mp_worker_returns_miss_before_backend_is_initialized() -> None:
    worker = MPKVPoolWorker(_make_vllm_config())

    result = worker.lookup_scheduler(32, ["01" * 32, "02" * 32], use_layerwise=False)
    assert result == 0


def test_mp_worker_initializes_own_backend_after_cache_mapping() -> None:
    adapter = _CPUMemoryAdapter()
    exported = export_worker_kv_caches({"layer.0": torch.arange(8)}, adapter)
    store = MagicMock()
    store.exists.return_value = [1, 1]
    backend_factory = MagicMock(return_value=store)
    config = _make_vllm_config()
    worker = MPKVPoolWorker(
        config,
        cache_importer=lambda spec: import_worker_kv_caches(spec, adapter),
        backend_factory=backend_factory,
    )

    worker.configure_kv_caches(exported.spec)

    assert worker.lookup_scheduler(32, ["01" * 32, "02" * 32], use_layerwise=False) == 32
    backend_factory.assert_called_once_with(config.parallel_config, 3, False)
    assert store.set_device.call_count == 2
    worker.close()


def test_mp_worker_wait_for_save_imports_source_event() -> None:
    worker = _make_worker([0, 0])
    worker.kv_cache_spec = WorkerKVCacheSpec(
        caches={"layer.0": ()},
        storages=(
            KVCacheStorageSpec(
                size_bytes=1,
                device_type="npu",
                device_uuid="host-0",
                handle_type="test",
                handle_version=1,
                handle=b"cache-handle",
            ),
        ),
    )
    send_thread = MagicMock()
    worker.kv_send_thread = send_thread
    metadata = AscendConnectorMetadata(set(), set())
    request = ReqMeta("request-0", can_save=True)
    metadata.add_request(request)
    event_spec = NPUEventSpec("host-0", b"event-handle")
    imported_event = MagicMock()

    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.pool.worker.import_npu_event",
        return_value=imported_event,
    ) as import_event:
        worker.wait_for_save(metadata, event_spec)

    import_event.assert_called_once_with(event_spec)
    assert request.current_event is imported_event
    assert request.skip_null_blocks_by_group == worker.group_uses_align_state
    send_thread.add_stored_request.assert_called_once_with("request-0")
    send_thread.add_request.assert_called_once_with(request)
    send_thread.request_queue.join.assert_called_once_with()


def test_mp_worker_layer_store_reuses_source_event() -> None:
    worker = _make_worker([0, 0])
    worker.kv_cache_spec = WorkerKVCacheSpec(
        caches={"layer.0": ()},
        storages=(
            KVCacheStorageSpec(
                size_bytes=1,
                device_type="npu",
                device_uuid="host-0",
                handle_type="test",
                handle_version=1,
                handle=b"cache-handle",
            ),
        ),
    )
    worker.sync_save_events = [MagicMock()]
    metadata = AscendConnectorMetadata(set(), set())
    event_spec = NPUEventSpec("host-0", b"event-handle")
    imported_event = MagicMock()

    with (
        patch.object(KVPoolWorker, "start_load_kv") as start_load,
        patch.object(KVPoolWorker, "save_kv_layer") as save_layer,
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.pool.worker.import_npu_event",
            return_value=imported_event,
        ) as import_event,
    ):
        worker.start_load_kv(metadata)
        worker.save_kv_layer_from_event(event_spec)

    start_load.assert_called_once_with(metadata)
    import_event.assert_called_once_with(event_spec)
    save_layer.assert_called_once_with(metadata)
    source_event = worker.sync_save_events[0]
    source_event.record()
    source_event.synchronize()
    imported_event.record.assert_not_called()
    imported_event.synchronize.assert_called_once_with()
