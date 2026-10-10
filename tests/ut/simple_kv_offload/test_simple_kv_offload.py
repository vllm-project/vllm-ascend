# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple import (
    simple_cpu_offload_connector as connector_module,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple import worker as worker_module
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.simple_cpu_offload_connector import (
    AscendSimpleCPUOffloadConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.worker import (
    SimpleCPUOffloadNPUWorker,
    _flatten_kv_value,
)


def test_factory_registration_uses_consolidated_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.distributed.kv_transfer import register_connector

    registrations: dict[str, tuple[str, str]] = {}

    def capture_registration(cls, name: str, module_path: str, class_name: str) -> None:
        registrations[name] = (module_path, class_name)

    # Keep the test independent of whether the vLLM plugin was already loaded
    # by the current pytest environment.
    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    monkeypatch.setattr(
        KVConnectorFactory,
        "register_connector",
        classmethod(capture_registration),
    )
    register_connector()

    assert registrations["SimpleCPUOffloadConnector"] == (
        "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.simple_cpu_offload_connector",
        "AscendSimpleCPUOffloadConnector",
    )


@pytest.mark.parametrize(
    ("role", "has_upstream_worker", "expect_npu_worker"),
    [
        (KVConnectorRole.WORKER, True, True),
        (KVConnectorRole.WORKER, False, False),
        (KVConnectorRole.SCHEDULER, False, False),
    ],
)
def test_connector_only_replaces_enabled_worker(
    monkeypatch: pytest.MonkeyPatch,
    role: KVConnectorRole,
    has_upstream_worker: bool,
    expect_npu_worker: bool,
) -> None:
    upstream_worker = SimpleNamespace(cpu_capacity_bytes=512) if has_upstream_worker else None

    def fake_upstream_init(self, vllm_config, connector_role, kv_cache_config):
        self.worker_handler = upstream_worker

    created: list[tuple[object, object, int]] = []
    npu_worker = object()

    def fake_npu_worker(vllm_config, kv_cache_config, cpu_capacity):
        created.append((vllm_config, kv_cache_config, cpu_capacity))
        return npu_worker

    monkeypatch.setattr(SimpleCPUOffloadConnector, "__init__", fake_upstream_init)
    monkeypatch.setattr(
        connector_module,
        "SimpleCPUOffloadNPUWorker",
        fake_npu_worker,
    )

    config = object()
    kv_cache_config = object()
    connector = AscendSimpleCPUOffloadConnector(config, role, kv_cache_config)

    if expect_npu_worker:
        assert connector.worker_handler is npu_worker
        assert created == [(config, kv_cache_config, 512)]
    else:
        assert connector.worker_handler is upstream_worker
        assert not created


def test_flatten_kv_value_preserves_separate_kv_tensors() -> None:
    key_cache = torch.empty(2, 4)
    value_cache = torch.empty(2, 4)

    flattened = _flatten_kv_value(key_cache)
    assert len(flattened) == 1
    assert flattened[0] is key_cache

    flattened = _flatten_kv_value((key_cache, value_cache))
    assert len(flattened) == 2
    assert flattened[0] is key_cache
    assert flattened[1] is value_cache


def test_build_block_views_uses_tensor_offset_not_whole_storage() -> None:
    # Simulate the aligned allocation used by NPUModelRunner: the visible
    # cache starts inside a larger storage containing leading/trailing padding.
    allocation = torch.arange(64, dtype=torch.uint8)
    cache = allocation[7:31].view(4, 6)

    views = SimpleCPUOffloadNPUWorker._build_block_views("layer", cache, num_blocks=4)

    assert list(views) == ["layer"]
    assert views["layer"].shape == (4, 6)
    assert views["layer"].data_ptr() == cache.data_ptr()
    assert torch.equal(views["layer"], cache)


def test_build_block_views_splits_outer_kv_segments() -> None:
    cache = torch.arange(48, dtype=torch.uint8).view(2, 4, 6)

    views = SimpleCPUOffloadNPUWorker._build_block_views("layer", cache, num_blocks=4)

    assert list(views) == ["layer.0", "layer.1"]
    assert views["layer.0"].shape == (4, 6)
    assert views["layer.1"].shape == (4, 6)
    assert torch.equal(views["layer.0"], cache[0])
    assert torch.equal(views["layer.1"], cache[1])


def test_register_kv_caches_keeps_separate_kv_and_initializes_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.init_args: tuple[object, ...] | None = None

        def init(self, *args) -> None:
            self.init_args = args

    load_stream = object()
    store_stream = object()
    streams = iter((load_stream, store_stream))
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Stream=lambda: next(streams)),
        raising=False,
    )
    monkeypatch.setattr(worker_module, "is_pin_memory_available", lambda: False)

    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker.kv_cache_config = SimpleNamespace(num_blocks=4)
    worker.cpu_capacity_bytes = 96
    worker._backend = FakeBackend()

    key_cache = torch.empty(4, 6, dtype=torch.uint8)
    value_cache = torch.empty(4, 6, dtype=torch.uint8)
    worker.register_kv_caches({"layer": (key_cache, value_cache)})

    assert worker.num_cpu_blocks == 8
    assert list(worker.gpu_kv_caches) == ["layer", "layer.1"]
    assert worker.cpu_kv_caches["layer"].shape == (8, 6)
    assert worker.cpu_kv_caches["layer.1"].shape == (8, 6)
    assert worker.load_stream is load_stream
    assert worker.store_stream is store_stream
    assert worker._backend.init_args == (
        worker.gpu_kv_caches,
        worker.cpu_kv_caches,
        key_cache.device,
        load_stream,
        store_stream,
    )


def test_get_finished_records_store_barrier_on_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEvent:
        def __init__(self) -> None:
            self.recorded_stream = None

        def record(self, stream) -> None:
            self.recorded_stream = stream

    class RecordingBackend:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def launch_copy(self, *args, **kwargs) -> None:
            self.calls.append(kwargs)

    current_stream = object()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=FakeEvent, current_stream=lambda: current_stream),
        raising=False,
    )

    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker._backend = RecordingBackend()
    worker._connector_metadata = SimpleCPUOffloadMetadata(
        load_event=1,
        load_gpu_blocks=[2],
        load_cpu_blocks=[3],
        store_event=4,
        store_gpu_blocks=[5],
        store_cpu_blocks=[6],
    )
    worker._store_compute_done = None
    worker._load_events = []
    worker._store_events = []
    worker._pending_load_event_indices = set()
    worker._pending_store_event_indices = set()
    worker._completed_store_events = {}

    assert worker.get_finished(set()) == (None, None)

    load_call, store_call = worker._backend.calls
    assert load_call["is_store"] is False
    assert "wait_event" not in load_call
    assert store_call["is_store"] is True
    store_event = store_call["wait_event"]
    assert isinstance(store_event, FakeEvent)
    assert store_event.recorded_stream is current_stream


# ---------------------------------------------------------------------------
# Copy backend worker thread: a failing job must not kill the thread, and a
# failed load must reach the engine instead of hanging (issue #17932).
# ---------------------------------------------------------------------------


def _make_backend(monkeypatch: pytest.MonkeyPatch, failures: set[int]):
    """Backend whose copy raises for the jobs whose event_idx is in `failures`."""
    from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple import copy_backend as backend_module

    seen: list[int] = []

    def fake_copy_blocks(src_blocks, dst_blocks, params) -> None:
        event_idx = len(seen)
        seen.append(event_idx)
        if event_idx in failures:
            raise RuntimeError("swap_blocks_batch failed")

    monkeypatch.setattr(backend_module, "copy_blocks", fake_copy_blocks)
    monkeypatch.setattr(backend_module, "build_params", lambda *args, **kwargs: object())

    backend = backend_module.NPUDmaCopyBackend()
    backend.init({}, {}, torch.device("cpu"), torch.npu.Stream(), torch.npu.Stream())
    return backend


def _drain(backend, events_list, expected: int, timeout: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while len(events_list) < expected and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(events_list) == expected, f"worker thread stopped after {len(events_list)} of {expected} jobs"


def test_copy_loop_survives_a_failing_job(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend(monkeypatch, failures={0})
    events: list[tuple[int, object]] = []

    backend.launch_copy([1], [11], is_store=False, event_idx=0, events_list=events)
    backend.launch_copy([2], [12], is_store=False, event_idx=1, events_list=events)

    # Both jobs complete: the thread is still alive after the first one raised.
    _drain(backend, events, expected=2)
    assert [event_idx for event_idx, _ in events] == [0, 1]
    assert backend._thread is not None and backend._thread.is_alive()


def test_failed_load_reports_its_destination_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend(monkeypatch, failures={0})
    events: list[tuple[int, object]] = []

    backend.launch_copy([1, 2], [11, 12], is_store=False, event_idx=0, events_list=events)
    _drain(backend, events, expected=1)

    # The event is still recorded, so the high-water mark advances and the
    # request is reported as finished receiving, as upstream requires.
    assert [event_idx for event_idx, _ in events] == [0]
    assert backend.drain_load_errors() == {11, 12}
    # Draining is destructive: the engine must not see them twice.
    assert backend.drain_load_errors() == set()


def test_failed_store_is_not_reported_as_a_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend(monkeypatch, failures={0})
    events: list[tuple[int, object]] = []

    # A store failure leaves HBM intact; the blocks just stay uncached.
    backend.launch_copy([1], [11], is_store=True, event_idx=0, events_list=events)
    _drain(backend, events, expected=1)

    assert backend.drain_load_errors() == set()


def test_successful_copies_report_no_load_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend(monkeypatch, failures=set())
    events: list[tuple[int, object]] = []

    backend.launch_copy([1], [11], is_store=False, event_idx=0, events_list=events)
    backend.launch_copy([2], [12], is_store=True, event_idx=1, events_list=events)
    _drain(backend, events, expected=2)

    assert backend.drain_load_errors() == set()


def test_shutdown_reports_a_thread_that_already_died(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple import copy_backend as backend_module

    backend = _make_backend(monkeypatch, failures=set())
    thread = backend._thread
    assert thread is not None

    # Simulate the pre-fix behavior: the loop is gone without seeing the sentinel.
    backend._queue.put(None)
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    backend._loop_exited_cleanly = False

    # vLLM's logger does not propagate to the root logger, so caplog sees nothing.
    warnings: list[str] = []
    monkeypatch.setattr(
        backend_module.logger, "warning", lambda msg, *args: warnings.append(msg % args if args else msg)
    )

    backend.shutdown()
    assert any("already gone at shutdown" in message for message in warnings)


def test_clean_shutdown_logs_no_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple import copy_backend as backend_module

    backend = _make_backend(monkeypatch, failures=set())
    warnings: list[str] = []
    monkeypatch.setattr(
        backend_module.logger, "warning", lambda msg, *args: warnings.append(msg % args if args else msg)
    )

    backend.shutdown()
    assert warnings == []


def test_worker_and_connector_surface_backend_load_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker._backend = SimpleNamespace(drain_load_errors=lambda: {7, 8})
    assert worker.get_block_ids_with_load_errors() == {7, 8}

    connector = AscendSimpleCPUOffloadConnector.__new__(AscendSimpleCPUOffloadConnector)
    connector.worker_handler = worker
    assert connector.get_block_ids_with_load_errors() == {7, 8}

    # An upstream (non-NPU) worker has no such channel; report nothing.
    connector.worker_handler = SimpleNamespace()
    assert connector.get_block_ids_with_load_errors() == set()

    # Prefix caching disabled: upstream leaves the worker unset.
    connector.worker_handler = None
    assert connector.get_block_ids_with_load_errors() == set()
