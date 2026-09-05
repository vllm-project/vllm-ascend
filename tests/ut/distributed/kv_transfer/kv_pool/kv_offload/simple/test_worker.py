# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple import worker as worker_module
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.worker import (
    SimpleCPUOffloadNPUWorker,
    _flatten_kv_value,
)


def make_empty_worker():
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker._backend = MagicMock()
    worker._connector_metadata = None
    worker._store_compute_done = None
    worker._pending_load_event_indices = set()
    worker._pending_store_event_indices = set()
    worker._completed_store_events = {}
    worker._load_events = []
    worker._store_events = []
    worker._load_hwm = worker._store_hwm = -1
    return worker


def test_initialization_installs_npu_dma_backend(monkeypatch):
    base_init = MagicMock(return_value=None)
    monkeypatch.setattr(worker_module.SimpleCPUOffloadWorker, "__init__", base_init)
    config, caches = object(), object()
    worker = SimpleCPUOffloadNPUWorker(config, caches, 64)
    base_init.assert_called_once_with(config, caches, 64)
    assert isinstance(worker._backend, worker_module.NPUDmaCopyBackend)
    worker._backend.shutdown()


def test_empty_registration_does_not_initialize_dma():
    worker = make_empty_worker()
    worker.register_kv_caches({})
    worker._backend.init.assert_not_called()


def test_aliasing_layers_share_one_cpu_mirror(monkeypatch):
    monkeypatch.setattr(worker_module, "is_pin_memory_available", lambda: False)
    worker = make_empty_worker()
    worker.cpu_capacity_bytes = 0
    worker.kv_cache_config = SimpleNamespace(num_blocks=3)
    tensor = torch.zeros(3, 8)
    worker.register_kv_caches({"a": tensor, "b": tensor})
    assert list(worker.gpu_kv_caches) == ["a"]
    assert worker.num_cpu_blocks == 1
    assert worker.cpu_kv_caches["a"].shape == (1, 32)
    worker._backend.init.assert_called_once()


@pytest.mark.parametrize("shape", [(), (2,), (1, 2, 4)])
def test_invalid_block_dimensions_fail_explicitly(shape):
    with pytest.raises(RuntimeError, match="cannot locate blocks dim"):
        SimpleCPUOffloadNPUWorker._build_block_views("layer", torch.zeros(shape), 4)


@pytest.mark.parametrize("metadata_present", [False, True])
def test_completed_events_release_only_finished_jobs(metadata_present):
    worker = make_empty_worker()
    worker._pending_load_event_indices = {1, 2, 3}
    worker._pending_store_event_indices = {4, 5}
    done = MagicMock(query=MagicMock(return_value=True))
    pending = MagicMock(query=MagicMock(return_value=False))
    worker._load_events = [(1, done), (2, done), (3, pending)]
    worker._store_events = [(4, done), (5, pending)]
    worker._connector_metadata = SimpleCPUOffloadMetadata(load_event_to_reqs={1: {"r1"}}) if metadata_present else None
    assert worker.get_finished(set()) == (None, {"r1"} if metadata_present else None)
    assert worker._pending_load_event_indices == {3}
    assert worker._pending_store_event_indices == {5}
    assert worker._completed_store_events == {4: 1}
    worker._backend.launch_copy.assert_not_called()


def test_store_barrier_is_reused_across_steps():
    worker = make_empty_worker()
    worker._connector_metadata = SimpleCPUOffloadMetadata(store_gpu_blocks=[1], store_cpu_blocks=[2], store_event=7)
    worker.get_finished(set())
    barrier = worker._store_compute_done
    worker.get_finished(set())
    assert worker._store_compute_done is barrier
    assert torch.npu.Event.call_count == 1
    assert barrier.record.call_count == 2
    assert worker._backend.launch_copy.call_count == 2


def test_registration_requests_pinned_host_memory_when_available(monkeypatch):
    worker = make_empty_worker()
    worker.kv_cache_config = SimpleNamespace(num_blocks=2)
    worker.cpu_capacity_bytes = 16
    zeros = torch.zeros
    allocator = MagicMock(side_effect=lambda *args, **kwargs: zeros(*args, **{**kwargs, "pin_memory": False}))
    monkeypatch.setattr(torch, "zeros", allocator)
    monkeypatch.setattr(worker_module, "is_pin_memory_available", lambda: True)
    worker.register_kv_caches({"a": torch.arange(8, dtype=torch.int8).reshape(2, 4)})
    assert worker.cpu_kv_caches["a"].shape == (4, 4)
    assert allocator.call_args.kwargs["pin_memory"] is True
    worker._backend.init.assert_called_once()


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
