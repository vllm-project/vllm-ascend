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
    worker.register_kv_caches({"layer": (key_cache, value_cache), "alias": key_cache.view_as(key_cache)})

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


def test_register_kv_caches_keeps_kv_views_of_one_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model Runner V2 slices K and V out of a single allocation.

    Both views have to be registered: deduplicating by storage pointer alone
    drops every V cache, so a reloaded prefix cache would miss half of it.
    """

    class FakeBackend:
        def __init__(self) -> None:
            self.init_args: tuple[object, ...] | None = None

        def init(self, *args) -> None:
            self.init_args = args

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Stream=lambda: object()),
        raising=False,
    )
    monkeypatch.setattr(worker_module, "is_pin_memory_available", lambda: False)

    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker.kv_cache_config = SimpleNamespace(num_blocks=4)
    worker.cpu_capacity_bytes = 96
    worker._backend = FakeBackend()

    num_blocks, page_elems = 4, 6
    backing = torch.empty(2 * num_blocks * page_elems, dtype=torch.uint8)
    key_cache = backing[: num_blocks * page_elems].view(num_blocks, page_elems)
    value_cache = backing[num_blocks * page_elems :].view(num_blocks, page_elems)
    assert key_cache.untyped_storage().data_ptr() == value_cache.untyped_storage().data_ptr()

    worker.register_kv_caches({"layer": (key_cache, value_cache), "alias": key_cache.view_as(key_cache)})

    assert list(worker.gpu_kv_caches) == ["layer", "layer.1"]
    assert worker.num_cpu_blocks == 8
    assert worker.cpu_kv_caches["layer"].shape == (8, page_elems)
    assert worker.cpu_kv_caches["layer.1"].shape == (8, page_elems)


def test_offload_copies_packed_page_once_and_matches_pool_capacity(monkeypatch):
    """Latent/RoPE views and overlay aliases must not duplicate physical pages."""
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Stream=lambda: object()), raising=False)
    monkeypatch.setattr(worker_module, "is_pin_memory_available", lambda: False)
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker._backend = SimpleNamespace(init=lambda *args: None)
    worker.cpu_capacity_bytes = 256
    descriptor = SimpleNamespace(layers=["packed", "overlay"], block_stride=16, size=128, offset=0, layer_stride=0)
    worker.kv_cache_config = SimpleNamespace(
        num_blocks=4,
        kv_cache_tensors=[descriptor],
    )
    # Leading alignment padding, four 16-byte pages, and unused tuple padding.
    backing = torch.arange(160, dtype=torch.uint8)
    latent = backing.as_strided((4, 8), (16, 1), storage_offset=16)
    rope = backing.as_strided((4, 8), (16, 1), storage_offset=24)
    worker.register_kv_caches({"packed": (latent, rope), "overlay": (latent, rope)})
    assert list(worker.gpu_kv_caches) == ["packed"]
    assert torch.equal(worker.gpu_kv_caches["packed"].view(torch.uint8), backing[16:80].view(4, 16))
    # The scheduler budgets the whole 128-byte backing, including tuple padding.
    assert worker.num_cpu_blocks == 4 * 256 // 128
    assert worker.cpu_kv_caches["packed"].shape == (8, 16)


def test_descriptor_path_does_not_merge_separately_allocated_components(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Stream=lambda: object()), raising=False)
    monkeypatch.setattr(worker_module, "is_pin_memory_available", lambda: False)
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker._backend = SimpleNamespace(init=lambda *args: None)
    worker.cpu_capacity_bytes = 256
    worker.kv_cache_config = SimpleNamespace(
        num_blocks=4,
        kv_cache_tensors=[SimpleNamespace(layers=["layer"], size=64, offset=0, layer_stride=0, block_stride=8)],
    )
    # Padding in two independent allocations must not make them one packed page.
    key = torch.zeros(64, dtype=torch.uint8)[:32].view(4, 8)
    value = torch.ones(64, dtype=torch.uint8)[:32].view(4, 8)
    worker.register_kv_caches({"layer": (key, value)})
    assert list(worker.gpu_kv_caches) == ["layer", "layer.1"]
    assert worker.num_cpu_blocks == 16
    assert torch.equal(worker.gpu_kv_caches["layer"], key)
    assert torch.equal(worker.gpu_kv_caches["layer.1"], value)


def test_transfer_hooks_record_store_barrier_once_on_npu(
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
    metadata = SimpleCPUOffloadMetadata(
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

    worker.bind_connector_metadata(metadata)
    # Completion polling must not submit transfers. The engine invokes the
    # load and store hooks separately from get_finished.
    worker._pending_load_event_indices.clear()
    worker._pending_store_event_indices.clear()
    assert worker.get_finished(set()) == (None, None)
    assert worker._backend.calls == []
    worker.start_load_kv()
    worker.wait_for_save()
    worker.wait_for_save()
    worker.clear_connector_metadata()

    load_call, store_call = worker._backend.calls
    assert load_call["is_store"] is False
    assert "wait_event" not in load_call
    assert store_call["is_store"] is True
    store_event = store_call["wait_event"]
    assert isinstance(store_event, FakeEvent)
    assert store_event.recorded_stream is current_stream
    worker.bind_connector_metadata(metadata)
    worker.wait_for_save()
    assert len(worker._backend.calls) == 3


@pytest.mark.parametrize("padding", [0, 16])
def test_offload_interleaved_layers_share_one_page_and_pool_capacity(monkeypatch, padding):
    """A block-outermost descriptor must copy both layers once without resizing."""
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Stream=lambda: object()), raising=False)
    monkeypatch.setattr(worker_module, "is_pin_memory_available", lambda: False)
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker._backend = SimpleNamespace(init=lambda *args: None)
    worker.cpu_capacity_bytes = 256
    worker.kv_cache_config = SimpleNamespace(
        num_blocks=4,
        kv_cache_tensors=[SimpleNamespace(layers=["a", "b"], size=64, offset=0, layer_stride=8, block_stride=16)],
    )
    backing = torch.arange(64 + padding, dtype=torch.uint8)
    caches = {
        name: backing.as_strided((4, 8), (16, 1), storage_offset=padding + offset)
        for name, offset in (("a", 0), ("b", 8))
    }
    original_size = backing.untyped_storage().nbytes()
    worker.register_kv_caches(caches)
    assert worker.num_cpu_blocks == 16
    assert len(worker.gpu_kv_caches) == 1
    assert backing.untyped_storage().nbytes() == original_size
    page = next(iter(worker.gpu_kv_caches.values()))
    assert torch.equal(page.view(torch.uint8), backing[padding:].view(4, 16))
    mirror = next(iter(worker.cpu_kv_caches.values()))
    mirror[0].copy_(page[2])
    saved = page[2].clone()
    page[2].zero_()
    page[2].copy_(mirror[0])
    assert torch.equal(page[2], saved)


def test_offload_rejects_view_past_storage_without_resizing():
    backing = torch.zeros(64, dtype=torch.uint8)
    cache = backing.as_strided((4, 8), (16, 1), storage_offset=8)
    with pytest.raises(ValueError, match="exceeds its backing storage"):
        SimpleCPUOffloadNPUWorker._build_block_views("interleaved", cache, 4)
    assert backing.untyped_storage().nbytes() == 64


@pytest.mark.parametrize("padding", [0, 16])
@pytest.mark.parametrize("split_components", [False, True])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_offload_interleaved_kv_tensor_copies_each_physical_page_once(
    monkeypatch, padding, split_components, num_layers
):
    """MRV1 exposes (K/V, blocks, ...) with K and V interleaved per block."""
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Stream=lambda: object()), raising=False)
    monkeypatch.setattr(worker_module, "is_pin_memory_available", lambda: False)
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker._backend = SimpleNamespace(init=lambda *args: None)
    worker.cpu_capacity_bytes = 256
    names = [f"layer{i}" for i in range(num_layers)]
    worker.kv_cache_config = SimpleNamespace(
        num_blocks=4,
        kv_cache_tensors=[
            SimpleNamespace(layers=names, size=64 * num_layers, offset=0, layer_stride=64, block_stride=16)
        ],
    )
    backings = [torch.arange(64 + padding, dtype=torch.uint8) for _ in names]
    caches = [backing.as_strided((2, 4, 8), (8, 16, 1), storage_offset=padding) for backing in backings]
    worker.register_kv_caches(
        {name: tuple(cache.unbind(0)) if split_components else cache for name, cache in zip(names, caches)}
    )
    assert len(worker.gpu_kv_caches) == num_layers
    assert worker.num_cpu_blocks == 16 // num_layers
    for name, backing, cache in zip(names, backings, caches):
        page = worker.gpu_kv_caches[name]
        assert torch.equal(page.view(torch.uint8), backing[padding:].view(4, 16))
        assert backing.untyped_storage().nbytes() == 64 + padding
        mirror = worker.cpu_kv_caches[name]
        expected = cache[:, 2].clone()
        mirror[0].copy_(page[2])
        cache[:, 2].zero_()
        page[2].copy_(mirror[0])
        assert torch.equal(cache[:, 2], expected)


def test_offload_rejects_fallback_larger_than_scheduler_budget():
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker.cpu_capacity_bytes = 256
    worker.kv_cache_config = SimpleNamespace(
        num_blocks=4,
        kv_cache_tensors=[SimpleNamespace(layers=["layer"], size=32, offset=0, layer_stride=0, block_stride=8)],
    )
    caches = (torch.zeros(4, 8, dtype=torch.uint8), torch.zeros(4, 8, dtype=torch.uint8))
    with pytest.raises(ValueError, match="scheduler block budget"):
        worker.register_kv_caches({"layer": caches})
