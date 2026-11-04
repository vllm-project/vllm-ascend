"""Exercise the E2E byte probe after real Worker binding and restore its copy callbacks."""

import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.e2e.pull_request.one_card.ascendstore_v1_performance import AscendStoreBenchmarkWorkerExtension
from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeEvent,
    FakeLoadStartGate,
    make_backend_spec,
    make_coordinator,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.test_layerwise_bytes import TensorRangeBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import Resources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    RangeStoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import Worker


@pytest.mark.parametrize("layerwise", (False, True))
def test_byte_probe_captures_bound_worker_copies_and_preserves_saved_bytes(tmp_path, monkeypatch, layerwise):
    topology = make_topology(physical_layers=(2, 5))
    topology = replace(topology, config=replace(topology.config, use_layerwise=layerwise))
    backend = TensorRangeBackend()
    caches = {
        layer.layer_names[0]: torch.arange(1, 65, dtype=torch.float32).reshape(8, 4, 2, 1) + index * 1000
        for index, layer in enumerate(topology.groups[0].layers)
    }
    worker = Worker(
        topology,
        Resources(backend, make_backend_spec(), topology.transfer_groups),
        coordinator=make_coordinator(topology),
        layerwise=layerwise,
        source_ready_event_factory=FakeEvent,
        start_gate_factory=lambda: FakeLoadStartGate(opened=True),
    )
    worker.bind_kv_caches(caches)
    connector = SimpleNamespace(worker=worker)
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.kv_transfer", SimpleNamespace(get_kv_transfer_group=lambda: connector)
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=Mock()), raising=False)
    extension = AscendStoreBenchmarkWorkerExtension()
    monkeypatch.setattr(extension, "_drain_stores", worker.fence_previous_store)
    names = tuple(caches)
    expected = {name: cache[1].clone() for name, cache in caches.items()}
    original_store, original_load = backend.copy_key_range_store, backend.copy_key_range_load
    try:
        extension.start_ascendstore_byte_accuracy(str(tmp_path / "saved-kv"))
        command = RangeStoreCommand("source", 0, 4, ((1,),), (b"a",), 4, 17)
        worker.begin_step(KVTransferStep(store=StoreCommandBatch(source_ready_commands=(command,))))
        for name in names:
            worker.save_layer(name)
        worker.finish_step()
        worker.end_step()
        stored = extension.finish_ascendstore_byte_accuracy("store")
        extension.clear_ascendstore_accuracy_kv()
        worker.begin_step(KVTransferStep((LoadCommand("target", 0, 4, ((2,),), (b"a",)),)))
        worker.start_load()
        for name in names:
            worker.wait_for_layer_load(name)
        assert not worker.collect_load_result().failed_locations
        worker.end_step()
        loaded = extension.finish_ascendstore_byte_accuracy("load")
        assert loaded == stored
        assert loaded["bytes"] == sum(value.numel() * value.element_size() for value in expected.values())
        for name, cache in caches.items():
            torch.testing.assert_close(cache[2], expected[name])
    finally:
        if hasattr(extension, "_byte_accuracy"):
            extension.stop_ascendstore_byte_accuracy()
        worker.close()
    if layerwise:
        assert worker.transfer._store_ranges == original_store
        assert worker.transfer._load_ranges == original_load
