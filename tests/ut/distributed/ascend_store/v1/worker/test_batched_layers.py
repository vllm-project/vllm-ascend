"""Batched geometry preserves bytes, heterogeneous layer membership and Load fan-out."""

from dataclasses import replace

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.gva_fixtures import FakeGVABackend, make_gva_spec
from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeEvent,
    FakeLoadStartGate,
    make_backend_spec,
    make_coordinator,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.test_execution_split import TensorGVAStore
from tests.ut.distributed.ascend_store.v1.worker.test_layerwise import make_worker
from tests.ut.distributed.ascend_store.v1.worker.test_layerwise_bytes import TensorRangeBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import Resources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    RangeStoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import Worker


@pytest.mark.parametrize("mode", ("key_range", "gva"))
def test_partition_batches_preserve_payload_fanout_and_independent_layer_membership(mode):
    topology = make_topology(group_ids=(0, 1), physical_layers=(2, 5, 9))
    topology = replace(
        topology,
        groups=(
            replace(topology.groups[0], layers=(topology.groups[0].layers[0], topology.groups[0].layers[2])),
            replace(topology.groups[1], layers=topology.groups[1].layers[1:]),
        ),
    )
    backend = TensorRangeBackend() if mode == "key_range" else FakeGVABackend()
    if mode == "gva":
        backend.native_store = TensorGVAStore()
        backend.exists = lambda keys: tuple(region is not None for region in backend.query_gva_regions(keys))
    caches = {
        layer.layer_names[0]: torch.arange(64, dtype=torch.float32).reshape(8, 4, 2, 1)
        + 1000 * (group.group_id * 3 + index)
        for group in topology.groups
        for index, layer in enumerate(group.layers)
    }
    expected = {name: cache[1].clone() for name, cache in caches.items()}
    untouched = {name: cache[0].clone() for name, cache in caches.items()}
    spec = make_backend_spec() if mode == "key_range" else make_gva_spec()
    worker = Worker(
        topology,
        Resources(backend, spec, topology.transfer_groups),
        coordinator=make_coordinator(topology),
        layerwise=True,
        source_ready_event_factory=FakeEvent,
        start_gate_factory=lambda: FakeLoadStartGate(opened=True),
    )
    worker.bind_kv_caches(caches)
    names = tuple(sorted(worker.context.layer_ids_by_name, key=worker.context.layer_ids_by_name.get))
    try:
        command = RangeStoreCommand("source", 0, 4, ((1,), (1,)), (b"a",), 4, 17)
        worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
        for name in names:
            worker.save_layer(name)
        worker.finish_step()
        worker.end_step()
        for cache in caches.values():
            cache[5:7].fill_(-999)
        loads = (
            LoadCommand("a", 0, 4, ((5,), (5,)), (b"a",)),
            LoadCommand("b", 0, 4, ((6,), (6,)), (b"a",)),
        )
        worker.begin_step(KVTransferStep(loads))
        worker.start_load()
        for name in names:
            worker.wait_for_layer_load(name)
        assert worker.collect_load_result().completed_request_ids == {"a", "b"}
        worker.end_step()
        for name, cache in caches.items():
            torch.testing.assert_close(cache[5], expected[name])
            torch.testing.assert_close(cache[6], expected[name])
            torch.testing.assert_close(cache[0], untouched[name])
    finally:
        worker.close()


def test_gva_batch_addition_preserves_python_integer_addresses():
    backend = FakeGVABackend()
    worker, _, _ = make_worker(backend=backend)
    command = LoadCommand("r", 0, 4, ((1,),), (b"a",))
    selection = worker.load._prepare((command,))
    size = worker.context.remote[0][0]["size_bytes"]
    base = (1 << 64) - 16
    backend.native_store.objects[selection.keys[0]] = (base, size, True)
    try:
        worker.begin_step(KVTransferStep((command,)))
        worker.start_load()
        for name in sorted(worker.context.layer_ids_by_name, key=worker.context.layer_ids_by_name.get):
            worker.wait_for_layer_load(name)
        assert not worker.collect_load_result().failed_locations
        worker.end_step()
        copied = [address for call in backend.native_store.calls if call[0] == "copy" for address in call[1]]
        offsets = worker.context.remote[0][0]["offsets"].tolist()
        assert copied == [base + offset for offset in offsets]
        assert max(copied) >= (1 << 64)
    finally:
        worker.close()
