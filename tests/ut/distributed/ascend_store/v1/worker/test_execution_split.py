"""Request admission and direct GVA argument correctness."""

import ctypes
from unittest.mock import Mock

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.gva_fixtures import FakeGVABackend, FakeGVAStore
from tests.ut.distributed.ascend_store.v1.helpers import make_topology
from tests.ut.distributed.ascend_store.v1.worker.fixtures import make_worker as make_bulk
from tests.ut.distributed.ascend_store.v1.worker.test_layerwise import layer_names
from tests.ut.distributed.ascend_store.v1.worker.test_layerwise import make_worker as make_layerwise
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    RangeStoreCommand,
    StoreCommandBatch,
)


@pytest.mark.parametrize("layerwise", (False, True))
def test_request_admission_is_retained_until_its_result_is_collected(layerwise):
    worker, _, _ = make_layerwise() if layerwise else make_bulk()
    command = LoadCommand("r", 0, 4, ((1,),), (b"a",))
    try:
        worker.begin_step(KVTransferStep((command,)))
        worker.start_load()
        for name in layer_names(worker):
            worker.wait_for_layer_load(name)
        worker.transfer.executor._write.submit(lambda: None).result(timeout=2)
        worker.end_step()
        worker.begin_step(KVTransferStep((command,)))
        with pytest.raises(RuntimeError, match="pending asynchronous Load"):
            worker.start_load()
        assert worker.collect_load_result().completed_request_ids == {"r"}
        worker.start_load()
        for name in layer_names(worker):
            worker.wait_for_layer_load(name)
        worker.transfer.executor._write.submit(lambda: None).result(timeout=2)
        assert worker.collect_load_result().completed_request_ids == {"r"}
        worker.end_step()
    finally:
        worker.close()


class TensorGVAStore(FakeGVAStore):
    """Keep real CPU buffers at the addresses reported by GVA object sessions."""

    def __init__(self):
        super().__init__()
        self.buffers = {}

    def batch_alloc(self, keys, sizes):
        self.calls.append(("alloc", tuple(keys), tuple(sizes)))
        addresses = []
        for key, size in zip(keys, sizes, strict=True):
            assert key not in self.objects
            buffer = ctypes.create_string_buffer(size)
            address = ctypes.addressof(buffer)
            self.buffers[key] = buffer
            self.objects[key] = (address, size, False)
            addresses.append(address)
        return addresses

    def batch_copy(self, remote, local, sizes, direction):
        code = super().batch_copy(remote, local, sizes, direction)
        for remote_address, local_address, size in zip(remote, local, sizes, strict=True):
            source, destination = (remote_address, local_address) if direction else (local_address, remote_address)
            ctypes.memmove(destination, source, size)
        return code


def test_flat_gva_copies_preserve_bytes_and_fan_one_object_out_to_two_requests():
    backend = FakeGVABackend()
    backend.native_store = TensorGVAStore()
    worker, resources, _ = make_layerwise(backend=backend)
    caches = resources.kv_caches
    names = layer_names(worker)
    for index, cache in enumerate(caches.values()):
        cache.copy_(torch.arange(cache.numel(), dtype=cache.dtype).reshape(cache.shape) + index * 1000)
    expected = {name: caches[name][1].clone() for name in names}
    try:
        producer = RangeStoreCommand("producer", 0, 8, ((1, 2),), (b"a", b"b"), 8, 17)
        worker.begin_step(KVTransferStep(store=StoreCommandBatch((producer,))))
        for name in names:
            worker.save_layer(name)
        worker.finish_step()
        worker.end_step()
        assert worker.take_released_store_job_ids() == {17}
        assert len(backend.native_store.objects) == 2
        for cache in caches.values():
            cache[5:7].fill_(-999)
        commands = (LoadCommand("a", 0, 4, ((5,),), (b"a",)), LoadCommand("b", 0, 4, ((6,),), (b"a",)))
        worker.begin_step(KVTransferStep(commands))
        worker.start_load()
        for name in names:
            worker.wait_for_layer_load(name)
        assert worker.collect_load_result().completed_request_ids == {"a", "b"}
        worker.end_step()
        for name in names:
            torch.testing.assert_close(caches[name][5], expected[name])
            torch.testing.assert_close(caches[name][6], expected[name])
        assert not backend.native_store.leases
    finally:
        worker.close()


def test_empty_hybrid_step_does_not_create_a_compute_gate():
    gate = Mock(side_effect=AssertionError("An empty producer needs no compute gate"))
    worker, _, backend = make_layerwise(
        topology=make_topology(group_ids=(0, 1), physical_layers=(2,)), gate_factory=gate
    )
    worker.begin_step(KVTransferStep())
    worker.start_load()
    for name in layer_names(worker):
        worker.wait_for_layer_load(name)
    worker.finish_step()
    worker.end_step()
    worker.close()
    gate.assert_not_called()
    assert not any(call[0] in ("exists", "batch_copy_get", "batch_copy_put") for call in backend.calls)
