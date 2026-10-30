"""Replay actual CPU payloads through shared preparation and per-layer object sessions."""

import ctypes

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.bulk_prototype.fixtures import plan_load
from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeEvent,
    FakeLoadStartGate,
    make_backend_spec,
    make_coordinator,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.bulk.fixtures import TensorBytesBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.bulk_prototype import LookupQuery, Worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import Resources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    RangeStoreCommand,
    StoreCommandBatch,
)


class TensorRangeBackend(TensorBytesBackend):
    def __init__(self):
        super().__init__()
        self.writes = {}

    def start_key_range_store(self, keys, object_sizes):
        codes = super().start_key_range_store(keys, object_sizes)
        for key, size, code in zip(keys, object_sizes, codes, strict=True):
            if code == 0:
                self.writes[key] = bytearray(size)
        return codes

    def copy_key_range_store(self, keys, addresses, sizes, offsets):
        self._assert_registered_ranges(addresses, sizes)
        codes = super().copy_key_range_store(keys, addresses, sizes, offsets)
        for key, row_addresses, row_sizes, row_offsets, code in zip(
            keys, addresses, sizes, offsets, codes, strict=True
        ):
            if code == 0:
                for address, size, offset in zip(row_addresses, row_sizes, row_offsets, strict=True):
                    self.writes[key][offset : offset + size] = ctypes.string_at(address, size)
        return codes

    def commit_key_range_store(self, keys):
        codes = super().commit_key_range_store(keys)
        for key, code in zip(keys, codes, strict=True):
            if code == 0:
                self.objects[key] = bytes(self.writes.pop(key))
        return codes

    def revoke_key_range_store(self, keys):
        codes = super().revoke_key_range_store(keys)
        for key, code in zip(keys, codes, strict=True):
            if code == 0:
                self.writes.pop(key, None)
        return codes

    def copy_key_range_load(self, keys, addresses, sizes, offsets):
        self._assert_registered_ranges(addresses, sizes)
        codes = super().copy_key_range_load(keys, addresses, sizes, offsets)
        for key, row_addresses, row_sizes, row_offsets, code in zip(
            keys, addresses, sizes, offsets, codes, strict=True
        ):
            if code == 0:
                for address, size, offset in zip(row_addresses, row_sizes, row_offsets, strict=True):
                    ctypes.memmove(address, self.objects[key][offset : offset + size], size)
        return codes


@pytest.mark.parametrize("producer_layerwise,consumer_layerwise", ((True, True), (True, False), (False, True)))
def test_bulk_and_layerwise_share_complete_objects_and_preserve_local_prefix(producer_layerwise, consumer_layerwise):
    topology = make_topology(physical_layers=(2, 5, 9))
    backend = TensorRangeBackend()
    producer_caches = {
        layer.layer_names[0]: torch.arange(64, dtype=torch.float32).reshape(8, 4, 2, 1) + index * 1000
        for index, layer in enumerate(topology.groups[0].layers)
    }
    consumer_caches = {name: torch.full_like(cache, -999) for name, cache in producer_caches.items()}
    consumer_caches[next(iter(consumer_caches))][0] = -123
    producer = Worker(
        topology,
        Resources(backend, make_backend_spec(requires_exists_before_put=True), topology.transfer_groups),
        producer_caches,
        coordinator=make_coordinator(topology),
        layerwise=producer_layerwise,
        source_ready_event_factory=FakeEvent,
        start_gate_factory=lambda: FakeLoadStartGate(opened=True),
    )
    # Independent client lifetime over a shared object dictionary.
    consumer_backend = TensorRangeBackend()
    consumer_backend.objects = backend.objects
    consumer = Worker(
        topology,
        Resources(consumer_backend, make_backend_spec(), topology.transfer_groups),
        consumer_caches,
        coordinator=make_coordinator(topology),
        layerwise=consumer_layerwise,
        source_ready_event_factory=FakeEvent,
        start_gate_factory=lambda: FakeLoadStartGate(opened=True),
    )
    names = tuple(producer_caches)
    hashes = (b"a", b"b")
    try:
        producer.begin_step(
            KVTransferStep(store=StoreCommandBatch((RangeStoreCommand("producer", 0, 8, ((1, 2),), hashes, 8, 17),)))
        )
        for name in names:
            producer.save_layer(name)
        producer.finish_step()
        producer.fence_previous_store()
        producer.end_step()
        assert len(backend.objects) == 2 and not backend.writes

        # A valid four-token prefix is already present at the consumer's Block 5.
        for name in names:
            consumer_caches[name][5] = producer_caches[name][1]
        query = LookupQuery(8, (0,), hashes, local_available_tokens=4)
        recovered = consumer.lookup(query)
        assert recovered.available_end_token == 8
        command = plan_load(
            topology.config,
            query,
            recovered,
            request_id="consumer",
            block_ids_by_group=((5, 6),),
            request_token_count=12,
        )
        assert command.start_token == 4
        consumer.begin_step(KVTransferStep((command,)))
        consumer.start_load()
        for name in names:
            consumer.wait_for_layer_load(name)
        consumer.transfer.executor._write.submit(lambda: None).result(timeout=2)
        assert consumer.collect_load_result().completed_request_ids == {"consumer"}
        consumer.end_step()
        for name in names:
            torch.testing.assert_close(consumer_caches[name][[5, 6]], producer_caches[name][[1, 2]])
            assert torch.all(consumer_caches[name][7] == -999)
        copied = [call for call in consumer_backend.calls if call[0] in ("get", "batch_copy_get")]
        assert all(len(call[1]) == 1 for call in copied)
    finally:
        producer.close()
        consumer.close()


def test_shared_layer_buffers_are_rejected_before_registration_or_copy():
    topology = make_topology(physical_layers=(2, 5))
    backend = TensorRangeBackend()
    resources = Resources(backend, make_backend_spec(), topology.transfer_groups)
    shared = torch.zeros((8, 4, 2, 1))
    caches = {layer.layer_names[0]: shared for layer in topology.groups[0].layers}
    with pytest.raises(ValueError, match="shared layer buffers"):
        Worker(topology, resources, caches, coordinator=make_coordinator(topology), layerwise=True)
    assert backend.closed
    assert not any(call[0] in ("register_buffer", "batch_copy_get", "batch_copy_put") for call in backend.calls)
