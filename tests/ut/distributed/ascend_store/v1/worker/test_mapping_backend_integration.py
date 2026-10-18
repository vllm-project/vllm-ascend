"""Factory -> Mapping/Execution -> native v1 Backend -> CPU byte-copy SDK."""

import ctypes
from types import SimpleNamespace

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeEvent,
    FakeLoadStartGate,
    begin_step,
    make_topology,
    make_worker,
)
from tests.ut.distributed.ascend_store.v1.worker.native_copy_fixtures import ByteCopySDK, native_backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import vllm_adapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import BackendSpec, LayerwiseAccessKind
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import BulkMapping
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import LookupRequest
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import LoadCommand, RangeStoreCommand
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.route import KVPoolRouteSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import base as worker_base
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import worker as layerwise_worker


def test_layerwise_binds_both_mappings_for_each_group_before_visiting_the_next(monkeypatch):
    topology = make_topology(group_ids=(0, 3), physical_layers=(2, 5))
    worker, _, _ = make_worker(topology=topology, layerwise=True, bind=False)
    events = []
    bulk_mapping, layer_mapping = worker_base.BulkMapping, layerwise_worker.LayerwiseMapping

    def bind_bulk(*args, **kwargs):
        events.append(("bulk", kwargs["group_id"]))
        return bulk_mapping(*args, **kwargs)

    def bind_layer(mapping):
        group_id = next(group_id for group_id, bound in worker._mappings.items() if bound is mapping)
        events.append(("layer", group_id))
        return layer_mapping(mapping)

    monkeypatch.setattr(worker_base, "BulkMapping", bind_bulk)
    monkeypatch.setattr(layerwise_worker, "LayerwiseMapping", bind_layer)
    try:
        worker.bind_kv_caches({})
        assert events == [("bulk", 0), ("layer", 0), ("bulk", 3), ("layer", 3)]
        assert tuple(worker._mappings) == tuple(worker._layer_mappings) == (0, 3)
        assert all(worker._layer_mappings[group_id].bulk is mapping for group_id, mapping in worker._mappings.items())
    finally:
        worker.close()


def factory_worker(monkeypatch, name, layerwise, topology, objects):
    sdk = ByteCopySDK(objects)
    backend = native_backend(name, sdk)
    access = LayerwiseAccessKind.KEY_RANGE if name == "mooncake" else LayerwiseAccessKind.GVA
    spec = BackendSpec(name, lambda *_args, **_kwargs: backend, access, True)
    route = KVPoolRouteSpec(topology, name, 64, use_layerwise=layerwise)
    monkeypatch.setattr(vllm_adapter, "resolve_kv_pool_route_spec", lambda *_args: route)
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: spec)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(),
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config={}),
    )
    worker = vllm_adapter.create_kv_pool_worker(config, SimpleNamespace(num_blocks=8, kv_cache_groups=topology.groups))
    worker._source_ready_event_factory = FakeEvent
    if layerwise:
        worker._start_gate_factory = lambda: FakeLoadStartGate(opened=True)
    caches = {}
    for group in topology.transfer_groups:
        for layer in group.layers:
            # Two entries per physical layer, with different payload sizes and page padding.
            caches[layer.layer_names[0]] = (
                (torch.arange(8 * 7, dtype=torch.float32).reshape(8, 7) + group.group_id * 100)[:, :4],
                (torch.arange(8 * 5, dtype=torch.int16).reshape(8, 5) + layer.physical_layer_id * 100)[:, :2],
            )
    worker.bind_kv_caches(caches)
    assert all(isinstance(mapping, BulkMapping) for mapping in worker._mappings.values())
    return worker, caches, sdk


@pytest.mark.parametrize(
    "backend_name,pp_rank,dcp_rank,dcp_size",
    (
        ("mooncake", 0, 0, 1),
        ("mooncake", 1, 0, 1),
        ("memcache", 0, 0, 1),
        ("memcache", 1, 0, 1),
        ("memcache", 1, 0, 2),
        ("memcache", 1, 1, 2),
    ),
)
@pytest.mark.parametrize("layerwise_store", (False, True), ids=("bulk_to_layerwise", "layerwise_to_bulk"))
def test_factory_routes_share_complete_objects(monkeypatch, backend_name, layerwise_store, pp_rank, dcp_rank, dcp_size):
    # Original group IDs are sparse. Nonzero PP uses absolute layers but object-relative offsets.
    topology = make_topology(
        group_ids=(0, 3),
        physical_layers=(2, 5),
        pp_size=pp_rank + 1,
        pp_rank=pp_rank,
        dcp_rank=dcp_rank,
        dcp_size=dcp_size,
        tp_rank=dcp_rank,
        tp_size=dcp_size,
        put_step=dcp_size,
    )
    objects = {}
    writer, source, write_sdk = factory_worker(monkeypatch, backend_name, layerwise_store, topology, objects)
    block_ids = tuple((1,) if group_id in topology.transfer_group_ids else () for group_id in range(4))
    command = RangeStoreCommand("request", TokenRange(0, 4), block_ids, (b"a",), 4, 17)
    expected = {name: tuple(tensor[1].clone() for tensor in tensors) for name, tensors in source.items()}
    try:
        begin_step(writer, store=(command,))
        if layerwise_store:
            for layer_id in (2, 5):
                writer.save_layer(f"layers.{layer_id}.group.0")
        writer.finish_step()
        completions = writer.fence_previous_store()
        assert all(completion.evidence.succeeded for completion in completions)
        assert writer.take_released_store_job_ids() == {17}
        writer.end_step()
        keys = {mapping.keys((b"a",))[0][0] for mapping in writer._mappings.values()}
        assert set(objects) == keys
        assert all(ctypes.sizeof(value) == 40 for value in objects.values())
        assert all(f"@pp_rank:{pp_rank}@" in key for key in keys)
        assert all(f"@dcp:{dcp_rank}@" in key for key in keys)
        for group_id, mapping in writer._mappings.items():
            expected_bytes = b"".join(
                tensor[1].numpy().tobytes()
                for layer in topology.groups[group_id].layers
                for tensor in source[layer.layer_names[0]]
            )
            assert objects[mapping.keys((b"a",))[0][0]].raw == expected_bytes
    finally:
        writer.close()
    assert write_sdk.closed

    reader, destination, read_sdk = factory_worker(monkeypatch, backend_name, not layerwise_store, topology, objects)
    try:
        for tensors in destination.values():
            for tensor in tensors:
                tensor.fill_(-1)
        assert {mapping.keys((b"a",))[0][0] for mapping in reader._mappings.values()} == keys
        if pp_rank == 0:
            assert reader.lookup(LookupRequest(TokenRange(0, 4), (0, 3), (b"a",))).available_end_token == 4
        target_ids = tuple((6,) if group_id in topology.transfer_group_ids else () for group_id in range(4))
        begin_step(reader, load=(LoadCommand("request", TokenRange(0, 4), target_ids, (b"a",)),))
        reader.start_load()
        if not layerwise_store:
            for layer_id in (2, 5):
                reader.wait_for_layer_load(f"layers.{layer_id}.group.0")
        assert not reader.collect_load_result().failed_locations
        for name, tensors in destination.items():
            for tensor, expected_tensor in zip(tensors, expected[name], strict=True):
                torch.testing.assert_close(tensor[6], expected_tensor)
                assert torch.all(tensor[5] == -1)
                assert torch.all(tensor[7] == -1)
        reader.end_step()
    finally:
        reader.close()
    assert read_sdk.closed
