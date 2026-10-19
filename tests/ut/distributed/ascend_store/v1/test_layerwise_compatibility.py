"""Production wire/layout compatibility through real v1 factories and CPU copies."""

from __future__ import annotations

import ctypes
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.distributed.ascend_store.test_layerwise_keys import TestBoundLayerwiseKeys
from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeEvent,
    FakeLoadStartGate,
    begin_step,
    make_pool_config,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.native_copy_fixtures import ByteCopySDK, native_backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import memcache_backend, mooncake_layerwise
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import PoolKey
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import factory
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    GVA_SHARD_ALIGNMENT_BYTES,
    resolve_backend_spec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache import layout as memcache_layout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.naming import WireKeyBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration.resolve import (
    resolve_object_partitions,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import BulkMapping
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.messages import (
    LoadCommand,
    LookupRequest,
    RangeStoreCommand,
)


def production_worker(monkeypatch, name, topology, objects, readable, *, store=True, collective=None):
    sdk = ByteCopySDK(objects, readable)
    backend = native_backend(name, sdk)
    spec = replace(resolve_backend_spec(name), backend_factory=lambda *_args, **_kwargs: backend)
    pool = make_pool_config(topology, name, use_layerwise=True, store_enabled=store)
    monkeypatch.setattr(factory, "resolve_backend_spec", lambda _name: spec)
    monkeypatch.setattr(factory, "resolve_kv_pool_config", lambda *_args: pool)
    monkeypatch.setattr(factory, "resolve_worker_topology", lambda *_args: topology)
    if collective is not None:
        monkeypatch.setattr(factory, "StoreCollective", lambda _ranks: collective)
    config = TestBoundLayerwiseKeys.make_config(topology.pp_size)
    config.parallel_config.tensor_parallel_size = topology.tp_size
    config.parallel_config.decode_context_parallel_size = topology.dcp_size
    config.parallel_config.prefill_context_parallel_size = topology.pcp_size
    config.parallel_config.rank = topology.groups[0].key_metadata.pp_rank * topology.tp_size + topology.tp_rank
    config.model_config.max_model_len = 64
    config.model_config.get_layers_start_end_indices = lambda p: (
        (0, 7) if topology.pp_size == 1 else ((0, 3), (3, 7))[p.rank // topology.tp_size]
    )
    config.model_config.get_num_layers = lambda p: 7 if topology.pp_size == 1 else (3, 4)[p.rank // topology.tp_size]
    config.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    config.kv_transfer_config = SimpleNamespace(kv_connector_extra_config={})
    cache = SimpleNamespace(num_blocks=8, kv_cache_groups=topology.groups)
    pp, cp = resolve_object_partitions(
        config,
        name,
        use_layerwise=True,
        put_step=topology.put_step,
        target_layer_counts=topology.consumer_pipeline_partitions,
    )
    stage_start, _ = config.model_config.get_layers_start_end_indices(config.parallel_config)
    topology = replace(
        topology,
        pp_partition=pp,
        cp_partition=cp,
        stage_layer_start=stage_start if topology.pp_size > 1 else 0,
        stage_layer_count=config.model_config.get_num_layers(config.parallel_config),
        object_shard_index=topology.groups[0].key_metadata.dcp_rank if cp.object_shard_count > 1 else 0,
    )
    pool = replace(pool, pp_partition=pp, cp_partition=cp)
    worker = factory.create_kv_pool_worker(config, cache)
    worker._source_ready_event_factory = FakeEvent
    worker._start_gate_factory = lambda: FakeLoadStartGate(opened=True)
    tensors = {
        layer.layer_names[0]: (
            torch.arange(8 * 7, dtype=torch.float32).reshape(8, 7)[:, :4] + topology.tp_rank * 100,
            torch.arange(8 * 5, dtype=torch.int16).reshape(8, 5)[:, :2] + layer.physical_layer_id * 100,
        )
        for group in topology.transfer_groups
        for layer in group.layers
    }
    worker.bind_kv_caches(tensors)
    return worker, tensors, sdk, config, cache


def command(topology):
    ids = tuple((1,) if group.group_id in topology.transfer_group_ids else () for group in topology.groups)
    return RangeStoreCommand("request", 0, 4, ids, (b"a",), 4, 17)


@pytest.mark.parametrize("name", ("mooncake", "memcache"))
@pytest.mark.parametrize("pp_rank", (0, 1))
@pytest.mark.parametrize("group_ids", ((0,), (0, 3)))
@pytest.mark.parametrize("production_writes", (False, True))
def test_factory_round_trip_matches_production_identity_and_bytes(
    monkeypatch, name, pp_rank, group_ids, production_writes
):
    topology = make_topology(group_ids=group_ids, physical_layers=(3, 4), pp_size=pp_rank + 1, pp_rank=pp_rank)
    objects: dict[str, ctypes.Array[ctypes.c_char]] = {}
    readable: set[str] = set()
    monkeypatch.setattr(PoolKey, "to_string", Mock(side_effect=AssertionError("unused canonical prefixes")))
    monkeypatch.setattr(BulkMapping, "__init__", Mock(side_effect=AssertionError("unused Bulk view")))
    if name == "mooncake":
        monkeypatch.setattr(
            memcache_layout, "build_memcache_object_layouts", Mock(side_effect=AssertionError("unused PP/CP placement"))
        )
    worker, tensors, sdk, config, cache = production_worker(monkeypatch, name, topology, objects, readable)
    protocol = mooncake_layerwise if name == "mooncake" else memcache_backend
    production_keys = protocol.bind_layerwise_keys(
        vllm_config=config,
        kv_cache_config=cache,
        model_name="model",
        use_hybrid=len(topology.groups) > 1,
        grouped_block_size=[4] * len(topology.groups),
    )
    object_size = 140 if name == "memcache" else 40
    offset = 60 if name == "memcache" and pp_rank else 0
    expected = {name: tuple(tensor[1].clone() for tensor in values) for name, values in tensors.items()}
    expected_keys = {production_keys.make_full_key(group, "61", 0, pp_rank) for group in group_ids}
    # Traffic consumes bound names and coordinates; no model/configuration rebinding is allowed.
    for owner, method in (
        (WireKeyBuilder, "bind"),
        (protocol, "bind_layerwise_keys"),
        (memcache_layout, "bind_layer_object_span"),
        (memcache_layout, "bind_context_object_span"),
        (config.model_config, "get_layers_start_end_indices"),
        (config.model_config, "get_num_layers"),
    ):
        monkeypatch.setattr(owner, method, Mock(side_effect=AssertionError(f"runtime rebind: {method}")))
    try:
        if production_writes:
            for group_id in group_ids:
                key = production_keys.make_full_key(group_id, "61", 0, pp_rank)
                objects[key] = ctypes.create_string_buffer(object_size)
                payload = b"".join(
                    tensor[1].numpy().tobytes()
                    for layer in topology.groups[group_id].layers
                    for tensor in tensors[layer.layer_names[0]]
                )
                ctypes.memmove(ctypes.addressof(objects[key]) + offset, payload, len(payload))
            readable.update(expected_keys)
        else:
            begin_step(worker, store=(command(topology),))
            for layer_id in (3, 4):
                worker.save_layer(f"layers.{layer_id}.group.{group_ids[0]}")
            worker.finish_step()
            assert all(result.evidence.succeeded for result in worker.fence_previous_store())
            worker.end_step()
        assert set(objects) == expected_keys
        for group_id in group_ids:
            key = production_keys.make_full_key(group_id, "61", 0, pp_rank)
            assert worker._mappings[group_id].naming.keys([b"a"]) == ((key,),)
            assert not hasattr(worker._mappings[group_id], "binding")
            assert (
                worker._group_object_layout(group_id)[1]
                is worker._mappings[group_id].local.physical_layer_ids_by_object
            )
            assert ctypes.sizeof(objects[key]) == object_size
            payload = b"".join(
                tensor[1].numpy().tobytes()
                for layer in topology.groups[group_id].layers
                for tensor in tensors[layer.layer_names[0]]
            )
            assert objects[key].raw[offset : offset + len(payload)] == payload
            assert not any(objects[key].raw[:offset] + objects[key].raw[offset + len(payload) :])
        if pp_rank == 0:
            assert worker.lookup(LookupRequest(0, 4, group_ids, (b"a",))).available_end_token == 4
        for values in tensors.values():
            for tensor in values:
                tensor.fill_(-1)
        ids = tuple((6,) if group.group_id in group_ids else () for group in topology.groups)
        begin_step(worker, load=(LoadCommand("load", 0, 4, ids, (b"a",)),))
        worker.start_load()
        for layer_id in (3, 4):
            worker.wait_for_layer_load(f"layers.{layer_id}.group.{group_ids[0]}")
        assert not worker.collect_load_result().failed_locations
        for tensor_name, values in tensors.items():
            for tensor, saved in zip(values, expected[tensor_name], strict=True):
                torch.testing.assert_close(tensor[6], saved)
                assert torch.all(tensor[5] == -1) and torch.all(tensor[7] == -1)
        worker.end_step()
    finally:
        worker.close()
    assert sdk.closed


class ThreadCollective:
    def __init__(self, rank, barrier, outcomes):
        self.rank, self.barrier, self.outcomes = rank, barrier, outcomes
        self.entered = threading.Event()
        self.closed = False

    def all_gather(self, value):
        self.outcomes[self.rank] = value
        self.entered.set()
        self.barrier.wait(timeout=5)
        result = tuple(self.outcomes)
        self.barrier.wait(timeout=5)
        return result

    def close(self):
        self.closed = True


@pytest.mark.parametrize("failure", (None, "copy", "open", "missing_layer"))
@pytest.mark.parametrize("tp_size", (2, 4))
def test_shared_dcp_writers_do_not_publish_until_every_shard_finishes(monkeypatch, failure, tp_size):
    objects: dict[str, ctypes.Array[ctypes.c_char]] = {}
    readable: set[str] = set()
    barrier, outcomes = threading.Barrier(tp_size), [None] * tp_size
    workers, endpoints, sources, sdks = [], [], [], []
    for rank in range(tp_size):
        topology = make_topology(
            physical_layers=(3, 4),
            tp_rank=rank,
            tp_size=tp_size,
            put_step=tp_size,
            dcp_rank=rank % 2,
            dcp_size=2,
            pp_size=2,
            pp_rank=1,
        )
        endpoint = ThreadCollective(rank, barrier, outcomes)
        worker, tensors, sdk, _, _ = production_worker(
            monkeypatch, "memcache", topology, objects, readable, collective=endpoint
        )
        endpoints.append(endpoint)
        workers.append(worker)
        sources.append(tensors)
        sdks.append(sdk)
    if failure == "copy":
        monkeypatch.setattr(sdks[1], "batch_copy", lambda *_args: -7)
    elif failure == "open":
        monkeypatch.setattr(sdks[1], "batch_alloc", lambda keys, _sizes: [0] * len(keys))
    try:
        for index, worker in enumerate(workers):
            begin_step(worker, store=(command(worker._topology),))
            for layer_id in (3, 4):
                if failure == "missing_layer" and index == 1 and layer_id == 4:
                    continue
                worker.save_layer(f"layers.{layer_id}.group.0")
            worker.end_step()  # Seal asynchronously; do not fence before the peer enters.
            if index == 0:
                assert endpoints[0].entered.wait(timeout=3)
                assert not readable
                assert not any(call[0] == "publish" for call in sdks[0].calls)
        results = [worker.fence_previous_store() for worker in workers]
        assert all(all(result.evidence.succeeded for result in batch) for batch in results) is (failure is None)
        key = "model@pp1@61@0"
        assert set(objects) == {key}
        assert ctypes.sizeof(objects[key]) == 2 * GVA_SHARD_ALIGNMENT_BYTES
        if failure is None:
            assert readable == {key}
            for rank, tensors in enumerate(sources[:2]):
                payload = b"".join(tensor[1].numpy().tobytes() for values in tensors.values() for tensor in values)
                offset = rank * GVA_SHARD_ALIGNMENT_BYTES + 60
                assert objects[key].raw[offset : offset + len(payload)] == payload
        else:
            assert not readable
            assert not any(call[0] == "publish" for sdk in sdks for call in sdk.calls)
            assert workers[0]._backend_sessions.pending_store_keys == (key,)
    finally:
        # The real adapter correctly retains failed writes without an ownership-safe
        # abort. Only this fake CPU store can confirm disposal for test teardown.
        for worker in workers:
            worker._backend.revoke_gva = Mock(side_effect=lambda keys: (0,) * len(keys))
            worker.close()
    assert all(endpoint.closed for endpoint in endpoints)
