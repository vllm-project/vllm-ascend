"""Validate AscendStore v1 through its token-space domain model."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
    RequestTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendAdapter,
    BackendStoreEvidence,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector import AscendStoreV1Connector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.execution.io import (
    BackendExistenceMissingFilter,
    BackendIO,
    IdentityMissingFilter,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.execution.resources import KVResources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.execution.timeline import (
    AsynchronousLoadTimeline,
    LoadCompletion,
    LoadTransfer,
    StoreTimeline,
    SynchronousLoadTimeline,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.graph.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.graph.elements import (
    BindingBatch,
    KVBinding,
    KVChunk,
    LocalKVSlice,
    PhysicalCoordinate,
    RemoteKVObject,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.graph.projection import (
    ContiguousBindingProjection,
    IdentityConsumerProjection,
    KVProjection,
    PipelinePartitionConsumerProjection,
    StridedBindingProjection,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.graph.reachability import (
    ChunkAvailability,
    GroupAvailability,
    GroupSelection,
    HybridReachability,
    KVSelection,
    UnitaryReachability,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.graph.topology import (
    KVCacheGroupTopology,
    KVTopology,
    TPPartitionSpec,
    resolve_consumer_pipeline_partitions,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.kv_pool import KVPoolGraph
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.availability import (
    ExternalPrefixPlan,
    LookupQuery,
    RemoteAvailability,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.planner import TransferPlanner
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.progress import (
    AllocationLoadPublication,
    LoadCandidate,
    RequestSnapshot,
    ScheduledLoadPublication,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.spec import TransferPlanningSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import (
    LookupCodec,
    LookupRequest,
    LookupResult,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    KVTransferStep,
    LoadCommand,
    LoadCommandBatch,
    StoreCommand,
    StoreCommandBatch,
)


class FakeDatabase:
    hash_block_size = 4

    def __init__(self, block_sizes: dict[int, int] | None = None) -> None:
        self.block_sizes = block_sizes or {0: 4}
        self.group_kv_caches_base_addr = {group_id: [1000 + group_id * 1000] for group_id in self.block_sizes}
        self.group_block_len = {group_id: [block_size * 8] for group_id, block_size in self.block_sizes.items()}
        self.group_block_stride = {group_id: [block_size * 8] for group_id, block_size in self.block_sizes.items()}

    def get_block_size(self, group_id: int) -> int:
        return self.block_sizes[group_id]

    def process_token_key_strings(
        self,
        token_len,
        block_hashes,
        mask_num,
        kv_cache_group_id,
        chunk_filter,
    ):
        block_size = self.block_sizes[kv_cache_group_id]
        records = []
        for start in range(0, token_len, block_size):
            end = min(start + block_size, token_len)
            if end - start < block_size or start < mask_num or not chunk_filter(start):
                continue
            hash_index = end // self.hash_block_size - 1
            if hash_index >= len(block_hashes):
                continue
            key = (
                f"@model:test@head_or_tp_rank:0@dcp:0@pp_rank:0@group:{kv_cache_group_id}@chunk:{start // block_size}@"
            )
            records.append((start, end, key, block_hashes[hash_index]))
        return records

    def decode_adaptor_prefill_pp(self, keys, addresses, sizes, kv_cache_group_id):
        del kv_cache_group_id
        return [f"{keys[0]}pp:0", f"{keys[0]}pp:1"], [[addresses[0][0]], [addresses[0][0] + 8]], [sizes, sizes]


class FakeBackend:
    requires_exists_before_put = False

    def __init__(self) -> None:
        self.presence: list[int] = []
        self.get_result: list[int] | None = []
        self.put_result = BackendStoreEvidence((), True, True)
        self.calls = []

    def exists(self, keys):
        self.calls.append(("exists", tuple(keys)))
        return self.presence or [1] * len(keys)

    def get(self, keys, addresses, sizes):
        self.calls.append(("get", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
        return self.get_result

    def put(self, keys, addresses, sizes):
        self.calls.append(("put", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
        return self.put_result

    def set_device(self):
        self.calls.append(("set_device",))


class FakeEvent:
    def __init__(self) -> None:
        self.recorded = False
        self.synchronized = False

    def record(self) -> None:
        self.recorded = True

    def synchronize(self) -> None:
        self.synchronized = True


class FakeResources:
    def __init__(self, database, backend) -> None:
        self.token_database = database
        self.backend = backend
        self.registered = False
        self.closed = False

    def register_kv_caches(self, kv_caches) -> None:
        self.registered = bool(kv_caches)

    def close(self) -> None:
        self.closed = True


class FakeAvailabilityProbe:
    def __init__(self, availability: RemoteAvailability | None) -> None:
        self.availability = availability
        self.queries = []
        self.closed = False

    def query(self, query):
        self.queries.append(query)
        return self.availability

    def close(self):
        self.closed = True


def make_topology(
    *,
    groups=(0,),
    tp_mismatch=False,
    tp_rank=0,
    pcp_rank=0,
    pcp_size=1,
    key_rank_count=None,
) -> KVTopology:
    group_topologies = tuple(
        KVCacheGroupTopology(group_id, 4, (f"layers.{group_id}",), KeyMetadata("model", 0, 0, 0, group_id))
        for group_id in range(max(groups) + 1)
    )
    partition = TPPartitionSpec(
        tp_mismatch,
        key_rank_count or (2 if tp_mismatch else 1),
        2 if tp_mismatch else 1,
    )
    return KVTopology(
        tp_rank,
        1,
        1,
        pcp_rank,
        pcp_size,
        1,
        1,
        4,
        4,
        partition,
        group_topologies,
        tuple(groups),
        None,
    )


def make_projection(database, topology: KVTopology) -> KVProjection:
    binding_projection = (
        StridedBindingProjection(database, topology)
        if topology.tp_partition.tp_mismatch
        else ContiguousBindingProjection(database)
    )
    return KVProjection(database, topology, binding_projection)


def make_binding_batch(group_id=0, *, coordinate=None) -> BindingBatch:
    coordinate = coordinate or PhysicalCoordinate()
    chunk = KVChunk(group_id, 0, TokenRange(0, 4), b"a")
    remote = RemoteKVObject(chunk, "key", coordinate)
    local = LocalKVSlice(chunk, 1, (100,), (16,), coordinate)
    binding = KVBinding(remote, local)
    return BindingBatch(group_id, (binding,))


def make_partition_config(*, role="kv_consumer", layers=10, **extra_config):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role=role, kv_connector_extra_config=extra_config),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=layers)),
    )


def test_consumer_pipeline_uses_explicit_prefill_partitions() -> None:
    config = make_partition_config(consumer_is_to_put=True, prefill_pp_size=3, prefill_pp_layer_partition="2,3,5")
    assert resolve_consumer_pipeline_partitions(config) == (2, 3, 5)


def test_consumer_pipeline_preserves_default_partition_distribution() -> None:
    config = make_partition_config(consumer_is_to_put=True, prefill_pp_size=3)
    assert resolve_consumer_pipeline_partitions(config) == (3, 4, 3)


def test_resources_pass_consumer_partitions_to_token_database(monkeypatch) -> None:
    topology = replace(make_topology(), consumer_pipeline_partitions=(2, 2))
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.execution.resources.create_backend",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    resources = KVResources.create(
        SimpleNamespace(),
        {"backend": "mooncake"},
        topology.kv_cache_groups,
        topology.hash_block_size,
        8,
        topology.consumer_pipeline_partitions,
    )
    assert resources.token_database.partitions == [2, 2]


def test_resources_register_memory_geometry_once() -> None:
    registered_buffers = []
    backend = SimpleNamespace(
        register_buffer=lambda addresses, sizes: registered_buffers.append((addresses, sizes)),
    )
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0, 0)], [4], None, hash_block_size=4)
    resources = KVResources(backend, database, 2, {0: ("layers.0", "layers.1")})
    kv_caches = {
        "layers.0": torch.zeros((2, 1)),
        "layers.1": torch.zeros((2, 1)),
    }
    resources.register_kv_caches(kv_caches)
    assert database.group_num_layers["kv"] == {0: 2}
    assert len(registered_buffers) == 1
    assert len(registered_buffers[0][0]) == 2
    with pytest.raises(RuntimeError, match="already registered"):
        resources.register_kv_caches(kv_caches)


def make_kv_pool_graph(
    backend,
    database=None,
    *,
    groups=(0,),
    async_load=False,
    tp_mismatch=False,
    store=True,
    registered=False,
    key_rank_count=None,
):
    database = database or FakeDatabase({group_id: 4 for group_id in groups})
    resources = FakeResources(database, backend)
    topology = make_topology(groups=groups, tp_mismatch=tp_mismatch, key_rank_count=key_rank_count)
    backend_io = BackendIO(backend)
    load_timeline = AsynchronousLoadTimeline(backend.set_device) if async_load else SynchronousLoadTimeline()
    projection = make_projection(database, topology)
    consumer_projection = IdentityConsumerProjection()
    missing_filter = (
        BackendExistenceMissingFilter(backend) if backend.requires_exists_before_put else IdentityMissingFilter()
    )
    graph = KVPoolGraph(
        resources,
        topology,
        UnitaryReachability(groups[0], 64, 4)
        if len(groups) == 1
        else SimpleNamespace(
            group_ids=groups,
            select_for_load=lambda hashes, token_range: KVSelection(
                token_range,
                tuple(hashes),
                tuple(GroupSelection(group_id, None) for group_id in groups),
            ),
            select_for_store=lambda hashes, token_range, prompt: KVSelection(
                token_range,
                tuple(hashes),
                tuple(GroupSelection(group_id, None) for group_id in groups),
            ),
        ),
        projection,
        consumer_projection,
        missing_filter,
        backend_io,
        load_timeline,
        StoreTimeline(backend.set_device) if store else None,
    )
    if registered:
        graph.register_kv_caches({"cache": object()})
    else:
        projection.compile_memory_mapping()
    return graph, resources


@pytest.mark.parametrize(("start_token", "end_token"), [(-1, 0), (4, 3)])
def test_token_range_rejects_invalid_coordinates(start_token, end_token) -> None:
    with pytest.raises(ValueError):
        TokenRange(start_token, end_token)


def test_binding_requires_the_same_chunk_and_subrepresentation() -> None:
    chunk = KVChunk(0, 0, TokenRange(0, 4), b"a")
    other = replace(chunk, block_index=1)
    with pytest.raises(ValueError, match="same semantic chunk"):
        KVBinding(RemoteKVObject(chunk, "key"), LocalKVSlice(other, 1, (100,), (16,)))
    with pytest.raises(ValueError, match="same physical subrepresentation"):
        KVBinding(
            RemoteKVObject(chunk, "key", PhysicalCoordinate(effective_tp_rank=0)),
            LocalKVSlice(chunk, 1, (100,), (16,), PhysicalCoordinate(effective_tp_rank=1)),
        )


def test_projection_preserves_original_group_identity() -> None:
    database = FakeDatabase({0: 4, 3: 4})
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(3, None),))
    projection = make_projection(database, make_topology(groups=(3,)))
    batch = projection.project_remote_objects(projection.project_chunks(selection))[0]
    assert batch.group_id == 3
    assert batch.chunks[0].group_id == 3
    assert "@group:3@" in batch.remote_objects[0].key


def test_projection_reuses_every_lookup_coordinate() -> None:
    database = FakeDatabase()
    topology = replace(make_topology(), pp_size=2, dcp_size=2, tp_partition=TPPartitionSpec(False, 2, 1))
    projection = make_projection(database, topology)
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    projected_groups = projection.project_chunks(selection)
    first = projection.project_remote_objects(projected_groups)[0]
    second = projection.project_remote_objects(projected_groups)[0]
    assert len(first.remote_objects) == 8
    assert len({remote.coordinate for remote in first.remote_objects}) == 8
    assert all(left.coordinate is right.coordinate for left, right in zip(first.remote_objects, second.remote_objects))


def test_projection_keeps_lookup_rank_major_key_order() -> None:
    database = FakeDatabase()
    topology = replace(make_topology(), pp_size=2, dcp_size=2, tp_partition=TPPartitionSpec(False, 2, 1))
    projection = make_projection(database, topology)
    selection = KVSelection(TokenRange(0, 8), (b"a", b"b"), (GroupSelection(0, None),))
    remote_objects = projection.project_remote_objects(projection.project_chunks(selection))[0].remote_objects
    assert [remote.coordinate for remote in remote_objects] == [
        PhysicalCoordinate(pp_rank=pp_rank, dcp_rank=dcp_rank, head_rank=head_rank)
        for pp_rank in range(2)
        for dcp_rank in range(2)
        for head_rank in range(2)
        for _ in range(2)
    ]
    assert [remote.key for remote in remote_objects] == [
        f"@model:test@head_or_tp_rank:{head_rank}@dcp:{dcp_rank}@pp_rank:{pp_rank}@group:0@chunk:{chunk_index}@"
        for pp_rank in range(2)
        for dcp_rank in range(2)
        for head_rank in range(2)
        for chunk_index in range(2)
    ]


def test_single_lookup_representation_rewrites_a_nonzero_base_rank() -> None:
    metadata = KeyMetadata("model", 3, 0, 0)
    database = ChunkedTokenDatabase([metadata], [4], None, hash_block_size=4)
    topology = make_topology()
    topology = replace(
        topology,
        kv_cache_groups=(replace(topology.kv_cache_groups[0], key_metadata=metadata, uses_align_state=True),),
    )
    projection = make_projection(database, topology)
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    remote_object = projection.project_remote_objects(projection.project_chunks(selection))[0].remote_objects[0]
    assert "@head_or_tp_rank:0@" in remote_object.key


def test_binding_requires_registered_memory_mapping() -> None:
    projection = make_projection(FakeDatabase(), make_topology())
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    allocation_batches = projection.assign_local_blocks(projection.project_chunks(selection), ((1,),))
    with pytest.raises(RuntimeError, match="before cache registration"):
        projection.bind_representations(allocation_batches)


def test_registered_memory_geometry_is_bound_once() -> None:
    database = FakeDatabase()
    projection = make_projection(database, make_topology())
    projection.compile_memory_mapping()
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    allocation_batches = projection.assign_local_blocks(projection.project_chunks(selection), ((1,),))

    database.group_kv_caches_base_addr[0][0] = 5000
    local_slice = projection.bind_representations(allocation_batches)[0].bindings[0].local_slice

    assert local_slice.addresses == (1032,)
    with pytest.raises(RuntimeError, match="already bound"):
        projection.compile_memory_mapping()


def test_strided_mapping_constructs_matching_edges_atomically() -> None:
    database = FakeDatabase()
    topology = make_topology(tp_mismatch=True, tp_rank=1)
    projection = make_projection(database, topology)
    projection.compile_memory_mapping()
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    allocation_batches = projection.assign_local_blocks(projection.project_chunks(selection), ((3,),))
    batch = projection.bind_representations(allocation_batches)[0]
    assert [binding.remote_object.coordinate.effective_tp_rank for binding in batch.bindings] == [2, 3]
    assert all(binding.remote_object.coordinate == binding.local_slice.coordinate for binding in batch.bindings)
    assert "@head_or_tp_rank:2@" in batch.bindings[0].remote_object.key


def test_strided_mapping_compiles_each_memory_segment_geometry() -> None:
    database = FakeDatabase()
    database.group_kv_caches_base_addr[0] = [1000, 2000]
    database.group_block_len[0] = [32, 64]
    database.group_block_stride[0] = [32, 64]
    topology = make_topology(tp_mismatch=True)
    projection = make_projection(database, topology)
    projection.compile_memory_mapping()
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    allocation_batches = projection.assign_local_blocks(projection.project_chunks(selection), ((1,),))
    first_representation = projection.bind_representations(allocation_batches)[0].bindings[0]
    assert first_representation.local_slice.sizes == (4, 4, 4, 4, 8, 8, 8, 8)


def test_consumer_pipeline_projection_rejects_misaligned_memory_segments() -> None:
    database = FakeDatabase()
    database.group_kv_caches_base_addr[0] = [1000, 2000]
    database.group_block_len[0] = [32, 32]
    database.group_block_stride[0] = [32, 32]
    database.group_num_layers = {"kv": {0: 2}}
    consumer_projection = PipelinePartitionConsumerProjection(database, (1, 1))
    consumer_projection.compile_memory_mapping()
    projection = make_projection(database, make_topology())
    projection.compile_memory_mapping()
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    allocation_batches = projection.assign_local_blocks(projection.project_chunks(selection), ((1,),))
    batches = projection.bind_representations(projection.select_owned_allocations(allocation_batches))
    binding = batches[0].bindings[0]
    bad_slice = replace(binding.local_slice, addresses=(binding.local_slice.addresses[0],))
    batches = (BindingBatch(0, (KVBinding(binding.remote_object, bad_slice),)),)
    with pytest.raises(ValueError, match="misaligned"):
        consumer_projection.project(batches)


def test_consumer_pipeline_projection_uses_compiled_layer_segments() -> None:
    database = FakeDatabase()
    database.group_kv_caches_base_addr[0] = [1000, 2000]
    database.group_block_len[0] = [32, 32]
    database.group_block_stride[0] = [32, 32]
    database.group_num_layers = {"kv": {0: 2}}
    consumer_projection = PipelinePartitionConsumerProjection(database, (1, 1))
    consumer_projection.compile_memory_mapping()
    projection = make_projection(database, make_topology())
    projection.compile_memory_mapping()
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    allocation_batches = projection.assign_local_blocks(projection.project_chunks(selection), ((1,),))
    batches = projection.bind_representations(projection.select_owned_allocations(allocation_batches))
    projected = consumer_projection.project(batches)[0].bindings
    assert [binding.local_slice.addresses for binding in projected] == [(1032,), (2032,)]
    assert [binding.remote_object.coordinate.consumer_pp_slice for binding in projected] == [0, 1]
    assert "@pp_rank:1@" in projected[1].remote_object.key


def test_store_ownership_precedes_physical_fanout() -> None:
    database = FakeDatabase()
    topology = make_topology(tp_mismatch=True, pcp_rank=1, pcp_size=2)
    projection = make_projection(database, topology)
    projection.compile_memory_mapping()
    selection = KVSelection(TokenRange(0, 8), (b"a", b"b"), (GroupSelection(0, None),))
    allocation_batches = projection.assign_local_blocks(projection.project_chunks(selection), ((1, 2),))
    batch = projection.bind_representations(projection.select_owned_allocations(allocation_batches))[0]
    assert {binding.local_slice.block_id for binding in batch.bindings} == {2}
    assert len(batch.bindings) == 2


def test_strided_store_ownership_does_not_drop_chunks_as_tp_replicas() -> None:
    database = FakeDatabase()
    topology = replace(make_topology(tp_mismatch=True), put_step=2)
    projection = make_projection(database, topology)
    projection.compile_memory_mapping()
    selection = KVSelection(TokenRange(0, 8), (b"a", b"b"), (GroupSelection(0, None),))
    allocation_batches = projection.assign_local_blocks(projection.project_chunks(selection), ((1, 2),))
    batch = projection.bind_representations(projection.select_owned_allocations(allocation_batches))[0]
    assert {binding.local_slice.block_id for binding in batch.bindings} == {1, 2}
    assert len(batch.bindings) == 4


def test_strided_mapping_keeps_align_state_null_blocks() -> None:
    database = FakeDatabase()
    topology = make_topology(tp_mismatch=True)
    topology = replace(topology, kv_cache_groups=(replace(topology.kv_cache_groups[0], uses_align_state=True),))
    projection = make_projection(database, topology)
    projection.compile_memory_mapping()
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    allocation_batches = projection.assign_local_blocks(projection.project_chunks(selection), ((0,),))
    batch = projection.bind_representations(allocation_batches)[0]
    assert len(batch.bindings) == 2
    assert all(binding.local_slice.block_id == 0 for binding in batch.bindings)


def test_identity_consumer_projection_preserves_strided_bindings() -> None:
    database = FakeDatabase()
    topology = make_topology(tp_mismatch=True)
    projection = make_projection(database, topology)
    projection.compile_memory_mapping()
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    allocation_batches = projection.assign_local_blocks(projection.project_chunks(selection), ((1,),))
    batches = projection.bind_representations(projection.select_owned_allocations(allocation_batches))
    assert IdentityConsumerProjection().project(batches) is batches


def test_unitary_reachability_resolves_only_contiguous_available_chunks() -> None:
    reachability = UnitaryReachability(0, max_model_len=64, cache_transfer_granularity=4)
    selection = reachability.select_for_lookup((b"a", b"b", b"c"), TokenRange(0, 12))
    availability = GroupAvailability(
        0,
        (
            ChunkAvailability(TokenRange(0, 4), b"a", True),
            ChunkAvailability(TokenRange(4, 8), b"b", False),
            ChunkAvailability(TokenRange(8, 12), b"c", True),
        ),
    )
    assert reachability.resolve_available_end(selection, (availability,)) == 4


def test_kv_pool_graph_lookup_requires_every_physical_representation() -> None:
    backend = FakeBackend()
    backend.presence = [1, 0]
    graph, _ = make_kv_pool_graph(backend, store=False, key_rank_count=2)
    result = graph.lookup(LookupRequest(TokenRange(0, 4), (0,), (b"a",)))
    assert result == LookupResult(0)


def test_kv_pool_graph_lookup_contains_backend_protocol_failure_as_a_miss() -> None:
    backend = FakeBackend()
    backend.presence = [1, 1]
    graph, _ = make_kv_pool_graph(backend, store=False)
    result = graph.lookup(LookupRequest(TokenRange(0, 4), (0,), (b"a",)))
    assert result == LookupResult(0)


def test_hybrid_lookup_and_store_share_retention_policy(monkeypatch) -> None:
    recorded = []

    def reachable_mask(_manager, **kwargs):
        recorded.append(kwargs["retention_interval"])
        return None

    import vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.graph.reachability as module

    monkeypatch.setattr(module, "_reachable_block_mask", reachable_mask)
    reachability = HybridReachability(
        (0, 1),
        [
            KVCacheGroupSpec(
                ["layers.0"],
                FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32),
            ),
            KVCacheGroupSpec(
                ["layers.1"],
                FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32),
            ),
        ],
        scheduler_block_size=4,
        hash_block_size=4,
        max_model_len=64,
        retention_interval=16,
    )
    reachability.lookup_mask(8)
    reachability.store_mask(8)
    assert recorded == [16, 16, 16, 16]


def test_lookup_codec_round_trip_preserves_token_coordinates_and_groups() -> None:
    codec = LookupCodec()
    request = LookupRequest(TokenRange(4, 12), (1, 3), (b"a", b"b"))
    assert codec.decode_request(codec.encode_request(request)) == request
    assert codec.decode_result(codec.encode_result(LookupResult(8))) == LookupResult(8)


def test_backend_io_keeps_results_attached_to_exact_bindings() -> None:
    backend = FakeBackend()
    backend.get_result = [0]
    outcomes = BackendIO(backend).load(make_binding_batch().bindings)
    assert outcomes[0].binding == make_binding_batch().bindings[0]
    assert outcomes[0].result_code == 0


@pytest.mark.parametrize("native_result", [None, [0, 0]])
def test_backend_io_marks_misaligned_load_results_unknown(native_result) -> None:
    backend = FakeBackend()
    backend.get_result = native_result
    outcomes = BackendIO(backend).load(make_binding_batch().bindings)
    assert [outcome.result_code for outcome in outcomes] == [None]


def test_synchronous_and_asynchronous_load_share_the_same_operation() -> None:
    backend = FakeBackend()
    backend.get_result = [0]
    backend_io = BackendIO(backend)
    binding_batch = make_binding_batch()
    transfer = LoadTransfer("request", binding_batch.bindings)

    def operation(item: LoadTransfer) -> LoadCompletion:
        return LoadCompletion(item.request_id, backend_io.load(item.traversal))

    synchronous = SynchronousLoadTimeline()
    synchronous.attach_operation(operation)
    synchronous.start()
    assert tuple(synchronous.submit([transfer]))[0].binding_evidence[0].binding == binding_batch.bindings[0]

    asynchronous = AsynchronousLoadTimeline(backend.set_device)
    asynchronous.attach_operation(operation)
    asynchronous.start()
    asynchronous.submit([transfer])
    asynchronous._queue.join()
    completion = asynchronous.collect()[0]
    asynchronous.close()
    assert completion.binding_evidence[0].binding == binding_batch.bindings[0]


def test_asynchronous_load_failure_terminates_and_drains_pending_requests() -> None:
    backend = FakeBackend()

    def fail_load(*args, **kwargs):
        raise RuntimeError("backend get failed")

    timeline = AsynchronousLoadTimeline(backend.set_device)
    timeline.attach_operation(fail_load)
    timeline.start()
    binding_batch = make_binding_batch()
    timeline.submit(
        [
            LoadTransfer("first", binding_batch.bindings),
            LoadTransfer("second", binding_batch.bindings),
        ]
    )
    timeline.join(timeout=1)
    assert not timeline.is_alive()
    timeline._queue.join()
    with pytest.raises(RuntimeError, match="unfinished requests: .*first.*second"):
        timeline.collect()
    with pytest.raises(RuntimeError, match="terminated during asynchronous Load"):
        timeline.submit([])
    with pytest.raises(RuntimeError, match="terminated during asynchronous Load"):
        timeline.close()


def test_kv_pool_graph_filters_remote_objects_before_waiting_for_source(monkeypatch) -> None:
    backend = FakeBackend()
    backend.requires_exists_before_put = True
    backend.presence = [1]
    event = FakeEvent()
    graph, _ = make_kv_pool_graph(backend, registered=True)
    monkeypatch.setattr(torch.npu, "Event", lambda: event)
    graph.submit_store(StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),)))
    result = graph.wait_for_previous_store()
    graph.close()
    assert result[0].evidence.succeeded
    assert not event.synchronized
    assert [call[0] for call in backend.calls] == ["set_device", "exists"]


def test_kv_pool_graph_keeps_the_configured_missing_filter(monkeypatch) -> None:
    backend = FakeBackend()
    graph, _ = make_kv_pool_graph(backend, registered=True)
    backend.requires_exists_before_put = True
    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    graph.submit_store(StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),)))
    graph.wait_for_previous_store()
    graph.close()
    assert "exists" not in [call[0] for call in backend.calls]


def test_kv_pool_graph_preserves_unknown_source_release_after_put_failure(monkeypatch) -> None:
    backend = FakeBackend()
    backend.put_result = BackendStoreEvidence(None, False, False, RuntimeError("put failed"))
    graph, resources = make_kv_pool_graph(backend, registered=True)
    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    graph.submit_store(StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),)))
    with pytest.raises(RuntimeError, match="Store failed"):
        graph.wait_for_previous_store()
    batch = graph._pending_store
    assert batch is not None
    result = batch.completions[0]
    assert not result.evidence.source_release_confirmed
    assert isinstance(result.evidence.error, RuntimeError)
    with pytest.raises(RuntimeError, match="previous Store failure"):
        graph.close()
    assert graph._store_timeline is not None
    assert not graph._store_timeline.is_alive()
    assert not resources.closed


def test_kv_pool_graph_loads_multiple_groups_in_one_backend_call() -> None:
    backend = FakeBackend()
    backend.get_result = [0, 0]
    graph, _ = make_kv_pool_graph(backend, FakeDatabase({0: 4, 1: 4}), groups=(0, 1), store=False)
    graph.load(LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,), (2,)), (b"a",)),)))
    get_calls = [call for call in backend.calls if call[0] == "get"]
    assert len(get_calls) == 1
    assert len(get_calls[0][1]) == 2


def test_kv_pool_graph_reports_grouped_load_failure_at_request_scope() -> None:
    backend = FakeBackend()
    backend.get_result = [0, -1]
    graph, _ = make_kv_pool_graph(backend, FakeDatabase({0: 4, 1: 4}), groups=(0, 1), store=False)
    with pytest.raises(RuntimeError, match="Hybrid KV Load failed"):
        graph.load(LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,), (2,)), (b"a",)),)))


def test_kv_pool_graph_preserves_nonzero_single_group_failure_identity() -> None:
    backend = FakeBackend()
    backend.get_result = [-1]
    graph, _ = make_kv_pool_graph(backend, FakeDatabase({3: 4}), groups=(3,), store=False)
    block_ids = ((), (), (), (7,))
    graph.load(LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), block_ids, (b"a",)),)))
    assert graph.collect_load_result().failed_block_ids == {7}


def test_kv_pool_graph_async_load_publishes_only_after_backend_completion() -> None:
    backend = FakeBackend()
    backend.get_result = [0]
    graph, _ = make_kv_pool_graph(backend, async_load=True, store=False, registered=True)
    graph.load(LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",)),)))
    graph._load_timeline._queue.join()
    assert graph.collect_load_result().completed_request_ids == {"request"}
    graph.close()


def test_kv_pool_graph_store_projects_then_fences_one_step(monkeypatch) -> None:
    backend = FakeBackend()
    backend.put_result = BackendStoreEvidence((0,), True, True)
    graph, resources = make_kv_pool_graph(backend, registered=True)
    event = FakeEvent()
    monkeypatch.setattr(torch.npu, "Event", lambda: event)
    graph.submit_store(StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),)))
    results = graph.wait_for_previous_store()
    graph.close()
    assert results[0].evidence.succeeded
    assert event.synchronized
    assert [call[0] for call in backend.calls].count("put") == 1
    assert resources.closed


def test_kv_pool_graph_close_keeps_resources_when_store_source_release_is_unknown() -> None:
    backend = FakeBackend()
    graph, resources = make_kv_pool_graph(backend, store=True)
    graph._pending_store = SimpleNamespace()
    graph._store_error = RuntimeError("unknown source release")
    with pytest.raises(RuntimeError):
        graph.close()
    assert not resources.closed


def make_planner(availability, *, async_load=False, save_decode=False, store_enabled=True):
    publication = AllocationLoadPublication() if async_load else ScheduledLoadPublication()
    planner = TransferPlanner(
        TransferPlanningSpec(4, 4, (0,), True),
        FakeAvailabilityProbe(availability),
        publication,
        store_enabled=store_enabled,
        save_decode_cache=save_decode,
    )
    return planner


def test_planner_full_hit_keeps_one_token_for_vllm_but_loads_allocated_chunk() -> None:
    planner = make_planner(RemoteAvailability(TokenRange(0, 12), 11))
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[0] * 12,
        num_tokens=12,
        block_hashes=[b"a", b"b", b"c"],
    )
    result = planner.lookup(LookupQuery("request", 12, 12, request.block_hashes, 0))
    planner.confirm_allocation(request, ([1, 2, 3],), 11)
    step = planner.build_step(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[SimpleNamespace(req_id="request", num_computed_tokens=11, block_ids=([1, 2, 3],))],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
            num_scheduled_tokens={"request": 1},
        )
    )
    assert result == ExternalPrefixPlan(11, False)
    assert step.load.commands[0].load_range == TokenRange(0, 12)
    assert step.store.commands == ()


@pytest.mark.parametrize(
    ("target_tokens", "saved_tokens", "hash_count", "load_tokens", "can_load", "discard_partial"),
    [
        (8, 0, 3, None, False, True),
        (8, 8, 3, None, False, True),
        (8, 0, 3, 8, True, True),
        (8, 0, 3, 8, False, True),
        (12, 8, 2, None, False, True),
        (7, 0, 2, None, False, True),
        (7, 0, 2, None, False, False),
    ],
)
def test_planner_transfer_ranges_match_legacy_standard_operation(
    target_tokens,
    saved_tokens,
    hash_count,
    load_tokens,
    can_load,
    discard_partial,
) -> None:
    hashes = [bytes([index + 1]) for index in range(hash_count)]
    legacy_tracker = RequestTracker(
        "request",
        target_tokens,
        allocated_block_ids_by_group=[[1, 2, 3]],
        num_saved_tokens=saved_tokens,
        num_prompt_tokens=12,
    )
    legacy_load = LoadSpec(0, load_tokens, can_load) if load_tokens is not None else None
    legacy = ReqMeta.from_request_tracker(
        legacy_tracker,
        4,
        load_spec=legacy_load,
        block_hashes=hashes,
        discard_partial_chunks=discard_partial,
        hash_block_size=4,
    )
    planner = TransferPlanner(
        TransferPlanningSpec(4, 4, (0,), discard_partial),
        FakeAvailabilityProbe(None),
        ScheduledLoadPublication(),
        store_enabled=True,
        save_decode_cache=False,
    )
    snapshot = RequestSnapshot(
        "request",
        target_tokens,
        ((1, 2, 3),),
        tuple(hashes),
        12,
        saved_tokens,
    )
    candidate = LoadCandidate(TokenRange(0, load_tokens), load_tokens) if load_tokens is not None and can_load else None
    load_command, store_command = planner._plan_commands(snapshot, candidate)

    if legacy is None or (not legacy.can_save and legacy.load_spec is None):
        assert load_command is store_command is None
    elif legacy.load_spec is not None:
        assert load_command is not None and store_command is None
        assert load_command.load_range == TokenRange(
            legacy.load_spec.vllm_cached_tokens,
            legacy.load_spec.kvpool_cached_tokens,
        )
    else:
        assert store_command is not None and load_command is None
        assert store_command.store_range == TokenRange(legacy.save_start_token, legacy.token_len_chunk)


def test_planner_async_load_is_published_after_allocation() -> None:
    planner = make_planner(RemoteAvailability(TokenRange(4, 8), 8), async_load=True)
    request = SimpleNamespace(request_id="request", prompt_token_ids=[0] * 8, block_hashes=[b"a", b"b"])
    assert planner.lookup(LookupQuery("request", 8, 9, request.block_hashes, 4)).load_is_deferred
    planner.confirm_allocation(request, ([1, 2],), 4)
    step = planner.build_step(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        )
    )
    assert step.load.commands[0].load_range == TokenRange(4, 8)


def test_planner_running_request_publishes_monotonic_store_frontier() -> None:
    planner = make_planner(None)
    request = SimpleNamespace(
        request_id="request",
        num_computed_tokens=0,
        num_prompt_tokens=8,
        prompt_token_ids=[0] * 8,
        block_hashes=[b"a"],
    )
    planner.confirm_allocation(request, ([1],), 0)
    first = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[SimpleNamespace(req_id="request", num_computed_tokens=0, block_ids=([1],))],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        num_scheduled_tokens={"request": 2},
    )
    assert planner.build_step(first).store.commands == ()
    request.num_computed_tokens = 2
    second = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["request"], new_block_ids=[None]),
        num_scheduled_tokens={"request": 2},
    )
    store = planner.build_step(second).store.commands[0]
    assert store.store_range == TokenRange(0, 4)
    assert planner.request_progress["request"].published_store_end_token == 4


def test_planner_rebuilds_progress_from_new_blocks_after_preemption() -> None:
    planner = make_planner(None)
    request = SimpleNamespace(
        request_id="request",
        num_computed_tokens=4,
        num_prompt_tokens=8,
        prompt_token_ids=[0] * 8,
        block_hashes=[b"a", b"b"],
    )
    planner.requests["request"] = request
    planner.request_progress["request"] = RequestSnapshot("request", 4, ((1,),), (b"a",), 8, 4)
    planner.build_step(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids={"request"},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        )
    )
    planner.confirm_allocation(request, ([3, 4],), 0)
    step = planner.build_step(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=["request"], new_block_ids=[([3, 4],)]),
            num_scheduled_tokens={"request": 4},
        )
    )
    assert step.store.commands[0].block_ids_by_group == ((3, 4),)
    assert planner.request_progress["request"].published_store_end_token == 8


def test_planner_finished_request_discards_pending_load_state() -> None:
    planner = make_planner(RemoteAvailability(TokenRange(0, 4), 4))
    planner.lookup(LookupQuery("request", 4, 5, [b"a"], 0))
    planner.build_step(
        SimpleNamespace(
            finished_req_ids={"request"},
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        )
    )
    assert "request" not in planner._pending_loads


def test_request_snapshot_is_immutable_and_advances_every_group() -> None:
    snapshot = RequestSnapshot("request", 4, ((1,), (10,)), (b"a",), 8)
    advanced = snapshot.advance(4, ([2], [11]), [b"a", b"b"])
    assert snapshot.block_ids_by_group == ((1,), (10,))
    assert advanced.block_ids_by_group == ((1, 2), (10, 11))


def test_transfer_step_keeps_load_and_store_commands_separate() -> None:
    load = LoadCommand("load", TokenRange(0, 4), ((1,),), (b"a",))
    store = StoreCommand("store", TokenRange(0, 4), ((2,),), (b"b",), 4)
    step = KVTransferStep(LoadCommandBatch((load,)), StoreCommandBatch((store,)))
    assert step.load.commands == (load,)
    assert step.store.commands == (store,)


def test_connector_translates_vllm_lookup_into_planner_query() -> None:
    queries = []
    instance = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    instance.planner = SimpleNamespace(lookup=lambda query: queries.append(query) or ExternalPrefixPlan(4, True))
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[0] * 8,
        num_tokens=9,
        block_hashes=[b"a", b"b"],
    )
    assert instance.get_num_new_matched_tokens(request, 4) == (4, True)
    assert queries == [LookupQuery("request", 8, 9, request.block_hashes, 4)]


def test_connector_routes_only_step_commands_to_kv_pool_graph(monkeypatch) -> None:
    received = []
    step = KVTransferStep(LoadCommandBatch(), StoreCommandBatch())
    instance = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    instance.graph = SimpleNamespace(load=lambda batch: received.append(batch))
    monkeypatch.setattr(instance, "_get_connector_metadata", lambda: step)
    instance.start_load_kv(SimpleNamespace())
    assert received == [step.load]


def test_backend_adapter_keeps_source_release_unknown_after_native_put_error() -> None:
    class Store:
        def batch_put_from_multi_buffers(self, keys, addresses, sizes, replicate_config):
            raise RuntimeError("native put failed")

    backend = SimpleNamespace(
        store=Store(),
        ensure_initialized=lambda: None,
        _build_replicate_config=lambda: object(),
    )
    result = BackendAdapter("mooncake", backend, SimpleNamespace()).put(["key"], [[100]], [[16]])
    assert not result.succeeded
    assert not result.source_release_confirmed
    assert isinstance(result.error, RuntimeError)


def test_backend_adapter_confirms_source_safety_before_native_handoff() -> None:
    backend = SimpleNamespace(
        store=None,
        ensure_initialized=lambda: (_ for _ in ()).throw(RuntimeError("init failed")),
    )
    result = BackendAdapter("mooncake", backend, SimpleNamespace()).put(["key"], [[100]], [[16]])
    assert not result.succeeded
    assert result.source_release_confirmed
