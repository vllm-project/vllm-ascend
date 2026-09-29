"""Validate AscendStore v1 through its token-space domain model."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
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
    BackendSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector import AscendStoreV1Connector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
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
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program import compiler as compiler_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.compiler import (
    ProgramCompilationError,
    compile_kv_pool_program,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.invocation import (
    LoadCompletion,
    LoadTransfer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.program import KVPoolProgram
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.representation import (
    BindingBatch,
    KVBinding,
    KVChunk,
    KVMemoryGeometry,
    KVMemorySegment,
    KVMemoryView,
    KVRegion,
    PhysicalCoordinate,
    RemoteKVObject,
    RemoteObjectKeyBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.spec.schedule import (
    KVPoolSchedule,
    LoadScheduleKind,
    StoreScheduleKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.spec.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
    KVPoolTopology,
    TPPartitionSpec,
    _resolve_group_layers,
    resolve_consumer_pipeline_partitions,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.chunk import KVChunkProjection
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.memory import (
    BindingProjection,
    ContiguousBindingProjection,
    KVBlockProjection,
    LayerwiseBindingProjection,
    StridedBindingProjection,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.reachability import (
    ChunkAvailability,
    GroupAvailability,
    GroupSelection,
    HybridReachability,
    KVSelection,
    UnitaryReachability,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.remote import (
    RemoteObjectProjection,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.store import (
    BackendExistenceMissingFilter,
    IdentityConsumerProjection,
    IdentityMissingFilter,
    PipelinePartitionConsumerProjection,
    StoreOwnershipProjection,
)
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
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.io import BackendIO, LayerwiseBackendIO
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.resources import KVPoolResources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.runtime import KVPoolRuntime
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.timeline.bulk import (
    AsyncLoadTimeline,
    LoadTimeline,
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
        self.put_result: list[int] | None | Exception = []
        self.session_start_result: list[int] = []
        self.session_copy_result: list[int] = []
        self.session_end_result = 0
        self.store_session_start_result: list[int] = []
        self.store_session_copy_result: list[int] | None | Exception = []
        self.store_session_commit_result: list[int] = []
        self.store_session_revoke_result: list[int] = []
        self.calls = []

    def exists(self, keys):
        self.calls.append(("exists", tuple(keys)))
        return self.presence or [1] * len(keys)

    def get(self, keys, addresses, sizes):
        self.calls.append(("get", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
        return self.get_result

    def put(self, keys, addresses, sizes):
        self.calls.append(("put", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
        if isinstance(self.put_result, Exception):
            raise self.put_result
        return self.put_result or [0] * len(keys)

    def set_device(self):
        self.calls.append(("set_device",))

    def validate_layerwise_support(self):
        self.calls.append(("validate_layerwise_support",))

    def batch_get_start(self, keys):
        self.calls.append(("batch_get_start", tuple(keys)))
        return tuple(self.session_start_result or [0] * len(keys))

    def batch_copy_get(self, keys, addresses, sizes, remote_offsets):
        self.calls.append(
            (
                "batch_copy_get",
                tuple(keys),
                tuple(map(tuple, addresses)),
                tuple(map(tuple, sizes)),
                tuple(map(tuple, remote_offsets)),
            )
        )
        return tuple(self.session_copy_result or [0] * len(keys))

    def batch_get_end(self, keys):
        self.calls.append(("batch_get_end", tuple(keys)))
        return self.session_end_result

    def batch_put_start(self, keys, object_sizes):
        self.calls.append(("batch_put_start", tuple(keys), tuple(object_sizes)))
        return tuple(self.store_session_start_result or [0] * len(keys))

    def batch_copy_put(self, keys, addresses, sizes, remote_offsets):
        self.calls.append(
            (
                "batch_copy_put",
                tuple(keys),
                tuple(map(tuple, addresses)),
                tuple(map(tuple, sizes)),
                tuple(map(tuple, remote_offsets)),
            )
        )
        if isinstance(self.store_session_copy_result, Exception):
            raise self.store_session_copy_result
        return self.store_session_copy_result or [0] * len(keys)

    def batch_commit(self, keys):
        self.calls.append(("batch_commit", tuple(keys)))
        return tuple(self.store_session_commit_result or [0] * len(keys))

    def batch_revoke(self, keys):
        self.calls.append(("batch_revoke", tuple(keys)))
        return tuple(self.store_session_revoke_result or [0] * len(keys))


class FakeEvent:
    def __init__(self) -> None:
        self.recorded = False
        self.synchronized = False

    def record(self) -> None:
        self.recorded = True

    def synchronize(self) -> None:
        self.synchronized = True


class FakeLoadStartGate:
    def __init__(self, opened=True) -> None:
        self._opened = threading.Event()
        if opened:
            self._opened.set()

    def open(self) -> None:
        self._opened.set()

    def cancel(self) -> None:
        self._opened.set()

    def wait(self, timeout) -> bool:
        return self._opened.wait(timeout)


class FakeResources:
    def __init__(self, database, backend, memory_geometry) -> None:
        del database
        self.backend = backend
        self.backend_spec = make_backend_spec(type(backend))
        self.memory_geometry = memory_geometry
        self.registered = False
        self.closed = False

    def bind_kv_caches(self, kv_caches):
        self.registered = bool(kv_caches)
        return self.memory_geometry

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
    physical_layers_by_group=None,
    consumer_pipeline_partitions=None,
) -> KVPoolTopology:
    physical_layers_by_group = physical_layers_by_group or {}
    group_topologies = tuple(
        KVPoolGroupTopology(
            group_id,
            4,
            tuple(
                KVPoolLayerTopology(physical_layer_id, (f"layers.{physical_layer_id}.group.{group_id}",))
                for physical_layer_id in physical_layers_by_group.get(group_id, (group_id,))
            ),
            KeyMetadata("model", 0, 0, 0, group_id),
        )
        for group_id in range(max(groups) + 1)
    )
    partition = TPPartitionSpec(
        tp_mismatch,
        key_rank_count or (2 if tp_mismatch else 1),
        2 if tp_mismatch else 1,
    )
    return KVPoolTopology(
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
        consumer_pipeline_partitions,
    )


def make_backend_spec(
    backend_type=FakeBackend,
    *,
    name="fake",
    backend_module=None,
    supports_layerwise=True,
    requires_exists_before_put=False,
) -> BackendSpec:
    return BackendSpec(
        name,
        backend_type,
        backend_module or SimpleNamespace(),
        supports_layerwise,
        requires_exists_before_put,
    )


def compile_program(
    monkeypatch,
    topology: KVPoolTopology,
    *,
    use_layerwise=False,
    async_load=False,
    store_enabled=False,
    supports_layerwise=True,
    requires_exists_before_put=False,
    kv_cache_groups: tuple[KVCacheGroupSpec, ...] = (),
) -> KVPoolProgram:
    monkeypatch.setattr(compiler_module, "resolve_kv_pool_topology", lambda *_: topology)
    monkeypatch.setattr(
        compiler_module,
        "resolve_backend_spec",
        lambda *_: make_backend_spec(
            supports_layerwise=supports_layerwise,
            requires_exists_before_put=requires_exists_before_put,
        ),
    )
    monkeypatch.setattr(compiler_module, "resolve_dcp_kv_cache_spec", lambda spec, _: spec)
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=64),
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_producer" if store_enabled else "kv_consumer",
            kv_connector_extra_config={
                "backend": "fake",
                "use_layerwise": use_layerwise,
                "load_async": async_load,
            },
        ),
    )
    kv_cache_config = SimpleNamespace(
        transfer_groups=kv_cache_groups,
        prefix_cache_retention_interval=None,
    )
    return compile_kv_pool_program(config, kv_cache_config)


@dataclass(frozen=True)
class ProjectionNodes:
    chunks: KVChunkProjection
    remote_objects: RemoteObjectProjection
    blocks: KVBlockProjection
    bindings: BindingProjection
    store_ownership: StoreOwnershipProjection


def make_projection(
    database,
    topology: KVPoolTopology,
    binding_projection: BindingProjection | None = None,
) -> ProjectionNodes:
    if binding_projection is None:
        binding_projection = (
            StridedBindingProjection(topology)
            if topology.tp_partition.tp_mismatch
            else ContiguousBindingProjection(topology)
        )
    return ProjectionNodes(
        KVChunkProjection(database, topology),
        RemoteObjectProjection(topology),
        KVBlockProjection(topology),
        binding_projection,
        StoreOwnershipProjection(topology),
    )


def compile_projection(nodes: ProjectionNodes, memory_geometry: KVMemoryGeometry) -> None:
    nodes.bindings.bind_memory(memory_geometry)


def project_remote_objects(nodes: ProjectionNodes, selection: KVSelection):
    _, object_keys = nodes.chunks.project(selection)
    return nodes.remote_objects.project(object_keys)


def project_bindings(
    nodes: ProjectionNodes,
    selection: KVSelection,
    block_ids_by_group: tuple[tuple[int, ...], ...],
    *,
    owned: bool = False,
) -> tuple[BindingBatch, ...]:
    chunks, object_keys = nodes.chunks.project(selection)
    block_assignments = nodes.blocks.project(chunks, block_ids_by_group)
    if owned:
        block_assignments = nodes.store_ownership.project(block_assignments)
    return tuple(
        nodes.bindings.project(assignments, keys)
        for assignments, keys in zip(block_assignments, object_keys, strict=True)
    )


def make_memory_geometry(database, topology: KVPoolTopology) -> KVMemoryGeometry:
    segments_by_group = {}
    groups_by_id = {group.group_id: group for group in topology.groups}
    for group_id, addresses in database.group_kv_caches_base_addr.items():
        group = groups_by_id[group_id]
        block_lengths = database.group_block_len[group_id]
        block_strides = database.group_block_stride[group_id]
        if len(addresses) % len(group.layers):
            raise ValueError("Test memory segments must divide evenly across physical layers")
        segments_per_layer = len(addresses) // len(group.layers)
        segments = []
        for index, (address, block_length, block_stride) in enumerate(
            zip(addresses, block_lengths, block_strides, strict=True)
        ):
            layer = group.layers[index // segments_per_layer]
            segments.append(
                KVMemorySegment(
                    layer.layer_names[0],
                    layer.physical_layer_id,
                    address,
                    block_length,
                    block_stride,
                    block_length // group.block_size,
                )
            )
        segments_by_group[group_id] = tuple(segments)
    return segments_by_group


def make_binding_batch(group_id=0, *, coordinate=None) -> BindingBatch:
    coordinate = coordinate or PhysicalCoordinate()
    chunk = KVChunk(group_id, 0, TokenRange(0, 4), b"a")
    region = KVRegion(chunk, (0,))
    remote = RemoteKVObject(chunk, "key", coordinate)
    binding = KVBinding(region, remote, 16, (0,), KVMemoryView(1, (100,), (16,)))
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


def test_group_topology_groups_cache_entries_by_physical_layer() -> None:
    layers = _resolve_group_layers(
        ["model.layers.1.v", "mtp.layers.0.attn", "model.layers.1.k"],
        base_layer_count=4,
    )
    assert layers == (
        KVPoolLayerTopology(1, ("model.layers.1.k", "model.layers.1.v")),
        KVPoolLayerTopology(4, ("mtp.layers.0.attn",)),
    )


def test_resources_bind_memory_geometry_once() -> None:
    registered_buffers = []
    backend = SimpleNamespace(
        register_buffer=lambda addresses, sizes: registered_buffers.append((addresses, sizes)),
    )
    group = KVPoolGroupTopology(
        0,
        4,
        (
            KVPoolLayerTopology(0, ("layers.0",)),
            KVPoolLayerTopology(1, ("layers.1",)),
        ),
        KeyMetadata("model", 0, 0, 0, 0),
    )
    resources = KVPoolResources(backend, make_backend_spec(type(backend)), 2, (group,))
    kv_caches = {
        "layers.0": torch.zeros((2, 1)),
        "layers.1": torch.zeros((2, 1)),
    }
    memory_geometry = resources.bind_kv_caches(kv_caches)
    assert [segment.physical_layer_id for segment in memory_geometry[0]] == [0, 1]
    assert len(registered_buffers) == 1
    assert len(registered_buffers[0][0]) == 2
    with pytest.raises(RuntimeError, match="already bound"):
        resources.bind_kv_caches(kv_caches)


def test_program_rejects_layerwise_backend_before_resource_binding(monkeypatch) -> None:
    with pytest.raises(ProgramCompilationError, match="requires a block-key Backend"):
        compile_program(monkeypatch, make_topology(), use_layerwise=True, supports_layerwise=False)


def make_kv_pool_runtime(
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
    topology = make_topology(groups=groups, tp_mismatch=tp_mismatch, key_rank_count=key_rank_count)
    memory_geometry = make_memory_geometry(database, topology)
    resources = FakeResources(database, backend, memory_geometry)
    projection = make_projection(database, topology)
    consumer_projection = IdentityConsumerProjection()
    missing_filter = BackendExistenceMissingFilter() if backend.requires_exists_before_put else IdentityMissingFilter()
    program = KVPoolProgram(
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
        projection.chunks,
        projection.remote_objects,
        projection.blocks,
        projection.bindings,
        projection.store_ownership,
        consumer_projection,
        missing_filter,
        "fake",
        KVPoolSchedule(
            LoadScheduleKind.ASYNC if async_load else LoadScheduleKind.SYNC,
            StoreScheduleKind.ASYNC if store else None,
            2,
        ),
    )
    runtime = KVPoolRuntime(program, resources)
    if registered:
        runtime.bind_kv_caches({"cache": object()})
    else:
        program.bind_memory(memory_geometry)
    return runtime, resources


def make_layerwise_load_runtime(
    backend, physical_layers=(0, 1), groups=(0,), prefetch_layers=2, start_gate_factory=None
):
    database = FakeDatabase({group_id: 4 for group_id in groups})
    for group_id in groups:
        database.group_kv_caches_base_addr[group_id] = [
            1000 + group_id * 10000 + 1000 * layer_id for layer_id in physical_layers
        ]
        database.group_block_len[group_id] = [32] * len(physical_layers)
        database.group_block_stride[group_id] = [64] * len(physical_layers)
    topology = make_topology(groups=groups, physical_layers_by_group=dict.fromkeys(groups, physical_layers))
    memory_geometry = make_memory_geometry(database, topology)
    resources = FakeResources(database, backend, memory_geometry)
    projection = make_projection(database, topology, LayerwiseBindingProjection(topology))
    reachability = (
        UnitaryReachability(groups[0], 64, 4)
        if len(groups) == 1
        else SimpleNamespace(
            group_ids=groups,
            select_for_load=lambda hashes, token_range: KVSelection(
                token_range,
                tuple(hashes),
                tuple(GroupSelection(group_id, None) for group_id in groups),
            ),
        )
    )
    program = KVPoolProgram(
        topology,
        reachability,
        projection.chunks,
        projection.remote_objects,
        projection.blocks,
        projection.bindings,
        projection.store_ownership,
        IdentityConsumerProjection(),
        IdentityMissingFilter(),
        "fake",
        KVPoolSchedule(LoadScheduleKind.LAYERWISE, None, prefetch_layers),
    )
    start_gate_factory = start_gate_factory or FakeLoadStartGate
    runtime = KVPoolRuntime(program, resources, start_gate_factory=start_gate_factory)
    runtime.bind_kv_caches({"cache": object()})
    return runtime, resources


def make_layerwise_store_runtime(backend):
    database = FakeDatabase()
    database.group_kv_caches_base_addr[0] = [1000, 2000]
    database.group_block_len[0] = [32, 32]
    database.group_block_stride[0] = [64, 64]
    topology = make_topology(physical_layers_by_group={0: (0, 1)})
    memory_geometry = make_memory_geometry(database, topology)
    resources = FakeResources(database, backend, memory_geometry)
    projection = make_projection(database, topology, LayerwiseBindingProjection(topology))
    program = KVPoolProgram(
        topology,
        UnitaryReachability(0, 64, 4),
        projection.chunks,
        projection.remote_objects,
        projection.blocks,
        projection.bindings,
        projection.store_ownership,
        IdentityConsumerProjection(),
        BackendExistenceMissingFilter() if backend.requires_exists_before_put else IdentityMissingFilter(),
        "fake",
        KVPoolSchedule(LoadScheduleKind.LAYERWISE, StoreScheduleKind.LAYERWISE, 2),
    )
    runtime = KVPoolRuntime(program, resources)
    runtime.bind_kv_caches({"cache": object()})
    return runtime, resources


def begin_kv_pool_step(
    runtime: KVPoolRuntime,
    load: LoadCommandBatch | None = None,
    store: StoreCommandBatch | None = None,
) -> None:
    runtime.begin_step(KVTransferStep(load or LoadCommandBatch(), store or StoreCommandBatch()))


@pytest.mark.parametrize(("start_token", "end_token"), [(-1, 0), (4, 3)])
def test_token_range_rejects_invalid_coordinates(start_token, end_token) -> None:
    with pytest.raises(ValueError):
        TokenRange(start_token, end_token)


def test_binding_requires_the_same_chunk_and_aligned_segments() -> None:
    chunk = KVChunk(0, 0, TokenRange(0, 4), b"a")
    other = replace(chunk, block_index=1)
    region = KVRegion(chunk, (0,))
    remote_object = RemoteKVObject(chunk, "key")
    with pytest.raises(ValueError, match="semantic chunk"):
        KVBinding(KVRegion(other, (0,)), remote_object, 16, (0,), KVMemoryView(1, (100,), (16,)))
    with pytest.raises(ValueError, match="align every remote offset"):
        KVBinding(region, remote_object, 16, (0, 8), KVMemoryView(1, (100,), (16,)))


def test_projection_preserves_original_group_identity() -> None:
    database = FakeDatabase({0: 4, 3: 4})
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(3, None),))
    projection = make_projection(database, make_topology(groups=(3,)))
    batch = project_remote_objects(projection, selection)[0]
    assert batch.group_id == 3
    assert batch.remote_objects[0].chunk.group_id == 3
    assert "@group:3@" in batch.remote_objects[0].key


def test_chunk_projection_splits_semantic_chunks_from_remote_object_keys() -> None:
    projection = make_projection(FakeDatabase(), make_topology())
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))

    chunk_batches, object_key_batches = projection.chunks.project(selection)
    assignments = projection.blocks.project(chunk_batches, ((1,),))

    assert assignments[0].assignments[0].chunk is chunk_batches[0].chunks[0]
    assert object_key_batches[0].keys[0].chunk is chunk_batches[0].chunks[0]
    assert "@group:0@" in object_key_batches[0].keys[0].base_key


def test_graph_compiler_selects_fixed_graph_rules(monkeypatch) -> None:
    topology = make_topology()
    cases = (
        (topology, {}, ContiguousBindingProjection, IdentityConsumerProjection, IdentityMissingFilter),
        (
            make_topology(tp_mismatch=True),
            {},
            StridedBindingProjection,
            IdentityConsumerProjection,
            IdentityMissingFilter,
        ),
        (
            topology,
            {"use_layerwise": True},
            LayerwiseBindingProjection,
            IdentityConsumerProjection,
            IdentityMissingFilter,
        ),
        (
            topology,
            {"requires_exists_before_put": True},
            ContiguousBindingProjection,
            IdentityConsumerProjection,
            BackendExistenceMissingFilter,
        ),
        (
            make_topology(consumer_pipeline_partitions=(1, 1)),
            {},
            ContiguousBindingProjection,
            PipelinePartitionConsumerProjection,
            IdentityMissingFilter,
        ),
    )
    for topology, options, binding_type, consumer_type, missing_filter_type in cases:
        program = compile_program(monkeypatch, topology, **options)
        assert isinstance(program._reachability, UnitaryReachability)
        assert isinstance(program._binding_projection, binding_type)
        assert isinstance(program._consumer_projection, consumer_type)
        assert isinstance(program._missing_filter, missing_filter_type)


def test_program_compiler_selects_hybrid_reachability(monkeypatch) -> None:
    groups = tuple(
        KVCacheGroupSpec(
            [f"layers.{group_id}"],
            FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32),
        )
        for group_id in range(2)
    )
    topology = make_topology(groups=(0, 1))
    program = compile_program(monkeypatch, topology, kv_cache_groups=groups)
    assert isinstance(program._reachability, HybridReachability)


def test_program_compiler_freezes_backend_identity_and_schedule(monkeypatch) -> None:
    program = compile_program(
        monkeypatch,
        make_topology(),
        async_load=True,
        store_enabled=True,
        requires_exists_before_put=True,
    )
    assert program.backend_name == "fake"
    assert program.schedule.load_kind is LoadScheduleKind.ASYNC
    assert program.schedule.store_kind is StoreScheduleKind.ASYNC


@pytest.mark.parametrize(
    ("topology", "use_layerwise", "message"),
    [
        (
            make_topology(tp_mismatch=True, consumer_pipeline_partitions=(1, 1)),
            False,
            "Consumer pipeline projection cannot yet be composed with TP mismatch",
        ),
        (
            make_topology(tp_mismatch=True),
            True,
            "Layerwise binding projection cannot yet be composed with TP mismatch",
        ),
        (
            make_topology(consumer_pipeline_partitions=(1, 1)),
            True,
            "Layerwise binding projection cannot yet be composed with consumer pipeline projection",
        ),
    ],
)
def test_program_compiler_rejects_conflicting_rules(monkeypatch, topology, use_layerwise, message) -> None:
    with pytest.raises(ProgramCompilationError, match=message):
        compile_program(monkeypatch, topology, use_layerwise=use_layerwise)


def test_binding_projection_requires_a_remote_key_for_every_local_assignment() -> None:
    database = FakeDatabase()
    topology = make_topology()
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    chunk_batches, _ = projection.chunks.project(selection)
    assignments = projection.blocks.project(chunk_batches, ((1,),))[0]

    with pytest.raises(ValueError, match="no remote object key"):
        projection.bindings.project(assignments, RemoteObjectKeyBatch(0, ()))


def test_projection_reuses_every_lookup_coordinate() -> None:
    database = FakeDatabase()
    topology = replace(make_topology(), pp_size=2, dcp_size=2, tp_partition=TPPartitionSpec(False, 2, 1))
    projection = make_projection(database, topology)
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    first = project_remote_objects(projection, selection)[0]
    second = project_remote_objects(projection, selection)[0]
    assert len(first.remote_objects) == 8
    assert len({remote.coordinate for remote in first.remote_objects}) == 8
    assert all(left.coordinate is right.coordinate for left, right in zip(first.remote_objects, second.remote_objects))


def test_projection_keeps_lookup_rank_major_key_order() -> None:
    database = FakeDatabase()
    topology = replace(make_topology(), pp_size=2, dcp_size=2, tp_partition=TPPartitionSpec(False, 2, 1))
    projection = make_projection(database, topology)
    selection = KVSelection(TokenRange(0, 8), (b"a", b"b"), (GroupSelection(0, None),))
    remote_objects = project_remote_objects(projection, selection)[0].remote_objects
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
        groups=(replace(topology.groups[0], key_metadata=metadata, uses_align_state=True),),
    )
    projection = make_projection(database, topology)
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    remote_object = project_remote_objects(projection, selection)[0].remote_objects[0]
    assert "@head_or_tp_rank:0@" in remote_object.key


def test_binding_requires_registered_memory_mapping() -> None:
    projection = make_projection(FakeDatabase(), make_topology())
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    with pytest.raises(RuntimeError, match="before cache registration"):
        project_bindings(projection, selection, ((1,),))


def test_registered_memory_geometry_is_bound_once() -> None:
    database = FakeDatabase()
    topology = make_topology()
    memory_geometry = make_memory_geometry(database, topology)
    projection = make_projection(database, topology)
    compile_projection(projection, memory_geometry)
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    database.group_kv_caches_base_addr[0][0] = 5000
    binding = project_bindings(projection, selection, ((1,),))[0].bindings[0]

    assert binding.memory.addresses == (1032,)
    with pytest.raises(RuntimeError, match="already bound"):
        compile_projection(projection, memory_geometry)


def test_strided_mapping_constructs_matching_edges_atomically() -> None:
    database = FakeDatabase()
    topology = make_topology(tp_mismatch=True, tp_rank=1)
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    batch = project_bindings(projection, selection, ((3,),))[0]
    assert [binding.remote_object.coordinate.effective_tp_rank for binding in batch.bindings] == [2, 3]
    assert "@head_or_tp_rank:2@" in batch.bindings[0].remote_object.key


def test_strided_mapping_compiles_each_memory_segment_geometry() -> None:
    database = FakeDatabase()
    database.group_kv_caches_base_addr[0] = [1000, 2000]
    database.group_block_len[0] = [32, 64]
    database.group_block_stride[0] = [32, 64]
    topology = make_topology(tp_mismatch=True)
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    first_representation = project_bindings(projection, selection, ((1,),))[0].bindings[0]
    assert first_representation.memory.sizes == (4, 4, 4, 4, 8, 8, 8, 8)


def test_binding_rejects_misaligned_memory_segments() -> None:
    database = FakeDatabase()
    database.group_kv_caches_base_addr[0] = [1000, 2000]
    database.group_block_len[0] = [32, 32]
    database.group_block_stride[0] = [32, 32]
    topology = make_topology(physical_layers_by_group={0: (0, 1)})
    memory_geometry = make_memory_geometry(database, topology)
    projection = make_projection(database, topology)
    compile_projection(projection, memory_geometry)
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    batches = project_bindings(projection, selection, ((1,),), owned=True)
    binding = batches[0].bindings[0]
    with pytest.raises(ValueError, match="align every remote offset"):
        replace(
            binding,
            memory=KVMemoryView(binding.memory.block_id, (binding.memory.addresses[0],), (binding.memory.sizes[0],)),
        )


def test_consumer_pipeline_projection_uses_compiled_layer_segments() -> None:
    database = FakeDatabase()
    database.group_kv_caches_base_addr[0] = [1000, 2000]
    database.group_block_len[0] = [32, 32]
    database.group_block_stride[0] = [32, 32]
    topology = make_topology(physical_layers_by_group={0: (0, 1)})
    memory_geometry = make_memory_geometry(database, topology)
    consumer_projection = PipelinePartitionConsumerProjection((1, 1))
    consumer_projection.bind_memory(memory_geometry)
    projection = make_projection(database, topology)
    compile_projection(projection, memory_geometry)
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    batches = project_bindings(projection, selection, ((1,),), owned=True)
    projected = consumer_projection.project(batches)[0].bindings
    assert [binding.memory.addresses for binding in projected] == [(1032,), (2032,)]
    assert [binding.remote_object.coordinate.consumer_pp_slice for binding in projected] == [0, 1]
    assert [binding.region.physical_layer_ids for binding in projected] == [(0,), (1,)]
    assert "@pp_rank:1@" in projected[1].remote_object.key


def test_layerwise_projection_splits_regions_inside_one_remote_object() -> None:
    database = FakeDatabase()
    topology = make_topology(physical_layers_by_group={0: (0, 1)})
    projection = make_projection(database, topology, LayerwiseBindingProjection(topology))
    memory_geometry: KVMemoryGeometry = {
        0: (
            KVMemorySegment("layers.0.k", 0, 1000, 32, 64, 8),
            KVMemorySegment("layers.0.v", 0, 2000, 32, 64, 8),
            KVMemorySegment("layers.1.kv", 1, 3000, 64, 128, 16),
        )
    }
    compile_projection(projection, memory_geometry)
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    bindings = project_bindings(projection, selection, ((1,),))[0].bindings

    assert [binding.region.physical_layer_ids for binding in bindings] == [(0,), (1,)]
    assert bindings[0].remote_object is bindings[1].remote_object
    assert [binding.remote_offsets for binding in bindings] == [(0, 32), (64,)]
    assert [binding.memory.sizes for binding in bindings] == [(32, 32), (64,)]
    assert [binding.memory.addresses for binding in bindings] == [(1064, 2064), (3128,)]


def test_store_ownership_precedes_physical_fanout() -> None:
    database = FakeDatabase()
    topology = make_topology(tp_mismatch=True, pcp_rank=1, pcp_size=2)
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    selection = KVSelection(TokenRange(0, 8), (b"a", b"b"), (GroupSelection(0, None),))
    batch = project_bindings(projection, selection, ((1, 2),), owned=True)[0]
    assert {binding.memory.block_id for binding in batch.bindings} == {2}
    assert len(batch.bindings) == 2


def test_strided_store_ownership_does_not_drop_chunks_as_tp_replicas() -> None:
    database = FakeDatabase()
    topology = replace(make_topology(tp_mismatch=True), put_step=2)
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    selection = KVSelection(TokenRange(0, 8), (b"a", b"b"), (GroupSelection(0, None),))
    batch = project_bindings(projection, selection, ((1, 2),), owned=True)[0]
    assert {binding.memory.block_id for binding in batch.bindings} == {1, 2}
    assert len(batch.bindings) == 4


def test_strided_mapping_keeps_align_state_null_blocks() -> None:
    database = FakeDatabase()
    topology = make_topology(tp_mismatch=True)
    topology = replace(topology, groups=(replace(topology.groups[0], uses_align_state=True),))
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    batch = project_bindings(projection, selection, ((0,),))[0]
    assert len(batch.bindings) == 2
    assert all(binding.memory.block_id == 0 for binding in batch.bindings)


def test_identity_consumer_projection_preserves_strided_bindings() -> None:
    database = FakeDatabase()
    topology = make_topology(tp_mismatch=True)
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    batches = project_bindings(projection, selection, ((1,),), owned=True)
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


def test_kv_pool_runtime_lookup_requires_every_physical_representation() -> None:
    backend = FakeBackend()
    backend.presence = [1, 0]
    runtime, _ = make_kv_pool_runtime(backend, store=False, key_rank_count=2)
    result = runtime.lookup(LookupRequest(TokenRange(0, 4), (0,), (b"a",)))
    assert result == LookupResult(0)


def test_kv_pool_runtime_lookup_contains_backend_protocol_failure_as_a_miss() -> None:
    backend = FakeBackend()
    backend.presence = [1, 1]
    runtime, _ = make_kv_pool_runtime(backend, store=False)
    result = runtime.lookup(LookupRequest(TokenRange(0, 4), (0,), (b"a",)))
    assert result == LookupResult(0)


def test_hybrid_lookup_and_store_share_retention_policy(monkeypatch) -> None:
    recorded = []

    def reachable_mask(_manager, **kwargs):
        recorded.append(kwargs["retention_interval"])
        return None

    import vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.reachability as module

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
    outcomes = BackendIO(backend, make_backend_spec()).load(make_binding_batch().bindings)
    assert outcomes[0].binding == make_binding_batch().bindings[0]
    assert outcomes[0].result_code == 0


@pytest.mark.parametrize("native_result", [None, [0, 0]])
def test_backend_io_marks_misaligned_load_results_unknown(native_result) -> None:
    backend = FakeBackend()
    backend.get_result = native_result
    outcomes = BackendIO(backend, make_backend_spec()).load(make_binding_batch().bindings)
    assert [outcome.result_code for outcome in outcomes] == [None]


def test_synchronous_and_asynchronous_load_share_the_same_operation() -> None:
    backend = FakeBackend()
    backend.get_result = [0]
    backend_io = BackendIO(backend, make_backend_spec())
    binding_batch = make_binding_batch()
    transfer = LoadTransfer("request", binding_batch.bindings)

    def operation(item: LoadTransfer) -> LoadCompletion:
        return LoadCompletion(item.request_id, backend_io.load(item.traversal))

    synchronous = LoadTimeline()
    synchronous.bind_operation(operation)
    synchronous.start()
    assert tuple(synchronous.submit([transfer]))[0].binding_evidence[0].binding == binding_batch.bindings[0]

    asynchronous = AsyncLoadTimeline(backend.set_device)
    asynchronous.bind_operation(operation)
    asynchronous.start()
    asynchronous.submit([transfer])
    asynchronous._executor._queue.join()
    completion = asynchronous.collect()[0]
    asynchronous.close()
    assert completion.binding_evidence[0].binding == binding_batch.bindings[0]


def test_synchronous_load_rejects_start_without_an_operation() -> None:
    with pytest.raises(RuntimeError, match="no bound operation"):
        LoadTimeline().start()


def test_asynchronous_load_failure_terminates_and_drains_pending_requests() -> None:
    backend = FakeBackend()

    def fail_load(*args, **kwargs):
        raise RuntimeError("backend get failed")

    timeline = AsyncLoadTimeline(backend.set_device)
    timeline.bind_operation(fail_load)
    timeline.start()
    binding_batch = make_binding_batch()
    timeline.submit(
        [
            LoadTransfer("first", binding_batch.bindings),
            LoadTransfer("second", binding_batch.bindings),
        ]
    )
    timeline._executor.join(timeout=1)
    assert not timeline._executor.is_alive()
    timeline._executor._queue.join()
    with pytest.raises(RuntimeError, match="unfinished requests: .*first.*second"):
        timeline.collect()
    with pytest.raises(RuntimeError, match="terminated during asynchronous Load"):
        timeline.submit([])
    with pytest.raises(RuntimeError, match="terminated during asynchronous Load"):
        timeline.close()


def test_kv_pool_runtime_filters_remote_objects_before_waiting_for_source(monkeypatch) -> None:
    backend = FakeBackend()
    backend.requires_exists_before_put = True
    backend.presence = [1]
    event = FakeEvent()
    runtime, _ = make_kv_pool_runtime(backend, registered=True)
    monkeypatch.setattr(torch.npu, "Event", lambda: event)
    store = StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    begin_kv_pool_step(runtime, store=store)
    runtime.finish_step()
    result = runtime.fence_previous_store()
    runtime.close()
    assert result[0].evidence.succeeded
    assert not event.synchronized
    assert [call[0] for call in backend.calls] == ["set_device", "exists"]


def test_kv_pool_runtime_keeps_the_configured_missing_filter(monkeypatch) -> None:
    backend = FakeBackend()
    runtime, _ = make_kv_pool_runtime(backend, registered=True)
    backend.requires_exists_before_put = True
    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    store = StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    begin_kv_pool_step(runtime, store=store)
    runtime.finish_step()
    runtime.fence_previous_store()
    runtime.close()
    assert "exists" not in [call[0] for call in backend.calls]


def test_kv_pool_runtime_preserves_unknown_source_release_after_put_failure(monkeypatch) -> None:
    backend = FakeBackend()
    backend.put_result = RuntimeError("put failed")
    runtime, resources = make_kv_pool_runtime(backend, registered=True)
    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    store = StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    begin_kv_pool_step(runtime, store=store)
    runtime.finish_step()
    with pytest.raises(RuntimeError, match="Store failed"):
        runtime.fence_previous_store()
    batch = runtime._pending_store_batch
    assert batch is not None
    result = batch.completions[0]
    assert not result.evidence.source_release_confirmed
    assert isinstance(result.evidence.error, RuntimeError)
    with pytest.raises(RuntimeError, match="previous Store failure"):
        runtime.close()
    assert runtime._timeline.store is not None
    assert not runtime._timeline.store._executor.is_alive()
    assert not resources.closed


def test_kv_pool_runtime_loads_multiple_groups_in_one_backend_call() -> None:
    backend = FakeBackend()
    backend.get_result = [0, 0]
    runtime, _ = make_kv_pool_runtime(backend, FakeDatabase({0: 4, 1: 4}), groups=(0, 1), store=False)
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,), (2,)), (b"a",)),))
    begin_kv_pool_step(runtime, load=load)
    runtime.start_load()
    get_calls = [call for call in backend.calls if call[0] == "get"]
    assert len(get_calls) == 1
    assert len(get_calls[0][1]) == 2


def test_kv_pool_runtime_reports_grouped_load_failure_at_request_scope() -> None:
    backend = FakeBackend()
    backend.get_result = [0, -1]
    runtime, _ = make_kv_pool_runtime(backend, FakeDatabase({0: 4, 1: 4}), groups=(0, 1), store=False)
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,), (2,)), (b"a",)),))
    begin_kv_pool_step(runtime, load=load)
    with pytest.raises(RuntimeError, match="Hybrid KV Load failed"):
        runtime.start_load()


def test_kv_pool_runtime_preserves_nonzero_single_group_failure_identity() -> None:
    backend = FakeBackend()
    backend.get_result = [-1]
    runtime, _ = make_kv_pool_runtime(backend, FakeDatabase({3: 4}), groups=(3,), store=False)
    block_ids = ((), (), (), (7,))
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), block_ids, (b"a",)),))
    begin_kv_pool_step(runtime, load=load)
    runtime.start_load()
    assert runtime.collect_load_result().failed_block_ids == {7}


def test_layerwise_load_advances_one_physical_layer_per_hook() -> None:
    backend = FakeBackend()
    runtime, resources = make_layerwise_load_runtime(backend)
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",)),))
    begin_kv_pool_step(runtime, load=load)

    runtime.start_load()
    runtime.wait_for_layer_load("layers.0.group.0")
    runtime.wait_for_layer_load("layers.1.group.0")
    runtime.close()

    session_calls = [call for call in backend.calls if call[0] in ("batch_get_start", "batch_get_end")]
    range_calls = [call for call in backend.calls if call[0] == "batch_copy_get"]
    assert [call[0] for call in session_calls] == ["batch_get_start", "batch_get_end"]
    assert session_calls[0][1] == session_calls[1][1]
    assert len(session_calls[0][1]) == 1
    assert [call[2] for call in range_calls] == [((1064,),), ((2064,),)]
    assert [call[4] for call in range_calls] == [((0,),), ((32,),)]
    assert resources.closed


def test_layerwise_load_maintains_a_bounded_prefetch_window(monkeypatch) -> None:
    backend = FakeBackend()
    second_layer_loaded = threading.Event()
    gates = []
    batch_copy_get = backend.batch_copy_get

    def observe_load_window(keys, addresses, sizes, remote_offsets):
        result = batch_copy_get(keys, addresses, sizes, remote_offsets)
        if len([call for call in backend.calls if call[0] == "batch_copy_get"]) == 2:
            second_layer_loaded.set()
        return result

    def make_start_gate():
        gate = FakeLoadStartGate(opened=False)
        gates.append(gate)
        return gate

    monkeypatch.setattr(backend, "batch_copy_get", observe_load_window)
    runtime, _ = make_layerwise_load_runtime(
        backend,
        physical_layers=(0, 1, 2),
        prefetch_layers=2,
        start_gate_factory=make_start_gate,
    )
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",)),))
    begin_kv_pool_step(runtime, load=load)

    runtime.start_load()

    assert not any(call[0] == "batch_copy_get" for call in backend.calls)
    runtime.wait_for_layer_load("layers.0.group.0")
    assert len([call for call in backend.calls if call[0] == "batch_copy_get"]) == 1
    gates[0].open()
    assert second_layer_loaded.wait(timeout=2)
    runtime.wait_for_layer_load("layers.1.group.0")
    assert len([call for call in backend.calls if call[0] == "batch_copy_get"]) == 2
    gates[1].open()
    runtime.wait_for_layer_load("layers.2.group.0")
    runtime.close()
    assert len([call for call in backend.calls if call[0] == "batch_copy_get"]) == 3


def test_layerwise_load_surfaces_background_range_failure_and_closes_sessions(monkeypatch) -> None:
    backend = FakeBackend()

    def fail_load_range(*args):
        del args
        raise RuntimeError("range get failed")

    monkeypatch.setattr(backend, "batch_copy_get", fail_load_range)
    runtime, resources = make_layerwise_load_runtime(backend)
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",)),))
    begin_kv_pool_step(runtime, load=load)

    runtime.start_load()

    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        runtime.wait_for_layer_load("layers.0.group.0")
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        runtime.close()
    assert resources.closed


def test_layerwise_load_closes_attempted_sessions_after_start_exception(monkeypatch) -> None:
    backend = FakeBackend()

    def fail_session_start(keys):
        backend.calls.append(("batch_get_start", tuple(keys)))
        raise RuntimeError("session start failed")

    monkeypatch.setattr(backend, "batch_get_start", fail_session_start)
    runtime, resources = make_layerwise_load_runtime(backend)
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",)),))

    begin_kv_pool_step(runtime, load=load)
    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        runtime.start_load()
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        runtime.close()
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    assert resources.closed


def test_layerwise_load_attempts_session_end_only_once_after_failure() -> None:
    backend = FakeBackend()
    backend.session_end_result = -1
    runtime, resources = make_layerwise_load_runtime(backend, physical_layers=(0,))
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",)),))
    begin_kv_pool_step(runtime, load=load)

    runtime.start_load()
    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        runtime.wait_for_layer_load("layers.0.group.0")
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        runtime.close()
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    assert resources.closed


def test_layerwise_hybrid_start_failure_aborts_open_sessions() -> None:
    backend = FakeBackend()
    backend.session_start_result = [0, -1]
    runtime, resources = make_layerwise_load_runtime(backend, groups=(0, 1))
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,), (2,)), (b"a",)),))

    begin_kv_pool_step(runtime, load=load)
    with pytest.raises(RuntimeError, match="Hybrid KV Load failed"):
        runtime.start_load()
    finish_calls = [call for call in backend.calls if call[0] == "batch_get_end"]
    assert len(finish_calls) == 1
    assert len(finish_calls[0][1]) == 1
    runtime.close()
    assert resources.closed


def test_layerwise_hybrid_range_failure_aborts_open_sessions() -> None:
    backend = FakeBackend()
    backend.session_copy_result = [0, -1]
    runtime, resources = make_layerwise_load_runtime(backend, groups=(0, 1), prefetch_layers=1)
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,), (2,)), (b"a",)),))
    begin_kv_pool_step(runtime, load=load)

    runtime.start_load()
    with pytest.raises(RuntimeError, match="Hybrid KV Load failed"):
        runtime.wait_for_layer_load("layers.0.group.0")
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    runtime.close()
    assert resources.closed


def test_layerwise_load_reports_session_start_failure_by_block() -> None:
    backend = FakeBackend()
    backend.session_start_result = [-1]
    runtime, _ = make_layerwise_load_runtime(backend)
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((7,),), (b"a",)),))
    begin_kv_pool_step(runtime, load=load)

    runtime.start_load()

    assert runtime.collect_load_result().failed_block_ids == {7}
    assert not any(call[0] == "batch_copy_get" for call in backend.calls)
    runtime.close()


def test_layerwise_load_closes_an_incomplete_session_before_reporting_it() -> None:
    backend = FakeBackend()
    runtime, _ = make_layerwise_load_runtime(backend)
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",)),))
    begin_kv_pool_step(runtime, load=load)

    runtime.start_load()
    runtime.wait_for_layer_load("layers.0.group.0")

    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        runtime.collect_load_result()
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        runtime.close()


def test_layerwise_store_publishes_ranges_before_committing_shared_objects(monkeypatch) -> None:
    backend = FakeBackend()
    events = []

    def make_event():
        event = FakeEvent()
        events.append(event)
        return event

    monkeypatch.setattr(torch.npu, "Event", make_event)
    runtime, resources = make_layerwise_store_runtime(backend)
    store = StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    begin_kv_pool_step(runtime, store=store)

    runtime.save_layer("layers.0.group.0")
    runtime.save_layer("layers.1.group.0")
    runtime.finish_step()
    runtime.close()

    store_calls = [call for call in backend.calls if call[0] in ("batch_put_start", "batch_copy_put", "batch_commit")]
    assert [call[0] for call in store_calls] == [
        "batch_put_start",
        "batch_copy_put",
        "batch_copy_put",
        "batch_commit",
    ]
    assert store_calls[0][2] == (64,)
    assert [call[4] for call in store_calls[1:3]] == [((0,),), ((32,),)]
    assert all(event.synchronized for event in events)
    assert runtime._pending_store_batch is None
    assert resources.closed


def test_layerwise_store_runs_session_lifecycle_on_its_executor(monkeypatch) -> None:
    backend = FakeBackend()
    backend.requires_exists_before_put = True
    backend.presence = [0]
    executor_threads = set()
    observe_presence = backend.exists
    batch_put_start = backend.batch_put_start
    batch_copy_put = backend.batch_copy_put
    batch_commit = backend.batch_commit

    def track_presence(keys):
        executor_threads.add(threading.current_thread().name)
        return observe_presence(keys)

    def track_start(keys, object_sizes):
        executor_threads.add(threading.current_thread().name)
        return batch_put_start(keys, object_sizes)

    def track_range(keys, addresses, sizes, remote_offsets):
        executor_threads.add(threading.current_thread().name)
        return batch_copy_put(keys, addresses, sizes, remote_offsets)

    def track_commit(keys):
        executor_threads.add(threading.current_thread().name)
        return batch_commit(keys)

    monkeypatch.setattr(backend, "exists", track_presence)
    monkeypatch.setattr(backend, "batch_put_start", track_start)
    monkeypatch.setattr(backend, "batch_copy_put", track_range)
    monkeypatch.setattr(backend, "batch_commit", track_commit)
    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    runtime, _ = make_layerwise_store_runtime(backend)
    store = StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    begin_kv_pool_step(runtime, store=store)

    runtime.save_layer("layers.0.group.0")
    runtime.save_layer("layers.1.group.0")
    runtime.finish_step()
    runtime.close()

    assert executor_threads == {"KVPoolLayerwiseStoreExecutor"}


def test_layerwise_store_revokes_attempted_sessions_after_start_exception(monkeypatch) -> None:
    backend = FakeBackend()

    def fail_session_start(keys, object_sizes):
        backend.calls.append(("batch_put_start", tuple(keys), tuple(object_sizes)))
        raise RuntimeError("session start failed")

    monkeypatch.setattr(backend, "batch_put_start", fail_session_start)
    runtime, resources = make_layerwise_store_runtime(backend)
    store = StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    begin_kv_pool_step(runtime, store=store)

    assert [call[0] for call in backend.calls].count("batch_revoke") == 1
    with pytest.raises(RuntimeError, match="Store failed"):
        runtime.finish_step()
    with pytest.raises(RuntimeError, match="previous Store failure"):
        runtime.close()
    assert [call[0] for call in backend.calls].count("batch_revoke") == 1
    assert resources.closed


def test_layerwise_store_revokes_attempted_sessions_after_mismatched_start_results() -> None:
    backend = FakeBackend()
    backend.store_session_start_result = [0, 0]
    runtime, resources = make_layerwise_store_runtime(backend)
    store = StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    begin_kv_pool_step(runtime, store=store)

    assert [call[0] for call in backend.calls].count("batch_revoke") == 1
    with pytest.raises(RuntimeError, match="Store failed"):
        runtime.finish_step()
    with pytest.raises(RuntimeError, match="previous Store failure"):
        runtime.close()
    assert resources.closed


def test_layerwise_store_preserves_unknown_source_release_after_range_failure(monkeypatch) -> None:
    backend = FakeBackend()
    backend.store_session_copy_result = RuntimeError("put failed")
    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    runtime, resources = make_layerwise_store_runtime(backend)
    store = StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    begin_kv_pool_step(runtime, store=store)

    runtime.save_layer("layers.0.group.0")
    runtime.save_layer("layers.1.group.0")
    with pytest.raises(RuntimeError, match="Store failed"):
        runtime.finish_step()

    assert runtime._pending_store_batch is not None
    assert not runtime._pending_store_batch.completions[0].evidence.source_release_confirmed
    assert "batch_commit" not in [call[0] for call in backend.calls]
    assert "batch_revoke" in [call[0] for call in backend.calls]
    with pytest.raises(RuntimeError, match="previous Store failure"):
        runtime.close()
    assert not resources.closed


def test_layerwise_store_does_not_open_sessions_for_existing_objects(monkeypatch) -> None:
    backend = FakeBackend()
    backend.requires_exists_before_put = True
    backend.presence = [1]
    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    runtime, _ = make_layerwise_store_runtime(backend)
    store = StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    begin_kv_pool_step(runtime, store=store)

    runtime.save_layer("layers.0.group.0")
    runtime.save_layer("layers.1.group.0")
    runtime.finish_step()
    runtime.close()

    assert runtime._pending_store_batch is None
    assert [call[0] for call in backend.calls].count("exists") == 1
    assert not any(call[0] in ("batch_put_start", "batch_copy_put", "batch_commit") for call in backend.calls)


def test_layerwise_store_revokes_objects_when_a_selected_layer_is_not_reached(monkeypatch) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    runtime, resources = make_layerwise_store_runtime(backend)
    store = StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    begin_kv_pool_step(runtime, store=store)

    runtime.save_layer("layers.0.group.0")
    with pytest.raises(RuntimeError, match="Store failed"):
        runtime.finish_step()

    assert "batch_commit" not in [call[0] for call in backend.calls]
    assert "batch_revoke" in [call[0] for call in backend.calls]
    with pytest.raises(RuntimeError, match="previous Store failure"):
        runtime.close()
    assert resources.closed


def test_kv_pool_runtime_async_load_publishes_only_after_backend_completion() -> None:
    backend = FakeBackend()
    backend.get_result = [0]
    runtime, _ = make_kv_pool_runtime(backend, async_load=True, store=False, registered=True)
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",)),))
    begin_kv_pool_step(runtime, load=load)
    runtime.start_load()
    runtime._timeline.load._executor._queue.join()
    assert runtime.collect_load_result().completed_request_ids == {"request"}
    runtime.close()


def test_kv_pool_runtime_owns_active_step_lifecycle() -> None:
    runtime, _ = make_kv_pool_runtime(FakeBackend(), store=False)
    step = KVTransferStep()

    with pytest.raises(RuntimeError, match="has not begun"):
        runtime.start_load()
    runtime.begin_step(step)
    with pytest.raises(RuntimeError, match="has not ended"):
        runtime.begin_step(step)
    runtime.end_step()
    with pytest.raises(RuntimeError, match="has not begun"):
        runtime.end_step()


def test_kv_pool_runtime_retains_async_load_owner_across_steps() -> None:
    backend = FakeBackend()
    backend.get_result = [0]
    runtime, _ = make_kv_pool_runtime(backend, async_load=True, store=False, registered=True)
    load = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",)),))

    begin_kv_pool_step(runtime, load=load)
    runtime.start_load()
    runtime.end_step()
    begin_kv_pool_step(runtime)
    runtime._timeline.load._executor._queue.join()

    assert runtime.collect_load_result().completed_request_ids == {"request"}
    runtime.close()


def test_kv_pool_runtime_store_projects_then_fences_one_step(monkeypatch) -> None:
    backend = FakeBackend()
    backend.put_result = [0]
    runtime, resources = make_kv_pool_runtime(backend, registered=True)
    event = FakeEvent()
    monkeypatch.setattr(torch.npu, "Event", lambda: event)
    store = StoreCommandBatch((StoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    begin_kv_pool_step(runtime, store=store)
    runtime.finish_step()
    results = runtime.fence_previous_store()
    runtime.close()
    assert results[0].evidence.succeeded
    assert event.synchronized
    assert [call[0] for call in backend.calls].count("put") == 1
    assert resources.closed


def test_kv_pool_runtime_close_keeps_resources_when_store_source_release_is_unknown() -> None:
    backend = FakeBackend()
    runtime, resources = make_kv_pool_runtime(backend, store=True)
    runtime._pending_store_batch = SimpleNamespace()
    runtime._store_error = RuntimeError("unknown source release")
    with pytest.raises(RuntimeError):
        runtime.close()
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


def test_chunk_continuation_keeps_layerwise_load_session_step_local() -> None:
    backend = FakeBackend()
    runtime, _ = make_layerwise_load_runtime(backend)
    planner = make_planner(RemoteAvailability(TokenRange(0, 4), 4))
    request = SimpleNamespace(
        request_id="request",
        num_computed_tokens=4,
        num_prompt_tokens=12,
        prompt_token_ids=[0] * 12,
        block_hashes=[b"a", b"b"],
    )
    planner.lookup(LookupQuery("request", 12, 12, request.block_hashes, 0))
    planner.confirm_allocation(request, ([1],), 4)
    first_step = planner.build_step(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[SimpleNamespace(req_id="request", num_computed_tokens=4, block_ids=([1, 2],))],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
            num_scheduled_tokens={"request": 4},
        )
    )
    runtime.begin_step(first_step)
    runtime.start_load()
    runtime.wait_for_layer_load("layers.0.group.0")
    runtime.wait_for_layer_load("layers.1.group.0")
    runtime.end_step()

    request.num_computed_tokens = 8
    request.block_hashes = [b"a", b"b", b"c"]
    continued_step = planner.build_step(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=["request"], new_block_ids=[([3],)]),
            num_scheduled_tokens={"request": 4},
        )
    )
    runtime.begin_step(continued_step)
    runtime.start_load()
    runtime.close()

    assert continued_step.load.commands == ()
    assert planner.request_progress["request"].block_ids_by_group == ((1, 2, 3),)
    assert [call[0] for call in backend.calls].count("batch_get_start") == 1
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1


def test_preemption_reloads_prefix_into_replacement_blocks() -> None:
    backend = FakeBackend()
    runtime, _ = make_layerwise_load_runtime(backend)
    planner = make_planner(RemoteAvailability(TokenRange(0, 4), 4))
    request = SimpleNamespace(
        request_id="request",
        num_computed_tokens=0,
        num_prompt_tokens=8,
        prompt_token_ids=[0] * 8,
        block_hashes=[b"a", b"b"],
    )
    planner.lookup(LookupQuery("request", 8, 8, request.block_hashes, 0))
    planner.confirm_allocation(request, ([1],), 4)
    first_step = planner.build_step(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[SimpleNamespace(req_id="request", num_computed_tokens=4, block_ids=([1],))],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
            num_scheduled_tokens={"request": 4},
        )
    )
    runtime.begin_step(first_step)
    runtime.start_load()
    runtime.wait_for_layer_load("layers.0.group.0")
    runtime.wait_for_layer_load("layers.1.group.0")
    runtime.end_step()

    planner.build_step(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids={"request"},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        )
    )
    planner.lookup(LookupQuery("request", 8, 8, request.block_hashes, 0))
    planner.confirm_allocation(request, ([7],), 4)
    resumed_step = planner.build_step(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=["request"], new_block_ids=[([7],)]),
            num_scheduled_tokens={"request": 4},
        )
    )
    runtime.begin_step(resumed_step)
    runtime.start_load()
    runtime.wait_for_layer_load("layers.0.group.0")
    runtime.wait_for_layer_load("layers.1.group.0")
    runtime.close()

    assert resumed_step.load.commands[0].block_ids_by_group == ((7,),)
    assert resumed_step.store.commands == ()
    assert planner.request_progress["request"].block_ids_by_group == ((7,),)
    assert [call[0] for call in backend.calls].count("batch_get_start") == 2
    assert [call[0] for call in backend.calls].count("batch_get_end") == 2


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


def test_connector_routes_only_step_commands_to_kv_pool_runtime() -> None:
    received = []
    step = KVTransferStep(LoadCommandBatch(), StoreCommandBatch())
    instance = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    instance.runtime = SimpleNamespace(
        begin_step=lambda value: received.append(("begin", value)),
        end_step=lambda: received.append(("end",)),
        start_load=lambda: received.append(("load",)),
        wait_for_layer_load=lambda layer_name: received.append(("layer", layer_name)),
    )
    instance.bind_connector_metadata(step)
    instance.start_load_kv(SimpleNamespace())
    instance.wait_for_layer_load("layers.0")
    instance.clear_connector_metadata()
    assert received == [("begin", step), ("load",), ("layer", "layers.0"), ("end",)]


def test_backend_io_keeps_source_release_unknown_after_native_put_error() -> None:
    class Store:
        def batch_put_from_multi_buffers(self, keys, addresses, sizes, replicate_config):
            raise RuntimeError("native put failed")

    backend = SimpleNamespace(
        store=Store(),
        ensure_initialized=lambda: None,
        _build_replicate_config=lambda: object(),
    )
    backend_spec = make_backend_spec(type(backend), name="mooncake")
    result = BackendIO(backend, backend_spec).store((make_binding_batch(),))
    assert not result.succeeded
    assert not result.source_release_confirmed
    assert isinstance(result.error, RuntimeError)


def test_backend_io_confirms_source_safety_before_native_handoff() -> None:
    backend = SimpleNamespace(
        store=None,
        ensure_initialized=lambda: (_ for _ in ()).throw(RuntimeError("init failed")),
    )
    backend_spec = make_backend_spec(type(backend), name="mooncake")
    result = BackendIO(backend, backend_spec).store((make_binding_batch(),))
    assert not result.succeeded
    assert result.source_release_confirmed


def test_backend_io_preserves_layerwise_load_session_results() -> None:
    calls = []

    class Backend:
        def validate_layerwise_support(self):
            calls.append(("validate",))

        def batch_get_start(self, keys):
            calls.append(("start", tuple(keys)))
            return [0]

        def batch_copy_get(self, keys, addresses, sizes, offsets):
            calls.append(
                (
                    "copy",
                    tuple(keys),
                    tuple(map(tuple, addresses)),
                    tuple(map(tuple, sizes)),
                    tuple(map(tuple, offsets)),
                )
            )
            return [0]

        def batch_get_end(self, keys):
            calls.append(("finish", tuple(keys)))
            return 0

    backend = Backend()
    backend_io = LayerwiseBackendIO(backend, make_backend_spec(Backend, name="mooncake"))

    backend_io.validate_support()
    assert backend_io.start_load_sessions(["key"]) == (0,)
    binding = make_binding_batch().bindings[0]
    assert backend_io.load((binding,))[0].result_code == 0
    backend_io.finish_load_sessions(["key"])
    assert [call[0] for call in calls] == ["validate", "start", "copy", "finish"]
