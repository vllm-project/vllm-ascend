"""Validate AscendStore v1 through its token-space domain model."""

from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, KVCacheSpec, MambaSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
    RequestTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import vllm_adapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector import AscendStoreV1Connector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.availability import (
    ExternalPrefixPlan,
    LookupQuery,
    RemoteAvailability,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.ownership import StoreSourceLeases
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.planner import TransferPlanner
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.progress import (
    AllocationLoadPublication,
    LoadCandidate,
    RequestSnapshot,
    ScheduledLoadPublication,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.spec import TransferPlanningSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.step import (
    ScheduledRequestKind,
    TransferPlanningStep,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program import compiler as compiler_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.compiler import (
    ProgramCompilationError,
    compile_kv_pool_program,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.program import (
    KVPoolProgram,
    _bind_transfer_layouts,
    _order_load_traversal,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.spec.compilation import (
    KVPoolCompilationSpec,
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
    resolve_group_layers,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.admission import (
    BackendExistenceStoreAdmission,
    UnconditionalStoreAdmission,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.block import (
    CheckpointBlockResolution,
    LocalBlockResolution,
    compile_block_resolutions,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.chunk import (
    CheckpointChunkProjection,
    SemanticChunkProjection,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.ownership import (
    StoreOwnershipSelection,
    compile_store_ownership,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.partition import (
    IdentityRegionPartition,
    PipelineRegionPartition,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.reachability import (
    HybridReachability,
    UnitaryReachability,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.region import (
    ContiguousRegionProjection,
    LayerwiseRegionProjection,
    StridedRegionProjection,
    TransferRegionProjection,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.remote import (
    RemoteObjectProjection,
    project_remote_identities,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.evidence import (
    ChunkAvailability,
    GroupAvailability,
    LoadCompletion,
    ReachablePrefix,
    RemoteObjectObservation,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.representation import (
    BindingBatch,
    KVBinding,
    KVChunk,
    KVMemoryGeometry,
    KVMemorySegment,
    KVMemoryView,
    KVRegion,
    KVTransferLayout,
    LocalKVRegion,
    PhysicalCoordinate,
    RemoteKVObject,
    RemoteObjectKeyBatch,
    RemoteObjectLayout,
    TransferLayoutBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.selection import (
    GroupSelection,
    KVSelection,
    LoadTransfer,
    StoreTransfer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import (
    LookupCodec,
    LookupRequest,
    LookupResult,
    TailKeyBoundary,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    CheckpointStoreCommand,
    KVTransferStep,
    LoadCommand,
    LoadCommandBatch,
    RangeStoreCommand,
    StateCheckpointSource,
    StoreCommandBatch,
    StoreSourceReleaseMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.backend import BackendIO, KeyRangeBackendIO
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.resources import KVPoolResources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.runtime import KVPoolRuntime
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.timeline.bulk import (
    AsyncLoadTimeline,
    LoadTimeline,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.vllm_adapter import (
    resolve_consumer_pipeline_partitions,
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


class FakeBlockPool:
    def __init__(self, size=32) -> None:
        self.blocks = [SimpleNamespace(block_id=block_id, ref_cnt=1) for block_id in range(size)]
        self.touched = []
        self.freed = []

    def touch(self, blocks) -> None:
        blocks = tuple(blocks)
        self.touched.append(tuple(block.block_id for block in blocks))
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks) -> None:
        blocks = tuple(blocks)
        self.freed.append(tuple(block.block_id for block in blocks))
        for block in blocks:
            block.ref_cnt -= 1


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
    align_state_groups=(),
    block_sizes_by_group=None,
) -> KVPoolTopology:
    physical_layers_by_group = physical_layers_by_group or {}
    block_sizes_by_group = block_sizes_by_group or {}
    group_topologies = tuple(
        make_group_topology(
            group_id,
            block_size=block_sizes_by_group.get(group_id, 4),
            physical_layer_ids=physical_layers_by_group.get(group_id, (group_id,)),
            uses_align_state=group_id in align_state_groups,
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


def make_group_topology(
    group_id: int,
    *,
    block_size: int = 4,
    physical_layer_ids: tuple[int, ...] | None = None,
    uses_align_state: bool = False,
    kv_cache_spec: KVCacheSpec | None = None,
    is_eagle_group: bool = False,
) -> KVPoolGroupTopology:
    if kv_cache_spec is None:
        if uses_align_state:
            kv_cache_spec = make_align_state_spec(block_size)
        else:
            kv_cache_spec = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            )
    return KVPoolGroupTopology(
        group_id=group_id,
        kv_cache_spec=kv_cache_spec,
        layers=tuple(
            KVPoolLayerTopology(physical_layer_id, (f"layers.{physical_layer_id}.group.{group_id}",))
            for physical_layer_id in physical_layer_ids or (group_id,)
        ),
        key_metadata=KeyMetadata("model", 0, 0, 0, group_id),
        is_eagle_group=is_eagle_group,
    )


def make_align_state_spec(block_size: int = 4) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
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
        LayerwiseAccessKind.KEY_RANGE if supports_layerwise else None,
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
    monkeypatch.setattr(
        compiler_module,
        "resolve_backend_spec",
        lambda *_: make_backend_spec(
            supports_layerwise=supports_layerwise,
            requires_exists_before_put=requires_exists_before_put,
        ),
    )
    if use_layerwise:
        load_kind = LoadScheduleKind.LAYERWISE
    elif async_load:
        load_kind = LoadScheduleKind.ASYNC
    else:
        load_kind = LoadScheduleKind.SYNC
    store_kind = None
    if store_enabled:
        store_kind = StoreScheduleKind.LAYERWISE if use_layerwise else StoreScheduleKind.ASYNC
    if kv_cache_groups:
        upstream_groups = dict(zip(topology.transfer_group_ids, kv_cache_groups, strict=True))
        topology = replace(
            topology,
            groups=tuple(
                replace(
                    group,
                    kv_cache_spec=upstream_groups[group.group_id].kv_cache_spec,
                    is_eagle_group=upstream_groups[group.group_id].is_eagle_group,
                )
                if group.group_id in upstream_groups
                else group
                for group in topology.groups
            ),
        )
    compilation_spec = KVPoolCompilationSpec(
        topology=topology,
        backend_name="fake",
        schedule=KVPoolSchedule(load_kind, store_kind, 2),
        max_model_len=64,
    )
    return compile_kv_pool_program(compilation_spec)


@dataclass(frozen=True)
class ProjectionNodes:
    topology: KVPoolTopology
    chunks: SemanticChunkProjection
    checkpoint_chunks: CheckpointChunkProjection
    remote_objects: RemoteObjectProjection
    blocks: LocalBlockResolution
    checkpoint_blocks: CheckpointBlockResolution
    store_ownership: StoreOwnershipSelection
    regions: TransferRegionProjection


def make_projection(
    database,
    topology: KVPoolTopology,
    region_projection: TransferRegionProjection | None = None,
) -> ProjectionNodes:
    transfer_groups = topology.transfer_groups
    align_state_group_ids = frozenset(group.group_id for group in transfer_groups if group.uses_align_state)
    if region_projection is None:
        region_projection = (
            StridedRegionProjection(topology)
            if topology.tp_partition.tp_mismatch
            else ContiguousRegionProjection(topology)
        )
    blocks, checkpoint_blocks = compile_block_resolutions(topology, align_state_group_ids)
    lookup_rank_counts = {
        group.group_id: topology.tp_size
        if group.group_id in align_state_group_ids
        else topology.tp_partition.key_rank_count
        for group in transfer_groups
    }
    return ProjectionNodes(
        topology,
        SemanticChunkProjection(
            database,
            transfer_groups,
            any(
                group.group_id in align_state_group_ids and group.block_size > database.hash_block_size
                for group in transfer_groups
            ),
        ),
        CheckpointChunkProjection(database, transfer_groups, align_state_group_ids),
        RemoteObjectProjection(topology, lookup_rank_counts),
        blocks,
        checkpoint_blocks,
        compile_store_ownership(topology, align_state_group_ids),
        region_projection,
    )


def compile_projection(nodes: ProjectionNodes, memory_geometry: KVMemoryGeometry) -> None:
    nodes.regions.bind_memory(memory_geometry)


def project_remote_objects(nodes: ProjectionNodes, selection: KVSelection):
    chunks = nodes.chunks.project(selection)
    object_keys = project_object_identities(nodes, chunks)
    return nodes.remote_objects.project_lookup(object_keys)


def project_object_identities(nodes: ProjectionNodes, chunks):
    groups_by_id = {group.group_id: group for group in nodes.topology.groups}
    return project_remote_identities(chunks, groups_by_id, nodes.topology.transfer_group_ids)


def project_bindings(
    nodes: ProjectionNodes,
    selection: KVSelection,
    block_ids_by_group: tuple[tuple[int, ...], ...],
    *,
    owned: bool = False,
) -> tuple[BindingBatch, ...]:
    chunks = nodes.chunks.project(selection)
    object_keys = project_object_identities(nodes, chunks)
    block_assignments = nodes.blocks.resolve(chunks, block_ids_by_group)
    if owned:
        block_assignments = nodes.store_ownership.select(block_assignments)
    transfer_layouts = tuple(nodes.regions.project(assignments) for assignments in block_assignments)
    remote_objects = tuple(
        nodes.remote_objects.project_transfer(layout_batch, keys)
        for layout_batch, keys in zip(transfer_layouts, object_keys, strict=True)
    )
    return tuple(
        _bind_transfer_layouts(layout_batch, objects)
        for layout_batch, objects in zip(transfer_layouts, remote_objects, strict=True)
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
    binding = KVBinding(
        LocalKVRegion(region, KVMemoryView(1, (100,), (16,))),
        remote,
        RemoteObjectLayout(coordinate, 16, (0,)),
    )
    return BindingBatch(group_id, (binding,))


def test_load_traversal_ordering_rotates_final_bindings_by_rank() -> None:
    first = make_binding_batch().bindings[0]
    second = replace(first, remote_object=replace(first.remote_object, key="second"))

    traversal = _order_load_traversal((BindingBatch(0, (first, second)),), 1)

    assert [binding.remote_object.key for binding in traversal] == ["second", "key"]


def test_backend_existence_admission_selects_final_store_bindings() -> None:
    existing_batch = make_binding_batch()
    missing_binding = replace(
        existing_batch.bindings[0],
        remote_object=replace(existing_batch.bindings[0].remote_object, key="missing"),
    )
    transfers = [
        StoreTransfer("existing", (existing_batch,)),
        StoreTransfer("missing", (BindingBatch(0, (missing_binding,)),)),
    ]
    admission = BackendExistenceStoreAdmission()
    targets = admission.observation_targets(transfers)
    observations = (
        RemoteObjectObservation(targets[0], True),
        RemoteObjectObservation(targets[1], False),
    )
    admitted = admission.select(transfers, observations)

    assert [target.key for target in targets] == ["key", "missing"]
    assert admitted[0].batches[0].bindings == ()
    assert admitted[1].batches[0].bindings == (missing_binding,)


def test_backend_existence_admission_rejects_mismatched_observations() -> None:
    transfers = [StoreTransfer("request", (make_binding_batch(),))]

    with pytest.raises(RuntimeError, match="Store observations"):
        BackendExistenceStoreAdmission().select(transfers, ())


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
    layers = resolve_group_layers(
        ["model.layers.1.v", "mtp.layers.0.attn", "model.layers.1.k"],
        base_layer_count=4,
    )
    assert layers == (
        KVPoolLayerTopology(1, ("model.layers.1.k", "model.layers.1.v")),
        KVPoolLayerTopology(4, ("mtp.layers.0.attn",)),
    )


def test_vllm_adapter_captures_authoritative_parallel_coordinates(monkeypatch) -> None:
    monkeypatch.setattr(vllm_adapter, "get_tp_group", lambda: SimpleNamespace(rank_in_group=2))
    monkeypatch.setattr(vllm_adapter, "get_pp_group", lambda: SimpleNamespace(rank_in_group=1))
    monkeypatch.setattr(vllm_adapter, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=3))
    monkeypatch.setattr(vllm_adapter, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=2))
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (4, 4))
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_dcp_kv_block_size", lambda *_: 4)
    monkeypatch.setattr(vllm_adapter, "resolve_dcp_kv_cache_spec", lambda cache_spec, _: cache_spec)

    cache_group = KVCacheGroupSpec(
        ["model.layers.0.attn"],
        FullAttentionSpec(block_size=4, num_kv_heads=8, head_size=1, dtype=torch.float32),
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        prefill_context_parallel_size=4,
        decode_context_parallel_size=4,
    )
    model_config = SimpleNamespace(
        model="org/model",
        max_model_len=64,
        use_mla=False,
        get_total_num_kv_heads=lambda: 8,
        get_total_num_hidden_layers=lambda: 1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(kv_role="kv_consumer", kv_connector_extra_config={"backend": "fake"}),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[cache_group],
        transfer_group_ids=(0,),
        prefix_cache_retention_interval=None,
    )

    spec = vllm_adapter.resolve_kv_pool_compilation_spec(vllm_config, kv_cache_config)

    assert spec.topology.tp_rank == 2
    assert spec.topology.pcp_rank == 3
    assert spec.topology.groups[0].key_metadata.pp_rank == 1
    assert spec.topology.groups[0].key_metadata.dcp_rank == 2
    assert spec.topology.groups[0].kv_cache_spec is cache_group.kv_cache_spec
    assert pickle.loads(pickle.dumps(spec)) == spec


def test_vllm_adapter_uses_scheduler_resumption_as_authoritative_request_kind() -> None:
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[0] * 4,
        num_prompt_tokens=4,
        num_tokens=7,
        block_hashes=[b"a", b"b"],
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["request"],
            resumed_req_ids={"request"},
            new_block_ids=[([3, 4],)],
            num_computed_tokens=[4],
        ),
        num_scheduled_tokens={"request": 3},
        finished_req_ids=set(),
        preempted_req_ids=None,
        kv_connector_block_state=None,
    )

    step = vllm_adapter.adapt_scheduler_output(
        scheduler_output,
        {"request": request},
        store_enabled=True,
    )

    assert isinstance(step, TransferPlanningStep)
    assert step.scheduled_requests[0].kind is ScheduledRequestKind.RESUMED
    assert step.scheduled_requests[0].block_ids_by_group == ((3, 4),)
    assert step.scheduled_requests[0].num_prompt_tokens == 4


def test_vllm_adapter_constructs_backend_before_binding_runtime_resources(monkeypatch) -> None:
    backend_arguments = []

    class Backend:
        def __init__(self, parallel_config, *, extra_config):
            backend_arguments.append((parallel_config, extra_config))

    backend_spec = make_backend_spec(Backend)
    program = SimpleNamespace(
        backend_name="fake",
        topology=SimpleNamespace(groups=("group",)),
    )
    parallel_config = object()
    extra_config = {"backend": "fake"}
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra_config),
    )
    kv_cache_config = SimpleNamespace(num_blocks=8)
    monkeypatch.setattr(vllm_adapter, "resolve_kv_pool_compilation_spec", lambda *_: "compilation")
    monkeypatch.setattr(vllm_adapter, "compile_kv_pool_program", lambda spec: program)
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda name: backend_spec)
    monkeypatch.setattr(
        vllm_adapter,
        "KVPoolRuntime",
        lambda compiled_program, resources: SimpleNamespace(program=compiled_program, resources=resources),
    )

    runtime = vllm_adapter.create_kv_pool_runtime(vllm_config, kv_cache_config)

    assert backend_arguments == [(parallel_config, extra_config)]
    assert runtime.program is program
    assert runtime.resources.backend_spec is backend_spec
    assert runtime.resources.num_blocks == 8
    assert runtime.resources._groups == ("group",)


def test_resources_bind_memory_geometry_once() -> None:
    registered_buffers = []
    backend = SimpleNamespace(
        register_buffer=lambda addresses, sizes: registered_buffers.append((addresses, sizes)),
    )
    group = make_group_topology(0, physical_layer_ids=(0, 1))
    resources = KVPoolResources(backend, make_backend_spec(type(backend)), 2, (group,))
    kv_caches = {
        "layers.0.group.0": torch.zeros((2, 1)),
        "layers.1.group.0": torch.zeros((2, 1)),
    }
    memory_geometry = resources.bind_kv_caches(kv_caches)
    assert [segment.physical_layer_id for segment in memory_geometry[0]] == [0, 1]
    assert len(registered_buffers) == 1
    assert len(registered_buffers[0][0]) == 2
    with pytest.raises(RuntimeError, match="already bound"):
        resources.bind_kv_caches(kv_caches)


def test_program_rejects_layerwise_backend_before_resource_binding(monkeypatch) -> None:
    with pytest.raises(ProgramCompilationError, match="requires a session Backend"):
        compile_program(monkeypatch, make_topology(), use_layerwise=True, supports_layerwise=False)


def test_program_uses_hybrid_reachability_to_exclude_positional_mamba_store(monkeypatch) -> None:
    topology = make_topology(align_state_groups=(0,))
    mamba_spec = MambaSpec(
        block_size=4,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=2,
    )
    program = compile_program(
        monkeypatch,
        topology,
        store_enabled=True,
        kv_cache_groups=(KVCacheGroupSpec(["layers.0"], mamba_spec),),
    )

    selection = program._reachable_region_selection.select_for_store((b"a",), TokenRange(0, 2), 2)

    assert isinstance(program._reachable_region_selection, HybridReachability)
    assert selection.groups == (GroupSelection(0, ()),)


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
    align_state_groups=(),
):
    database = database or FakeDatabase({group_id: 4 for group_id in groups})
    topology = make_topology(
        groups=groups,
        tp_mismatch=tp_mismatch,
        key_rank_count=key_rank_count,
        align_state_groups=align_state_groups,
    )
    memory_geometry = make_memory_geometry(database, topology)
    resources = FakeResources(database, backend, memory_geometry)
    projection = make_projection(database, topology)
    region_partition = IdentityRegionPartition()
    store_admission = (
        BackendExistenceStoreAdmission() if backend.requires_exists_before_put else UnconditionalStoreAdmission()
    )
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
            select_for_store=lambda hashes, token_range, prompt: KVSelection(
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
        projection.checkpoint_chunks,
        projection.remote_objects,
        projection.blocks,
        projection.checkpoint_blocks,
        projection.store_ownership,
        projection.regions,
        region_partition,
        store_admission,
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
    projection = make_projection(database, topology, LayerwiseRegionProjection(topology))
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
        projection.checkpoint_chunks,
        projection.remote_objects,
        projection.blocks,
        projection.checkpoint_blocks,
        projection.store_ownership,
        projection.regions,
        IdentityRegionPartition(),
        UnconditionalStoreAdmission(),
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
    projection = make_projection(database, topology, LayerwiseRegionProjection(topology))
    reachability = UnitaryReachability(0, 64, 4)
    program = KVPoolProgram(
        topology,
        reachability,
        projection.chunks,
        projection.checkpoint_chunks,
        projection.remote_objects,
        projection.blocks,
        projection.checkpoint_blocks,
        projection.store_ownership,
        projection.regions,
        IdentityRegionPartition(),
        BackendExistenceStoreAdmission() if backend.requires_exists_before_put else UnconditionalStoreAdmission(),
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
        KVBinding(
            LocalKVRegion(KVRegion(other, (0,)), KVMemoryView(1, (100,), (16,))),
            remote_object,
            RemoteObjectLayout(PhysicalCoordinate(), 16, (0,)),
        )
    with pytest.raises(ValueError, match="align every remote offset"):
        KVBinding(
            LocalKVRegion(region, KVMemoryView(1, (100,), (16,))),
            remote_object,
            RemoteObjectLayout(PhysicalCoordinate(), 16, (0, 8)),
        )


def test_projection_preserves_original_group_identity() -> None:
    database = FakeDatabase({0: 4, 3: 4})
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(3, None),))
    projection = make_projection(database, make_topology(groups=(3,)))
    batch = project_remote_objects(projection, selection)[0]
    assert batch.group_id == 3
    assert batch.remote_objects[0].chunk.group_id == 3
    assert "@group:3@" in batch.remote_objects[0].key


def test_chunk_and_identity_projections_keep_separate_edges() -> None:
    projection = make_projection(FakeDatabase(), make_topology())
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))

    chunk_batches = projection.chunks.project(selection)
    object_key_batches = project_object_identities(projection, chunk_batches)
    assignments = projection.blocks.resolve(chunk_batches, ((1,),))

    assert assignments[0].assignments[0].chunk is chunk_batches[0].chunks[0]
    assert object_key_batches[0].keys[0].chunk is chunk_batches[0].chunks[0]
    assert "@group:0@" in object_key_batches[0].keys[0].base_key


def test_object_identity_projection_preserves_existing_backend_keys() -> None:
    topology = make_topology()
    database = ChunkedTokenDatabase([topology.groups[0].key_metadata], [4], None, hash_block_size=4)
    projection = make_projection(database, topology)
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))

    chunks = projection.chunks.project(selection)
    key = project_object_identities(projection, chunks)[0].keys[0].base_key
    existing_records = database.process_token_key_strings(
        4, [b"a"], mask_num=0, kv_cache_group_id=0, chunk_filter=lambda _: True
    )
    existing_key = next(iter(existing_records))[2]

    assert key == existing_key


def test_graph_compiler_selects_fixed_graph_rules(monkeypatch) -> None:
    topology = make_topology()
    cases = (
        (topology, {}, ContiguousRegionProjection, IdentityRegionPartition, UnconditionalStoreAdmission),
        (
            make_topology(tp_mismatch=True),
            {},
            StridedRegionProjection,
            IdentityRegionPartition,
            UnconditionalStoreAdmission,
        ),
        (
            topology,
            {"use_layerwise": True},
            LayerwiseRegionProjection,
            IdentityRegionPartition,
            UnconditionalStoreAdmission,
        ),
        (
            topology,
            {"requires_exists_before_put": True},
            ContiguousRegionProjection,
            IdentityRegionPartition,
            BackendExistenceStoreAdmission,
        ),
        (
            make_topology(consumer_pipeline_partitions=(1, 1)),
            {},
            ContiguousRegionProjection,
            PipelineRegionPartition,
            UnconditionalStoreAdmission,
        ),
    )
    for topology, options, region_type, partition_type, admission_type in cases:
        program = compile_program(monkeypatch, topology, **options)
        assert isinstance(program._reachable_region_selection, UnitaryReachability)
        assert isinstance(program._transfer_region_projection, region_type)
        assert isinstance(program._region_partition, partition_type)
        assert isinstance(program._store_admission, admission_type)


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
    assert isinstance(program._reachable_region_selection, HybridReachability)


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
            "Layerwise region projection cannot yet be composed with TP mismatch",
        ),
        (
            make_topology(tp_mismatch=True, align_state_groups=(0,)),
            False,
            "Mamba align-state transfer cannot yet be composed with TP mismatch",
        ),
        (
            make_topology(consumer_pipeline_partitions=(1, 1)),
            True,
            "Layerwise region projection cannot yet be composed with consumer pipeline projection",
        ),
    ],
)
def test_program_compiler_rejects_conflicting_rules(monkeypatch, topology, use_layerwise, message) -> None:
    with pytest.raises(ProgramCompilationError, match=message):
        compile_program(monkeypatch, topology, use_layerwise=use_layerwise)


def test_transfer_binding_requires_a_remote_key_for_every_region() -> None:
    database = FakeDatabase()
    topology = make_topology()
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    chunk_batches = projection.chunks.project(selection)
    assignments = projection.blocks.resolve(chunk_batches, ((1,),))[0]
    regions = projection.regions.project(assignments)

    with pytest.raises(ValueError, match="no remote object key"):
        projection.remote_objects.project_transfer(regions, RemoteObjectKeyBatch(0, ()))


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
        f"model@dcp:{dcp_rank}@head_or_tp_rank:{head_rank}@pp_rank:{pp_rank}"
        f"@group:0@cache_role:kv@cache_family:default@{chunk_hash.hex()}"
        for pp_rank in range(2)
        for dcp_rank in range(2)
        for head_rank in range(2)
        for chunk_hash in (b"a", b"b")
    ]


def test_single_lookup_representation_rewrites_a_nonzero_base_rank() -> None:
    metadata = KeyMetadata("model", 3, 0, 0)
    database = ChunkedTokenDatabase([metadata], [4], None, hash_block_size=4)
    topology = make_topology()
    topology = replace(
        topology,
        groups=(replace(topology.groups[0], key_metadata=metadata, kv_cache_spec=make_align_state_spec()),),
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

    assert binding.local_region.memory.addresses == (1032,)
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
    assert first_representation.local_region.memory.sizes == (4, 4, 4, 4, 8, 8, 8, 8)


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
            local_region=replace(
                binding.local_region,
                memory=KVMemoryView(
                    binding.local_region.memory.block_id,
                    (binding.local_region.memory.addresses[0],),
                    (binding.local_region.memory.sizes[0],),
                ),
            ),
        )


def test_pipeline_region_partition_precedes_transfer_binding() -> None:
    database = FakeDatabase()
    database.group_kv_caches_base_addr[0] = [1000, 2000]
    database.group_block_len[0] = [32, 32]
    database.group_block_stride[0] = [32, 32]
    topology = make_topology(physical_layers_by_group={0: (0, 1)})
    memory_geometry = make_memory_geometry(database, topology)
    region_partition = PipelineRegionPartition((1, 1))
    region_partition.bind_memory(memory_geometry)
    projection = make_projection(database, topology)
    compile_projection(projection, memory_geometry)
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    chunks = projection.chunks.project(selection)
    object_keys = project_object_identities(projection, chunks)
    assignments = projection.store_ownership.select(projection.blocks.resolve(chunks, ((1,),)))
    layouts = tuple(projection.regions.project(batch) for batch in assignments)
    layouts = region_partition.project(layouts)
    remote_objects = projection.remote_objects.project_transfer(layouts[0], object_keys[0])
    projected = _bind_transfer_layouts(layouts[0], remote_objects).bindings
    assert [binding.local_region.memory.addresses for binding in projected] == [(1032,), (2032,)]
    assert [binding.remote_object.coordinate.consumer_pp_slice for binding in projected] == [0, 1]
    assert [binding.local_region.region.physical_layer_ids for binding in projected] == [(0,), (1,)]
    assert "@pp_rank:1@" in projected[1].remote_object.key


def test_pipeline_region_partition_projects_sparse_hybrid_groups_by_physical_layer() -> None:
    database = FakeDatabase({0: 4, 1: 4})
    database.group_kv_caches_base_addr = {0: [1000, 2000], 1: [3000, 4000]}
    database.group_block_len = {0: [32, 32], 1: [32, 32]}
    database.group_block_stride = {0: [32, 32], 1: [32, 32]}
    topology = make_topology(
        groups=(0, 1),
        physical_layers_by_group={0: (0, 2), 1: (1, 3)},
    )
    memory_geometry = make_memory_geometry(database, topology)
    region_partition = PipelineRegionPartition((2, 2))
    region_partition.bind_memory(memory_geometry)
    projection = make_projection(database, topology)
    compile_projection(projection, memory_geometry)
    selection = KVSelection(
        TokenRange(0, 4),
        (b"a",),
        (GroupSelection(0, None), GroupSelection(1, None)),
    )
    chunks = projection.chunks.project(selection)
    assignments = projection.blocks.resolve(chunks, ((1,), (2,)))
    layouts = tuple(projection.regions.project(batch) for batch in assignments)

    projected = region_partition.project(layouts)

    assert [[layout.local_region.region.physical_layer_ids for layout in batch.layouts] for batch in projected] == [
        [(0,), (2,)],
        [(1,), (3,)],
    ]
    assert [[layout.remote_layout.coordinate.consumer_pp_slice for layout in batch.layouts] for batch in projected] == [
        [0, 1],
        [0, 1],
    ]


def test_pipeline_region_partition_assigns_mtp_layers_to_the_final_rank() -> None:
    partition = PipelineRegionPartition((2, 2))
    partition.bind_memory(
        {
            0: (
                KVMemorySegment("layers.0", 0, 1000, 32, 32, 8),
                KVMemorySegment("mtp.layers.0", 4, 2000, 32, 32, 8),
            )
        }
    )
    chunk = KVChunk(0, 0, TokenRange(0, 4), b"a")
    batch = TransferLayoutBatch(
        0,
        (
            KVTransferLayout(
                LocalKVRegion(KVRegion(chunk, (0, 4)), KVMemoryView(1, (1032, 2032), (32, 32))),
                RemoteObjectLayout(PhysicalCoordinate(), 64, (0, 32)),
            ),
        ),
    )

    projected = partition.project((batch,))[0]

    assert [item.local_region.region.physical_layer_ids for item in projected.layouts] == [(0,), (4,)]
    assert [item.remote_layout.coordinate.consumer_pp_slice for item in projected.layouts] == [0, 1]


def test_layerwise_projection_splits_regions_inside_one_remote_object() -> None:
    database = FakeDatabase()
    topology = make_topology(physical_layers_by_group={0: (0, 1)})
    projection = make_projection(database, topology, LayerwiseRegionProjection(topology))
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

    assert [binding.local_region.region.physical_layer_ids for binding in bindings] == [(0,), (1,)]
    assert bindings[0].remote_object is bindings[1].remote_object
    assert [binding.remote_layout.offsets for binding in bindings] == [(0, 32), (64,)]
    assert [binding.local_region.memory.sizes for binding in bindings] == [(32, 32), (64,)]
    assert [binding.local_region.memory.addresses for binding in bindings] == [(1064, 2064), (3128,)]


def test_store_ownership_precedes_physical_fanout() -> None:
    database = FakeDatabase()
    topology = make_topology(tp_mismatch=True, pcp_rank=1, pcp_size=2)
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    selection = KVSelection(TokenRange(0, 8), (b"a", b"b"), (GroupSelection(0, None),))
    batch = project_bindings(projection, selection, ((1, 2),), owned=True)[0]
    assert {binding.local_region.memory.block_id for binding in batch.bindings} == {2}
    assert len(batch.bindings) == 2


def test_strided_store_ownership_does_not_drop_chunks_as_tp_replicas() -> None:
    database = FakeDatabase()
    topology = replace(make_topology(tp_mismatch=True), put_step=2)
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    selection = KVSelection(TokenRange(0, 8), (b"a", b"b"), (GroupSelection(0, None),))
    batch = project_bindings(projection, selection, ((1, 2),), owned=True)[0]
    assert {binding.local_region.memory.block_id for binding in batch.bindings} == {1, 2}
    assert len(batch.bindings) == 4


def test_strided_mapping_keeps_align_state_null_blocks() -> None:
    database = FakeDatabase()
    topology = make_topology(tp_mismatch=True)
    topology = replace(topology, groups=(replace(topology.groups[0], kv_cache_spec=make_align_state_spec()),))
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    batch = project_bindings(projection, selection, ((0,),))[0]
    assert len(batch.bindings) == 2
    assert all(binding.local_region.memory.block_id == 0 for binding in batch.bindings)


def test_identity_region_partition_preserves_strided_regions() -> None:
    database = FakeDatabase()
    topology = make_topology(tp_mismatch=True)
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    selection = KVSelection(TokenRange(0, 4), (b"a",), (GroupSelection(0, None),))
    chunks = projection.chunks.project(selection)
    assignments = projection.store_ownership.select(projection.blocks.resolve(chunks, ((1,),)))
    regions = tuple(projection.regions.project(batch) for batch in assignments)
    assert IdentityRegionPartition().project(regions) is regions


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
    assert reachability.resolve_available_end(selection, (availability,)) == ReachablePrefix(4)


def test_hybrid_reachability_preserves_partial_tail_object_identity() -> None:
    groups = (
        make_group_topology(
            0,
            block_size=16,
            kv_cache_spec=FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32),
        ),
        make_group_topology(
            1,
            block_size=16,
            kv_cache_spec=make_align_state_spec(16),
        ),
    )
    reachability = HybridReachability(
        groups,
        scheduler_block_size=16,
        hash_block_size=4,
        max_model_len=64,
    )
    selection = reachability.select_for_lookup((b"a", b"b", b"c"), TokenRange(0, 12))
    availability = tuple(
        GroupAvailability(
            group_id,
            (
                ChunkAvailability(TokenRange(0, 4), b"a", False),
                ChunkAvailability(TokenRange(0, 8), b"b", False),
                ChunkAvailability(TokenRange(0, 12), b"c", True),
            ),
        )
        for group_id in (0, 1)
    )

    assert reachability.resolve_available_end(selection, availability) == ReachablePrefix(
        12,
        (TailKeyBoundary(0, 12), TailKeyBoundary(1, 12)),
    )


def test_eagle_lookup_drops_the_unstable_trailing_block() -> None:
    group = make_group_topology(
        0,
        kv_cache_spec=FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32),
        is_eagle_group=True,
    )
    reachability = HybridReachability(
        (group,),
        scheduler_block_size=4,
        hash_block_size=4,
        max_model_len=64,
        use_eagle=True,
    )
    selection = reachability.select_for_lookup((b"a", b"b"), TokenRange(0, 8))
    availability = (
        GroupAvailability(
            0,
            (
                ChunkAvailability(TokenRange(0, 4), b"a", True),
                ChunkAvailability(TokenRange(4, 8), b"b", True),
            ),
        ),
    )

    assert reachability.resolve_available_end(selection, availability) == ReachablePrefix(4)


def test_partial_tail_identity_projects_full_local_blocks_for_load() -> None:
    topology = make_topology(
        groups=(0, 1),
        align_state_groups=(1,),
        block_sizes_by_group={0: 16, 1: 16},
    )
    database = FakeDatabase({0: 16, 1: 16})
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    selection = KVSelection(
        TokenRange(0, 12),
        (b"a", b"b", b"c"),
        (GroupSelection(0, None), GroupSelection(1, None)),
    )
    boundaries = (TailKeyBoundary(0, 12), TailKeyBoundary(1, 12))

    chunks = projection.chunks.project_load(selection, boundaries)
    assignments = projection.blocks.resolve(chunks, ((3,), (7,)))
    object_keys = project_object_identities(projection, chunks)

    assert [batch.chunks[0].content_hash for batch in chunks] == [b"c", b"c"]
    assert [batch.assignments[0].memory_token_count for batch in assignments] == [16, 16]
    assert [batch.keys[0].base_key.rsplit("@", 1)[-1] for batch in object_keys] == ["63", "63"]


def test_mamba_lookup_uses_its_tp_rank_namespace() -> None:
    topology = replace(
        make_topology(
            groups=(0, 1),
            align_state_groups=(1,),
            block_sizes_by_group={0: 16, 1: 16},
        ),
        tp_size=2,
    )
    projection = make_projection(FakeDatabase({0: 16, 1: 16}), topology)
    selection = KVSelection(
        TokenRange(0, 4),
        (b"a",),
        (GroupSelection(0, None), GroupSelection(1, None)),
    )
    chunks = projection.chunks.project_lookup(selection)
    object_keys = project_object_identities(projection, chunks)

    remote_objects = projection.remote_objects.project_lookup(object_keys)

    assert len(remote_objects[0].remote_objects) == 1
    assert len(remote_objects[1].remote_objects) == 2
    assert {item.coordinate.head_rank for item in remote_objects[1].remote_objects} == {0, 1}


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

    monkeypatch.setattr(FullAttentionManager, "reachable_block_mask", classmethod(reachable_mask))
    reachability = HybridReachability(
        (
            make_group_topology(
                0,
                kv_cache_spec=FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32),
            ),
            make_group_topology(
                1,
                kv_cache_spec=FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32),
            ),
        ),
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
    result = LookupResult(8, (TailKeyBoundary(1, 12), TailKeyBoundary(3, 8)))
    assert codec.decode_result(codec.encode_result(result)) == result


def test_backend_io_keeps_results_attached_to_exact_bindings() -> None:
    backend = FakeBackend()
    backend.get_result = [0]
    outcomes = BackendIO(backend, make_backend_spec()).load(make_binding_batch().bindings)
    assert outcomes[0].binding == make_binding_batch().bindings[0]
    assert outcomes[0].result_code == 0


@pytest.mark.parametrize("presence", [(), (2,)])
def test_backend_io_rejects_invalid_object_observations(monkeypatch, presence) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(backend, "exists", lambda _keys: presence)
    remote_object = make_binding_batch().bindings[0].remote_object

    with pytest.raises(ValueError, match="Backend returned"):
        BackendIO(backend, make_backend_spec()).observe_objects((remote_object,))


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
    store = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    begin_kv_pool_step(runtime, store=store)
    runtime.finish_step()
    result = runtime.fence_previous_store()
    runtime.close()
    assert result[0].evidence.succeeded
    assert not event.synchronized
    assert [call[0] for call in backend.calls] == ["set_device", "exists"]


def test_kv_pool_runtime_keeps_the_compiled_store_admission(monkeypatch) -> None:
    backend = FakeBackend()
    runtime, _ = make_kv_pool_runtime(backend, registered=True)
    backend.requires_exists_before_put = True
    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    store = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
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
    store = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 7),))
    begin_kv_pool_step(runtime, store=store)
    runtime.finish_step()
    with pytest.raises(RuntimeError, match="Store failed"):
        runtime.fence_previous_store()
    batch = runtime._pending_store_batch
    assert batch is not None
    result = batch.completions[0]
    assert not result.evidence.source_release_confirmed
    assert isinstance(result.evidence.error, RuntimeError)
    assert runtime.take_released_store_job_ids() == set()
    with pytest.raises(RuntimeError, match="previous Store failure"):
        runtime.close()
    assert runtime._timeline.store is not None
    assert not runtime._timeline.store._executor.is_alive()
    assert not resources.closed


def test_kv_pool_runtime_preserves_source_after_native_failure_result(monkeypatch) -> None:
    backend = FakeBackend()
    backend.put_result = [-1]
    runtime, resources = make_kv_pool_runtime(backend, registered=True)
    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    store = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 7),))
    begin_kv_pool_step(runtime, store=store)
    runtime.finish_step()

    with pytest.raises(RuntimeError, match="result codes"):
        runtime.fence_previous_store()

    assert runtime._pending_store_batch is not None
    assert not runtime._pending_store_batch.completions[0].evidence.source_release_confirmed
    assert runtime.take_released_store_job_ids() == set()
    with pytest.raises(RuntimeError, match="previous Store failure"):
        runtime.close()
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
    store = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
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
    store = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
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
    store = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
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
    store = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
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
    store = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
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
    store = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
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
    store = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
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
    store = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    begin_kv_pool_step(runtime, store=store)
    runtime.finish_step()
    results = runtime.fence_previous_store()
    runtime.close()
    assert results[0].evidence.succeeded
    assert event.synchronized
    assert [call[0] for call in backend.calls].count("put") == 1
    assert resources.closed


def test_kv_pool_runtime_submits_handed_off_checkpoint_without_forward(monkeypatch) -> None:
    backend = FakeBackend()
    backend.put_result = [0]
    runtime, _ = make_kv_pool_runtime(backend, registered=True, align_state_groups=(0,))
    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    command = CheckpointStoreCommand(
        "request",
        ((7,),),
        (b"a",),
        0,
        (StateCheckpointSource(0, 7, 4),),
        store_job_id=9,
    )

    runtime.begin_step(KVTransferStep(store=StoreCommandBatch(source_ready_commands=(command,))))
    runtime.end_step()
    completions = runtime.fence_previous_store()
    runtime.close()

    assert completions[0].evidence.succeeded
    assert runtime.take_released_store_job_ids() == {9}
    assert [call[0] for call in backend.calls].count("put") == 1


def test_kv_pool_runtime_coalesces_mixed_store_sources_at_step_fence(monkeypatch) -> None:
    backend = FakeBackend()
    backend.put_result = [0]
    runtime, _ = make_kv_pool_runtime(backend, registered=True, align_state_groups=(0,))
    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    pending = RangeStoreCommand("active", TokenRange(0, 4), ((3,),), (b"a",), 4)
    ready = CheckpointStoreCommand(
        "finished",
        ((7,),),
        (b"b",),
        0,
        (StateCheckpointSource(0, 7, 4),),
    )

    runtime.begin_step(
        KVTransferStep(store=StoreCommandBatch(source_pending_commands=(pending,), source_ready_commands=(ready,)))
    )

    assert [call[0] for call in backend.calls].count("put") == 0
    runtime.finish_step()
    runtime.fence_previous_store()
    runtime.close()

    assert [call[0] for call in backend.calls].count("put") == 2


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


def confirm_planner_allocation(planner, request, blocks, allocated_external_tokens):
    planner.confirm_allocation(
        request.request_id,
        tuple(tuple(block_ids) for block_ids in blocks),
        tuple(request.block_hashes),
        getattr(request, "num_prompt_tokens", len(request.prompt_token_ids)),
        allocated_external_tokens,
    )


def build_planner_step(planner, scheduler_output, requests=None, *, resumed_request_ids=()):
    requests = requests or {}
    for request in requests.values():
        if not hasattr(request, "prompt_token_ids"):
            request.prompt_token_ids = [0] * getattr(request, "num_prompt_tokens", 0)
        if not hasattr(request, "num_prompt_tokens"):
            request.num_prompt_tokens = len(request.prompt_token_ids)
    cached = scheduler_output.scheduled_cached_reqs
    cached.resumed_req_ids = set(resumed_request_ids)
    if not hasattr(scheduler_output, "kv_connector_block_state"):
        scheduler_output.kv_connector_block_state = None
    if not hasattr(scheduler_output, "num_scheduled_tokens"):
        scheduler_output.num_scheduled_tokens = {}
    if not hasattr(scheduler_output, "preempted_req_ids"):
        scheduler_output.preempted_req_ids = set()
    planning_step = vllm_adapter.adapt_scheduler_output(
        scheduler_output,
        requests,
        store_enabled=planner._store_enabled,
    )
    return planner.build_step(planning_step)


def test_planner_full_hit_keeps_one_token_for_vllm_but_loads_allocated_chunk() -> None:
    planner = make_planner(RemoteAvailability(TokenRange(0, 12), 11))
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[0] * 12,
        num_tokens=12,
        block_hashes=[b"a", b"b", b"c"],
    )
    result = planner.lookup(LookupQuery("request", 12, 12, request.block_hashes, 0))
    confirm_planner_allocation(planner, request, ([1, 2, 3],), 11)
    step = build_planner_step(
        planner,
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[SimpleNamespace(req_id="request", num_computed_tokens=11, block_ids=([1, 2, 3],))],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
            num_scheduled_tokens={"request": 1},
        ),
        {"request": request},
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
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[0] * 8,
        num_tokens=8,
        block_hashes=[b"a", b"b"],
    )
    assert planner.lookup(LookupQuery("request", 8, 9, request.block_hashes, 4)).load_is_deferred
    confirm_planner_allocation(planner, request, ([1, 2],), 4)
    step = build_planner_step(
        planner,
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        ),
        {"request": request},
    )
    assert step.load.commands[0].load_range == TokenRange(4, 8)


def test_planner_running_request_publishes_monotonic_store_frontier() -> None:
    planner = make_planner(None)
    request = SimpleNamespace(
        request_id="request",
        num_computed_tokens=0,
        num_prompt_tokens=8,
        prompt_token_ids=[0] * 8,
        num_tokens=8,
        block_hashes=[b"a"],
    )
    confirm_planner_allocation(planner, request, ([1],), 0)
    first = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[SimpleNamespace(req_id="request", num_computed_tokens=0, block_ids=([1],))],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        num_scheduled_tokens={"request": 2},
    )
    assert build_planner_step(planner, first, {"request": request}).store.commands == ()
    request.num_computed_tokens = 2
    second = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["request"],
            new_block_ids=[None],
            num_computed_tokens=[2],
        ),
        num_scheduled_tokens={"request": 2},
    )
    store = build_planner_step(planner, second, {"request": request}).store.commands[0]
    assert store.store_range == TokenRange(0, 4)
    assert planner.request_progress["request"].published_store_end_token == 4


def test_planner_uses_step_coordinate_to_distinguish_prefill_from_decode() -> None:
    planner = make_planner(None)
    request = SimpleNamespace(
        request_id="request",
        num_computed_tokens=8,
        num_prompt_tokens=8,
        num_tokens=8,
        block_hashes=[b"a", b"b"],
    )
    planner.request_progress["request"] = RequestSnapshot("request", 4, ((1, 2),), (b"a", b"b"), 8)
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["request"],
            new_block_ids=[None],
            num_computed_tokens=[4],
        ),
        num_scheduled_tokens={"request": 4},
    )

    step = build_planner_step(planner, scheduler_output, {"request": request})

    assert step.store.commands[0].store_range == TokenRange(0, 8)


def test_planner_never_publishes_speculative_draft_positions() -> None:
    planner = make_planner(None, save_decode=True)
    request = SimpleNamespace(
        request_id="request",
        num_computed_tokens=8,
        num_prompt_tokens=8,
        num_tokens=9,
        block_hashes=[b"a", b"b", b"c"],
    )
    planner.request_progress["request"] = RequestSnapshot(
        "request",
        8,
        ((1, 2, 3),),
        (b"a", b"b"),
        8,
        published_store_end_token=8,
        committable_end_token=8,
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["request"],
            new_block_ids=[None],
            num_computed_tokens=[8],
        ),
        num_scheduled_tokens={"request": 4},
    )

    step = build_planner_step(planner, scheduler_output, {"request": request})

    assert step.store.commands == ()
    assert planner.request_progress["request"].published_store_end_token == 8
    assert planner.request_progress["request"].request_token_len == 12
    assert planner.request_progress["request"].store_end_token == 9

    request.num_computed_tokens = 9
    request.num_tokens = 10
    next_step = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["request"],
            new_block_ids=[None],
            num_computed_tokens=[9],
        ),
        num_scheduled_tokens={"request": 1},
    )
    build_planner_step(planner, next_step, {"request": request})

    assert planner.request_progress["request"].request_token_len == 10
    assert planner.request_progress["request"].store_end_token == 10


def test_chunk_continuation_keeps_layerwise_load_session_step_local() -> None:
    backend = FakeBackend()
    runtime, _ = make_layerwise_load_runtime(backend)
    planner = make_planner(RemoteAvailability(TokenRange(0, 4), 4))
    request = SimpleNamespace(
        request_id="request",
        num_computed_tokens=4,
        num_prompt_tokens=12,
        prompt_token_ids=[0] * 12,
        num_tokens=12,
        block_hashes=[b"a", b"b"],
    )
    planner.lookup(LookupQuery("request", 12, 12, request.block_hashes, 0))
    confirm_planner_allocation(planner, request, ([1],), 4)
    first_step = build_planner_step(
        planner,
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[SimpleNamespace(req_id="request", num_computed_tokens=4, block_ids=([1, 2],))],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
            num_scheduled_tokens={"request": 4},
        ),
        {"request": request},
    )
    runtime.begin_step(first_step)
    runtime.start_load()
    runtime.wait_for_layer_load("layers.0.group.0")
    runtime.wait_for_layer_load("layers.1.group.0")
    runtime.end_step()

    request.num_computed_tokens = 8
    request.block_hashes = [b"a", b"b", b"c"]
    continued_step = build_planner_step(
        planner,
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["request"],
                new_block_ids=[([3],)],
                num_computed_tokens=[8],
            ),
            num_scheduled_tokens={"request": 4},
        ),
        {"request": request},
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
        all_token_ids=[0] * 8,
        num_tokens=8,
        block_hashes=[b"a", b"b"],
    )
    planner.lookup(LookupQuery("request", 8, 8, request.block_hashes, 0))
    confirm_planner_allocation(planner, request, ([1],), 4)
    first_step = build_planner_step(
        planner,
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[SimpleNamespace(req_id="request", num_computed_tokens=4, block_ids=([1],))],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
            num_scheduled_tokens={"request": 4},
        ),
        {"request": request},
    )
    runtime.begin_step(first_step)
    runtime.start_load()
    runtime.wait_for_layer_load("layers.0.group.0")
    runtime.wait_for_layer_load("layers.1.group.0")
    runtime.end_step()

    build_planner_step(
        planner,
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids={"request"},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        ),
    )
    planner.lookup(LookupQuery("request", 8, 8, request.block_hashes, 0))
    confirm_planner_allocation(planner, request, ([7],), 4)
    resumed_step = build_planner_step(
        planner,
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["request"],
                new_block_ids=[([7],)],
                num_computed_tokens=[0],
            ),
            num_scheduled_tokens={"request": 4},
        ),
        {"request": request},
        resumed_request_ids={"request"},
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
        all_token_ids=[0] * 8,
        num_tokens=8,
        block_hashes=[b"a", b"b"],
    )
    planner.request_progress["request"] = RequestSnapshot("request", 4, ((1,),), (b"a",), 8, 4)
    build_planner_step(
        planner,
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids={"request"},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        ),
    )
    confirm_planner_allocation(planner, request, ([3, 4],), 0)
    request.num_computed_tokens = 8
    step = build_planner_step(
        planner,
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["request"],
                new_block_ids=[([3, 4],)],
                num_computed_tokens=[4],
            ),
            num_scheduled_tokens={"request": 4},
        ),
        {"request": request},
        resumed_request_ids={"request"},
    )
    assert step.store.commands[0].block_ids_by_group == ((3, 4),)
    assert planner.request_progress["request"].published_store_end_token == 8
    assert planner.request_progress["request"].request_token_len == 8


def test_planner_finished_request_discards_pending_load_state() -> None:
    planner = make_planner(RemoteAvailability(TokenRange(0, 4), 4))
    planner.lookup(LookupQuery("request", 4, 5, [b"a"], 0))
    build_planner_step(
        planner,
        SimpleNamespace(
            finished_req_ids={"request"},
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        ),
    )
    assert "request" not in planner._pending_loads


def test_request_snapshot_replaces_authoritative_position_and_advances_every_group() -> None:
    snapshot = RequestSnapshot("request", 4, ((1,), (10,)), (b"a",), 8)
    advanced = snapshot.with_position(8, ([2], [11]), [b"a", b"b"], 8)
    assert snapshot.block_ids_by_group == ((1,), (10,))
    assert advanced.block_ids_by_group == ((1, 2), (10, 11))


def test_transfer_step_keeps_load_and_store_commands_separate() -> None:
    load = LoadCommand("load", TokenRange(0, 4), ((1,),), (b"a",))
    store = RangeStoreCommand("store", TokenRange(0, 4), ((2,),), (b"b",), 4)
    step = KVTransferStep(LoadCommandBatch((load,)), StoreCommandBatch((store,)))
    assert step.load.commands == (load,)
    assert step.store.commands == (store,)
    assert step.store.source_pending_commands == (store,)
    assert step.store.source_ready_commands == ()


def test_planner_publishes_boundary_state_beyond_the_normal_store_frontier() -> None:
    planner = TransferPlanner(
        TransferPlanningSpec(4, 4, (0, 1), True),
        FakeAvailabilityProbe(None),
        ScheduledLoadPublication(),
        store_enabled=True,
        save_decode_cache=False,
    )
    request = SimpleNamespace(block_hashes=[b"a", b"b"])
    planner.request_progress["request"] = RequestSnapshot(
        "request",
        8,
        ((1, 2), (10, 11)),
        (b"a", b"b"),
        8,
        published_store_end_token=4,
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        kv_connector_block_state=SimpleNamespace(boundary_state_offloads={"request": [(1, 11, 8)]}),
    )

    store = build_planner_step(planner, scheduler_output, {"request": request}).store
    command = store.commands[0]

    assert command == CheckpointStoreCommand(
        "request",
        ((1, 2), (10, 11)),
        (b"a", b"b"),
        4,
        (StateCheckpointSource(1, 11, 8),),
    )
    assert store.source_ready_commands == (command,)
    assert store.source_pending_commands == ()


def test_planner_publishes_each_boundary_state_as_one_remote_object_version() -> None:
    planner = TransferPlanner(
        TransferPlanningSpec(4, 4, (0, 1), True),
        FakeAvailabilityProbe(None),
        ScheduledLoadPublication(),
        store_enabled=True,
        save_decode_cache=False,
    )
    request = SimpleNamespace(block_hashes=[b"a", b"b", b"c"])
    planner.request_progress["request"] = RequestSnapshot(
        "request",
        12,
        ((1,), (10, 11)),
        (b"a", b"b", b"c"),
        8,
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        kv_connector_block_state=SimpleNamespace(boundary_state_offloads={"request": [(1, 10, 8), (1, 11, 12)]}),
    )

    commands = build_planner_step(planner, scheduler_output, {"request": request}).store.commands

    assert [command.sources[0].boundary_token for command in commands] == [8, 12]
    assert [command.sources[0].block_id for command in commands] == [10, 11]


def test_program_projects_boundary_state_from_the_exact_handed_off_block() -> None:
    topology = make_topology(align_state_groups=(0,), block_sizes_by_group={0: 8})
    database = FakeDatabase({0: 8})
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    reachability = UnitaryReachability(0, 64, 4)
    program = KVPoolProgram(
        topology,
        reachability,
        projection.chunks,
        projection.checkpoint_chunks,
        projection.remote_objects,
        projection.blocks,
        projection.checkpoint_blocks,
        projection.store_ownership,
        projection.regions,
        IdentityRegionPartition(),
        UnconditionalStoreAdmission(),
        "fake",
        KVPoolSchedule(LoadScheduleKind.SYNC, StoreScheduleKind.ASYNC, 2),
    )
    command = CheckpointStoreCommand("request", ((7,),), (b"a",), 0, (StateCheckpointSource(0, 7, 4),), store_job_id=3)
    transfer = program.select_store_transfers((command,))[0]

    assert transfer.store_job_id == 3
    assert transfer.batches[0].bindings[0].local_region.memory.block_id == 7
    assert transfer.batches[0].bindings[0].local_region.memory.sizes == (64,)
    assert transfer.batches[0].bindings[0].remote_object.key.endswith("@61")


def test_program_projects_other_groups_needed_by_a_sub_block_mamba_boundary() -> None:
    topology = make_topology(groups=(0, 1), align_state_groups=(1,), block_sizes_by_group={0: 8, 1: 8})
    database = FakeDatabase({0: 8, 1: 8})
    projection = make_projection(database, topology)
    compile_projection(projection, make_memory_geometry(database, topology))
    reachability = UnitaryReachability(0, 64, 4)
    program = KVPoolProgram(
        topology,
        reachability,
        projection.chunks,
        projection.checkpoint_chunks,
        projection.remote_objects,
        projection.blocks,
        projection.checkpoint_blocks,
        projection.store_ownership,
        projection.regions,
        IdentityRegionPartition(),
        UnconditionalStoreAdmission(),
        "fake",
        KVPoolSchedule(LoadScheduleKind.SYNC, StoreScheduleKind.ASYNC, 2),
    )
    command = CheckpointStoreCommand("request", ((3,), (11,)), (b"a",), 0, (StateCheckpointSource(1, 11, 4),))
    transfer = program.select_store_transfers((command,))[0]

    bindings_by_group = {batch.group_id: batch.bindings for batch in transfer.batches}
    assert set(bindings_by_group) == {0, 1}
    assert bindings_by_group[0][0].local_region.memory.block_id == 3
    assert bindings_by_group[1][0].local_region.memory.block_id == 11
    assert bindings_by_group[0][0].local_region.memory.sizes == (64,)
    assert bindings_by_group[1][0].local_region.memory.sizes == (64,)


def test_connector_store_leases_exclude_mamba_position_table_and_release_after_every_worker() -> None:
    pool = FakeBlockPool()
    leases = StoreSourceLeases(frozenset({0, 1}), frozenset({1}), 2)
    leases.bind_block_pool(pool)
    normal = RangeStoreCommand("request", TokenRange(0, 4), ((1, 2), (10, 11, 12)), (b"a",), 4)
    checkpoint = CheckpointStoreCommand(
        "request", ((1, 2), (10, 11, 12)), (b"a",), 0, (StateCheckpointSource(1, 11, 4),)
    )

    leased_normal = leases.acquire(normal)
    leased_checkpoint = leases.acquire(checkpoint)

    assert pool.touched == [(1, 2), (11, 1, 2)]
    leases.release({leased_normal.store_job_id: 1})
    assert pool.freed == []
    leases.release({leased_normal.store_job_id: 1})
    leases.release({leased_checkpoint.store_job_id: 2})
    assert pool.freed == [(2, 1), (2, 1, 11)]


def test_store_source_release_metadata_aggregates_worker_counts() -> None:
    metadata = StoreSourceReleaseMetadata({3: 1})

    metadata.aggregate(StoreSourceReleaseMetadata({3: 2, 4: 1}))

    assert metadata.released_store_jobs == {3: 3, 4: 1}


def test_connector_finished_partial_tail_pins_exact_source_without_delaying_request_free() -> None:
    pool = FakeBlockPool()
    connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    connector._store_enabled = True
    connector._hash_block_size = 4
    connector._transfer_group_ids = frozenset({0, 1})
    connector._align_state_group_ids = frozenset({1})
    connector._store_source_leases = StoreSourceLeases(frozenset({0, 1}), frozenset({1}), 1)
    connector._store_source_leases.bind_block_pool(pool)
    connector._finished_checkpoint_stores = []
    snapshot = RequestSnapshot("request", 8, ((1, 2), (10, 11)), (b"a", b"b"), 8, 4)
    connector.planner = SimpleNamespace(request_progress={"request": snapshot})
    request = SimpleNamespace(request_id="request", block_hashes=[b"a", b"b"])

    delay_free = connector.register_finished_partial_tail(request, ([1, 2], [10, 11]), [(1, 11, 8)])

    assert not delay_free
    assert pool.touched == [(11, 1, 2)]
    assert connector._finished_checkpoint_stores[0].sources == (StateCheckpointSource(1, 11, 8),)
    assert pool.blocks[11].ref_cnt == 2

    pool.free_blocks(pool.blocks[block_id] for block_id in (1, 2, 10, 11))

    assert pool.blocks[11].ref_cnt == 1
    store_job_id = connector._finished_checkpoint_stores[0].store_job_id
    assert store_job_id is not None
    connector._store_source_leases.release({store_job_id: 1})
    assert pool.blocks[11].ref_cnt == 0


def test_runtime_reports_store_job_only_after_source_release_is_confirmed(monkeypatch) -> None:
    backend = FakeBackend()
    runtime, _ = make_kv_pool_runtime(backend, registered=True)
    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    store = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 9),))
    begin_kv_pool_step(runtime, store=store)
    runtime.finish_step()

    runtime.fence_previous_store()

    assert runtime.take_released_store_job_ids() == {9}
    assert runtime.take_released_store_job_ids() == set()


def test_connector_translates_vllm_lookup_into_planner_query() -> None:
    queries = []
    instance = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    instance.planner = SimpleNamespace(lookup=lambda query: queries.append(query) or ExternalPrefixPlan(4, True))
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=None,
        num_prompt_tokens=8,
        num_tokens=9,
        block_hashes=[b"a", b"b"],
    )
    assert instance.get_num_new_matched_tokens(request, 4) == (4, True)
    assert queries == [LookupQuery("request", 8, 9, request.block_hashes, 4)]


def test_connector_translates_vllm_allocation_into_planner_facts() -> None:
    confirmations = []
    instance = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    instance.planner = SimpleNamespace(confirm_allocation=lambda *args: confirmations.append(args))
    instance._requests = {}
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=None,
        num_prompt_tokens=8,
        block_hashes=[b"a", b"b"],
    )
    blocks = SimpleNamespace(get_block_ids=lambda: ([1, 2],))

    instance.update_state_after_alloc(request, blocks, 4)

    assert instance._requests == {"request": request}
    assert confirmations == [("request", ((1, 2),), (b"a", b"b"), 8, 4)]


def test_connector_discards_tracked_requests_after_planning_lifecycle() -> None:
    planning_steps = []
    instance = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    instance.planner = SimpleNamespace(
        build_step=lambda planning_step: planning_steps.append(planning_step) or KVTransferStep()
    )
    instance._requests = {"request": SimpleNamespace()}
    instance._store_enabled = False
    instance._finished_checkpoint_stores = []
    instance._store_source_leases = SimpleNamespace(acquire=lambda command: command)
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={},
        finished_req_ids={"request"},
        preempted_req_ids=None,
        kv_connector_block_state=None,
    )

    instance.build_connector_meta(scheduler_output)

    assert planning_steps[0].finished_request_ids == {"request"}
    assert instance._requests == {}


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


@pytest.mark.parametrize("native_result", [None, [-1], [0, 0]])
def test_backend_io_requires_complete_success_to_release_source(native_result) -> None:
    class Store:
        def batch_put_from_multi_buffers(self, keys, addresses, sizes, replicate_config):
            return native_result

    backend = SimpleNamespace(
        store=Store(),
        ensure_initialized=lambda: None,
        _build_replicate_config=lambda: object(),
    )
    result = BackendIO(backend, make_backend_spec(type(backend), name="mooncake")).store((make_binding_batch(),))
    assert not result.succeeded
    assert not result.source_release_confirmed


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
    backend_io = KeyRangeBackendIO(backend, make_backend_spec(Backend, name="mooncake"))

    backend_io.validate_support()
    assert backend_io.start_load_sessions(["key"], [32]) == (0,)
    binding = make_binding_batch().bindings[0]
    assert backend_io.load((binding,))[0].result_code == 0
    backend_io.finish_load_sessions(["key"])
    assert [call[0] for call in calls] == ["validate", "start", "copy", "finish"]


def test_layerwise_backend_io_requires_complete_success_to_release_source() -> None:
    backend = SimpleNamespace(batch_copy_put=lambda *args: [-1])
    backend_io = KeyRangeBackendIO(backend, make_backend_spec(type(backend), name="mooncake"))

    result = backend_io.store((make_binding_batch(),))

    assert not result.succeeded
    assert not result.source_release_confirmed
