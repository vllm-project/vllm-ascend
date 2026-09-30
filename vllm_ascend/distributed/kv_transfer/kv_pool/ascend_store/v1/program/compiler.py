"""Compile one immutable KV Pool program from resolved startup facts."""

from __future__ import annotations

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ..backend import resolve_backend_spec
from .program import KVPoolProgram
from .spec.compilation import KVPoolCompilationSpec
from .stages.admission import BackendExistenceStoreAdmission, StoreAdmission, UnconditionalStoreAdmission
from .stages.block import compile_block_resolutions
from .stages.chunk import CheckpointChunkProjection, SemanticChunkProjection
from .stages.ownership import compile_store_ownership
from .stages.partition import IdentityRegionPartition, PipelineRegionPartition, RegionPartition
from .stages.reachability import HybridReachability, ReachableRegionSelection, UnitaryReachability
from .stages.region import (
    ContiguousRegionProjection,
    LayerwiseRegionProjection,
    StridedRegionProjection,
    TransferRegionProjection,
)
from .stages.remote import RemoteObjectProjection


class ProgramCompilationError(ValueError):
    """Report an invalid startup-time combination of KV Pool rules."""


def compile_kv_pool_program(spec: KVPoolCompilationSpec) -> KVPoolProgram:
    """Select fixed spatial and temporal stages without consulting process state."""

    topology = spec.topology
    schedule = spec.schedule
    use_layerwise = schedule.requires_layerwise_backend
    backend_spec = resolve_backend_spec(spec.backend_name)
    consumer_partitions = topology.consumer_pipeline_partitions
    transfer_groups = topology.transfer_groups
    align_state_group_ids = frozenset(group.group_id for group in transfer_groups if group.uses_align_state)
    has_align_state = bool(align_state_group_ids)

    if use_layerwise and topology.tp_partition.tp_mismatch:
        raise ProgramCompilationError("Layerwise region projection cannot yet be composed with TP mismatch")
    if topology.tp_partition.tp_mismatch and has_align_state:
        raise ProgramCompilationError("Mamba align-state transfer cannot yet be composed with TP mismatch")
    if topology.tp_partition.tp_mismatch and consumer_partitions is not None and len(consumer_partitions) > 1:
        raise ProgramCompilationError("Consumer pipeline projection cannot yet be composed with TP mismatch")
    if use_layerwise and consumer_partitions is not None and len(consumer_partitions) > 1:
        raise ProgramCompilationError(
            "Layerwise region projection cannot yet be composed with consumer pipeline projection"
        )
    if use_layerwise and has_align_state:
        raise ProgramCompilationError(
            "AscendStore v1 Layerwise transfer does not yet coordinate Mamba align-state copies"
        )
    if use_layerwise and backend_spec.layerwise_access is None:
        raise ProgramCompilationError(
            f"AscendStore v1 Layerwise transfer requires a session Backend; got {spec.backend_name!r}"
        )

    reachability: ReachableRegionSelection
    if len(transfer_groups) == 1 and not has_align_state:
        reachability = UnitaryReachability(
            topology.transfer_group_ids[0],
            spec.max_model_len,
            topology.cache_transfer_granularity,
        )
    else:
        reachability = HybridReachability(
            transfer_groups,
            scheduler_block_size=topology.cache_transfer_granularity,
            hash_block_size=topology.hash_block_size,
            max_model_len=spec.max_model_len,
            use_eagle=spec.use_eagle,
            retention_interval=spec.retention_interval,
        )

    token_database = ChunkedTokenDatabase(
        [group.key_metadata for group in topology.groups],
        [group.block_size for group in topology.groups],
        None,
        topology.hash_block_size,
    )
    transfer_region_projection: TransferRegionProjection
    if use_layerwise:
        transfer_region_projection = LayerwiseRegionProjection(topology)
    elif topology.tp_partition.tp_mismatch:
        transfer_region_projection = StridedRegionProjection(topology)
    else:
        transfer_region_projection = ContiguousRegionProjection(topology)

    region_partition: RegionPartition
    if consumer_partitions is not None and len(consumer_partitions) > 1:
        region_partition = PipelineRegionPartition(consumer_partitions)
    else:
        region_partition = IdentityRegionPartition()

    store_admission: StoreAdmission
    if backend_spec.requires_exists_before_put:
        store_admission = BackendExistenceStoreAdmission()
    else:
        store_admission = UnconditionalStoreAdmission()

    fine_grained_lookup = any(
        group.group_id in align_state_group_ids and group.block_size > topology.hash_block_size
        for group in transfer_groups
    )
    semantic_chunk_projection = SemanticChunkProjection(token_database, transfer_groups, fine_grained_lookup)
    checkpoint_chunk_projection = CheckpointChunkProjection(token_database, transfer_groups, align_state_group_ids)
    local_block_resolution, checkpoint_block_resolution = compile_block_resolutions(topology, align_state_group_ids)
    lookup_rank_counts = {}
    for group in transfer_groups:
        rank_count = topology.tp_partition.key_rank_count
        if group.group_id in align_state_group_ids:
            rank_count = topology.tp_size
        lookup_rank_counts[group.group_id] = rank_count
    return KVPoolProgram(
        topology=topology,
        reachable_region_selection=reachability,
        semantic_chunk_projection=semantic_chunk_projection,
        checkpoint_chunk_projection=checkpoint_chunk_projection,
        remote_object_projection=RemoteObjectProjection(topology, lookup_rank_counts),
        local_block_resolution=local_block_resolution,
        checkpoint_block_resolution=checkpoint_block_resolution,
        store_ownership_selection=compile_store_ownership(topology, align_state_group_ids),
        transfer_region_projection=transfer_region_projection,
        region_partition=region_partition,
        store_admission=store_admission,
        backend_name=spec.backend_name,
        schedule=schedule,
    )
