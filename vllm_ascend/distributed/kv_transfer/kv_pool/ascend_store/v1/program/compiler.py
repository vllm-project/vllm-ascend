"""Compile one immutable KV Pool program from resolved startup facts."""

from __future__ import annotations

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ..backend import resolve_backend_spec
from .program import KVPoolProgram
from .spec.compilation import KVPoolCompilationSpec
from .stages.admission import BackendExistenceStoreAdmission, UnconditionalStoreAdmission
from .stages.block import compile_block_resolutions
from .stages.chunk import CheckpointChunkProjection, SemanticChunkProjection
from .stages.ownership import compile_store_ownership
from .stages.partition import IdentityRegionPartition, PipelineRegionPartition
from .stages.reachability import HybridReachability, UnitaryReachability
from .stages.region import (
    ContiguousRegionProjection,
    LayerwiseRegionProjection,
    StridedRegionProjection,
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

    if use_layerwise and topology.tp_partition.tp_mismatch:
        raise ProgramCompilationError("Layerwise region projection cannot yet be composed with TP mismatch")
    if topology.tp_partition.tp_mismatch and any(group.uses_align_state for group in topology.groups):
        raise ProgramCompilationError("Mamba align-state transfer cannot yet be composed with TP mismatch")
    if topology.tp_partition.tp_mismatch and consumer_partitions is not None and len(consumer_partitions) > 1:
        raise ProgramCompilationError("Consumer pipeline projection cannot yet be composed with TP mismatch")
    if use_layerwise and consumer_partitions is not None and len(consumer_partitions) > 1:
        raise ProgramCompilationError(
            "Layerwise region projection cannot yet be composed with consumer pipeline projection"
        )
    if use_layerwise and any(group.uses_align_state for group in topology.groups):
        raise ProgramCompilationError(
            "AscendStore v1 Layerwise transfer does not yet coordinate Mamba align-state copies"
        )
    if use_layerwise and not backend_spec.supports_layerwise:
        raise ProgramCompilationError(
            f"AscendStore v1 Layerwise transfer requires a block-key Backend; got {spec.backend_name!r}"
        )

    if len(topology.transfer_group_ids) == 1 and not any(group.uses_align_state for group in topology.groups):
        reachability = UnitaryReachability(
            topology.transfer_group_ids[0],
            spec.max_model_len,
            topology.cache_transfer_granularity,
        )
    else:
        reachability_group_ids = tuple(group.group_id for group in spec.reachability_groups)
        if reachability_group_ids != topology.transfer_group_ids:
            raise ProgramCompilationError(
                f"Reachability groups {reachability_group_ids} do not match transfer groups "
                f"{topology.transfer_group_ids}"
            )
        reachability = HybridReachability(
            spec.reachability_groups,
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
    if use_layerwise:
        transfer_region_projection = LayerwiseRegionProjection(topology)
    elif topology.tp_partition.tp_mismatch:
        transfer_region_projection = StridedRegionProjection(topology)
    else:
        transfer_region_projection = ContiguousRegionProjection(topology)

    if consumer_partitions is not None and len(consumer_partitions) > 1:
        region_partition = PipelineRegionPartition(consumer_partitions)
    else:
        region_partition = IdentityRegionPartition()
    store_admission = (
        BackendExistenceStoreAdmission() if backend_spec.requires_exists_before_put else UnconditionalStoreAdmission()
    )

    semantic_chunk_projection = SemanticChunkProjection(token_database, topology)
    checkpoint_chunk_projection = CheckpointChunkProjection(token_database, topology)
    local_block_resolution, checkpoint_block_resolution = compile_block_resolutions(topology)
    return KVPoolProgram(
        topology=topology,
        reachable_region_selection=reachability,
        semantic_chunk_projection=semantic_chunk_projection,
        checkpoint_chunk_projection=checkpoint_chunk_projection,
        remote_object_projection=RemoteObjectProjection(topology),
        local_block_resolution=local_block_resolution,
        checkpoint_block_resolution=checkpoint_block_resolution,
        store_ownership_selection=compile_store_ownership(topology),
        transfer_region_projection=transfer_region_projection,
        region_partition=region_partition,
        store_admission=store_admission,
        backend_name=spec.backend_name,
        schedule=schedule,
    )
