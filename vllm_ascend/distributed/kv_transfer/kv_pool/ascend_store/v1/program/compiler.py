"""Compile one immutable KV Pool program from vLLM configuration."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_cache_spec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ..backend import resolve_backend_spec
from .program import KVPoolProgram
from .spec.schedule import KVPoolSchedule, LoadScheduleKind, StoreScheduleKind
from .spec.topology import resolve_kv_pool_topology
from .stages.chunk import KVChunkProjection
from .stages.memory import (
    ContiguousBindingProjection,
    KVBlockProjection,
    LayerwiseBindingProjection,
    StridedBindingProjection,
)
from .stages.reachability import HybridReachability, UnitaryReachability
from .stages.remote import RemoteObjectProjection
from .stages.store import (
    BackendExistenceMissingFilter,
    IdentityConsumerProjection,
    IdentityMissingFilter,
    PipelinePartitionConsumerProjection,
    StoreOwnershipProjection,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


class ProgramCompilationError(ValueError):
    """Report an invalid startup-time combination of KV Pool rules."""


def compile_kv_pool_program(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> KVPoolProgram:
    """Resolve every fixed spatial and temporal rule before resources are created."""

    topology = resolve_kv_pool_topology(vllm_config, kv_cache_config)
    transfer_config = vllm_config.kv_transfer_config
    extra_config = transfer_config.kv_connector_extra_config
    use_layerwise = bool(extra_config.get("use_layerwise", False))
    backend_name = extra_config.get("backend", "mooncake").strip().lower()
    backend_spec = resolve_backend_spec(backend_name)
    consumer_partitions = topology.consumer_pipeline_partitions

    if use_layerwise and topology.tp_partition.tp_mismatch:
        raise ProgramCompilationError("Layerwise binding projection cannot yet be composed with TP mismatch")
    if topology.tp_partition.tp_mismatch and consumer_partitions is not None and len(consumer_partitions) > 1:
        raise ProgramCompilationError("Consumer pipeline projection cannot yet be composed with TP mismatch")
    if use_layerwise and consumer_partitions is not None and len(consumer_partitions) > 1:
        raise ProgramCompilationError(
            "Layerwise binding projection cannot yet be composed with consumer pipeline projection"
        )
    if use_layerwise and any(group.uses_align_state for group in topology.groups):
        raise ProgramCompilationError(
            "AscendStore v1 Layerwise transfer does not yet coordinate Mamba align-state copies"
        )
    if use_layerwise and not backend_spec.supports_layerwise:
        raise ProgramCompilationError(
            f"AscendStore v1 Layerwise transfer requires a block-key Backend; got {backend_name!r}"
        )

    if len(topology.transfer_group_ids) == 1:
        reachability = UnitaryReachability(
            topology.transfer_group_ids[0],
            vllm_config.model_config.max_model_len,
            topology.cache_transfer_granularity,
        )
    else:
        transfer_groups = [
            replace(group, kv_cache_spec=resolve_dcp_kv_cache_spec(group.kv_cache_spec, topology.dcp_size))
            for group in kv_cache_config.transfer_groups
        ]
        reachability = HybridReachability(
            topology.transfer_group_ids,
            transfer_groups,
            scheduler_block_size=topology.cache_transfer_granularity,
            hash_block_size=topology.hash_block_size,
            max_model_len=vllm_config.model_config.max_model_len,
            use_eagle=_uses_eagle_block_drop(vllm_config),
            retention_interval=kv_cache_config.prefix_cache_retention_interval,
        )

    token_database = ChunkedTokenDatabase(
        [group.key_metadata for group in topology.groups],
        [group.block_size for group in topology.groups],
        None if consumer_partitions is None else list(consumer_partitions),
        topology.hash_block_size,
    )
    if use_layerwise:
        binding_projection = LayerwiseBindingProjection(topology)
    elif topology.tp_partition.tp_mismatch:
        binding_projection = StridedBindingProjection(topology)
    else:
        binding_projection = ContiguousBindingProjection(topology)

    if consumer_partitions is not None and len(consumer_partitions) > 1:
        consumer_projection = PipelinePartitionConsumerProjection(consumer_partitions)
    else:
        consumer_projection = IdentityConsumerProjection()
    missing_filter = (
        BackendExistenceMissingFilter() if backend_spec.requires_exists_before_put else IdentityMissingFilter()
    )

    if use_layerwise:
        load_schedule_kind = LoadScheduleKind.LAYERWISE
    elif extra_config.get("load_async", False):
        load_schedule_kind = LoadScheduleKind.ASYNC
    else:
        load_schedule_kind = LoadScheduleKind.SYNC
    store_enabled = transfer_config.kv_role in ("kv_producer", "kv_both") or extra_config.get(
        "consumer_is_to_put", False
    )
    store_schedule_kind = None
    if store_enabled:
        store_schedule_kind = StoreScheduleKind.LAYERWISE if use_layerwise else StoreScheduleKind.ASYNC

    return KVPoolProgram(
        topology=topology,
        reachability=reachability,
        chunk_projection=KVChunkProjection(token_database, topology),
        remote_object_projection=RemoteObjectProjection(topology),
        block_projection=KVBlockProjection(topology),
        binding_projection=binding_projection,
        store_ownership_projection=StoreOwnershipProjection(topology),
        consumer_projection=consumer_projection,
        missing_filter=missing_filter,
        backend_name=backend_name,
        schedule=KVPoolSchedule(
            load_schedule_kind,
            store_schedule_kind,
            _resolve_layerwise_prefetch_layers(extra_config, use_layerwise),
        ),
    )


def _resolve_layerwise_prefetch_layers(extra_config: dict[str, Any], use_layerwise: bool) -> int:
    default_prefetch_layers = 2
    if not use_layerwise:
        return default_prefetch_layers
    configured_prefetch_layers = extra_config.get("layerwise_prefetch_layers", default_prefetch_layers)
    if isinstance(configured_prefetch_layers, bool):
        raise ProgramCompilationError("layerwise_prefetch_layers must be a positive integer")
    try:
        prefetch_layers = int(configured_prefetch_layers)
    except (TypeError, ValueError) as error:
        raise ProgramCompilationError("layerwise_prefetch_layers must be a positive integer") from error
    if prefetch_layers <= 0:
        raise ProgramCompilationError("layerwise_prefetch_layers must be a positive integer")
    return prefetch_layers


def _uses_eagle_block_drop(vllm_config: VllmConfig) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return False
    use_eagle_block_drop = getattr(speculative_config, "use_eagle_block_drop", None)
    if callable(use_eagle_block_drop):
        return bool(use_eagle_block_drop())
    use_eagle = getattr(speculative_config, "use_eagle", None)
    return bool(use_eagle()) if callable(use_eagle) else False
