"""Assemble AscendStore's planner, graph and runtime at process boundaries."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_cache_spec

from ..attention_fence import reset_attention_compute_start_gate
from .backend import BLOCK_KEY_LAYERWISE_BACKENDS
from .execution.io import BackendIO, LayerwiseBackendIO
from .execution.resources import KVPoolResources
from .execution.runtime import KVPoolRuntime
from .execution.timeline import LoadTimelineProtocol, StoreTimelineProtocol
from .execution.timeline.bulk import AsyncLoadTimeline, LoadTimeline, StoreTimeline
from .execution.timeline.layerwise import (
    LayerwiseLoadTimeline,
    LayerwiseStoreTimeline,
    LayerwiseStoreTimelineProtocol,
)
from .graph.filter import BackendExistenceMissingFilter, IdentityMissingFilter, MissingFilter
from .graph.graph import KVPoolGraph
from .graph.projection.binding import (
    BindingProjection,
    ContiguousBindingProjection,
    LayerwiseBindingProjection,
    StridedBindingProjection,
)
from .graph.projection.block import KVBlockProjection
from .graph.projection.chunk import KVChunkProjection
from .graph.projection.consumer import (
    ConsumerProjection,
    IdentityConsumerProjection,
    PipelinePartitionConsumerProjection,
)
from .graph.projection.object import RemoteObjectProjection
from .graph.projection.ownership import StoreOwnershipProjection
from .graph.reachability import HybridReachability, KVReachability, UnitaryReachability
from .graph.topology import KVPoolTopology, resolve_kv_pool_topology
from .planning.availability import RemoteAvailabilityProbe
from .planning.planner import TransferPlanner
from .planning.progress import AllocationLoadPublication, LoadPublication, ScheduledLoadPublication
from .planning.spec import resolve_transfer_planning_spec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


def build_transfer_planner(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, lookup_address: str
) -> TransferPlanner:
    """Assemble the Scheduler-side progress and command planner."""

    spec = resolve_transfer_planning_spec(vllm_config, kv_cache_config)
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    load_publication: LoadPublication
    if extra_config.get("load_async", False) and not extra_config.get("use_layerwise", False):
        load_publication = AllocationLoadPublication()
    else:
        load_publication = ScheduledLoadPublication()
    availability_probe = RemoteAvailabilityProbe(
        lookup_address,
        group_ids=spec.transfer_group_ids,
        transfer_granularity=spec.cache_transfer_granularity,
        discard_partial_chunks=spec.discard_partial_chunks,
        enabled=_is_lookup_enabled(vllm_config),
    )
    return TransferPlanner(
        spec,
        availability_probe,
        load_publication,
        store_enabled=_is_store_enabled(vllm_config),
        save_decode_cache=extra_config.get("save_decode_cache", False),
    )


def build_kv_pool_runtime(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> KVPoolRuntime:
    """Assemble the fixed KV Pool graph and its process-owned runtime."""

    topology = resolve_kv_pool_topology(vllm_config, kv_cache_config)
    transfer_config = vllm_config.kv_transfer_config
    extra_config = transfer_config.kv_connector_extra_config
    use_layerwise = bool(extra_config.get("use_layerwise", False))
    backend_name = extra_config.get("backend", "mooncake").strip().lower()
    store_enabled = _is_store_enabled(vllm_config)
    consumer_partitions = topology.consumer_pipeline_partitions
    layerwise_prefetch_layers = 2
    if use_layerwise:
        configured_prefetch_layers = extra_config.get("layerwise_prefetch_layers", layerwise_prefetch_layers)
        if isinstance(configured_prefetch_layers, bool):
            raise ValueError("layerwise_prefetch_layers must be a positive integer")
        try:
            layerwise_prefetch_layers = int(configured_prefetch_layers)
        except (TypeError, ValueError) as error:
            raise ValueError("layerwise_prefetch_layers must be a positive integer") from error
        if layerwise_prefetch_layers <= 0:
            raise ValueError("layerwise_prefetch_layers must be a positive integer")
    if consumer_partitions is not None and len(consumer_partitions) > 1 and topology.tp_partition.tp_mismatch:
        raise ValueError("Consumer pipeline projection cannot yet be composed with TP mismatch")
    if use_layerwise and topology.tp_partition.tp_mismatch:
        raise ValueError("Layerwise binding projection cannot yet be composed with TP mismatch")
    if use_layerwise and consumer_partitions is not None and len(consumer_partitions) > 1:
        raise ValueError("Layerwise binding projection cannot yet be composed with consumer pipeline projection")
    if use_layerwise and any(group.uses_align_state for group in topology.groups):
        raise ValueError("AscendStore v1 Layerwise transfer does not yet coordinate Mamba align-state copies")
    if use_layerwise and backend_name not in BLOCK_KEY_LAYERWISE_BACKENDS:
        raise ValueError(f"AscendStore v1 Layerwise transfer requires a block-key Backend; got {backend_name!r}")
    resources = KVPoolResources.create(
        vllm_config.parallel_config,
        extra_config,
        topology.groups,
        topology.hash_block_size,
        kv_cache_config.num_blocks,
        topology.consumer_pipeline_partitions,
    )
    reachability = _build_reachability(vllm_config, kv_cache_config, topology)
    binding_projection: BindingProjection
    if use_layerwise:
        binding_projection = LayerwiseBindingProjection(topology)
    elif topology.tp_partition.tp_mismatch:
        binding_projection = StridedBindingProjection(topology)
    else:
        binding_projection = ContiguousBindingProjection(topology)
    consumer_projection: ConsumerProjection
    if consumer_partitions is not None and len(consumer_partitions) > 1:
        consumer_projection = PipelinePartitionConsumerProjection(consumer_partitions)
    else:
        consumer_projection = IdentityConsumerProjection()
    backend_io = LayerwiseBackendIO(resources.backend) if use_layerwise else BackendIO(resources.backend)
    missing_filter: MissingFilter
    if resources.backend.requires_exists_before_put:
        missing_filter = BackendExistenceMissingFilter()
    else:
        missing_filter = IdentityMissingFilter()
    load_timeline: LoadTimelineProtocol
    if use_layerwise:
        assert isinstance(backend_io, LayerwiseBackendIO)
        load_timeline = LayerwiseLoadTimeline(
            topology,
            backend_io,
            layerwise_prefetch_layers,
            backend_io.backend.set_device,
            reset_attention_compute_start_gate,
        )
    elif extra_config.get("load_async", False):
        load_timeline = AsyncLoadTimeline(backend_io.backend.set_device)
    else:
        load_timeline = LoadTimeline()
    store_timeline: StoreTimelineProtocol | LayerwiseStoreTimelineProtocol | None
    if use_layerwise and store_enabled:
        assert isinstance(backend_io, LayerwiseBackendIO)
        store_timeline = LayerwiseStoreTimeline(topology, backend_io, backend_io.backend.set_device)
    elif store_enabled:
        store_timeline = StoreTimeline(backend_io.backend.set_device)
    else:
        store_timeline = None
    graph = KVPoolGraph(
        topology=topology,
        reachability=reachability,
        chunk_projection=KVChunkProjection(resources.token_database, topology),
        remote_object_projection=RemoteObjectProjection(topology),
        block_projection=KVBlockProjection(topology),
        binding_projection=binding_projection,
        store_ownership_projection=StoreOwnershipProjection(topology),
        consumer_projection=consumer_projection,
        missing_filter=missing_filter,
    )
    return KVPoolRuntime(graph, resources, backend_io, load_timeline, store_timeline)


def _build_reachability(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    topology: KVPoolTopology,
) -> KVReachability:
    if len(topology.transfer_group_ids) == 1:
        return UnitaryReachability(
            topology.transfer_group_ids[0],
            vllm_config.model_config.max_model_len,
            topology.cache_transfer_granularity,
        )
    transfer_groups = [
        replace(group, kv_cache_spec=resolve_dcp_kv_cache_spec(group.kv_cache_spec, topology.dcp_size))
        for group in kv_cache_config.transfer_groups
    ]
    return HybridReachability(
        topology.transfer_group_ids,
        transfer_groups,
        scheduler_block_size=topology.cache_transfer_granularity,
        hash_block_size=topology.hash_block_size,
        max_model_len=vllm_config.model_config.max_model_len,
        use_eagle=_uses_eagle_block_drop(vllm_config),
        retention_interval=kv_cache_config.prefix_cache_retention_interval,
    )


def _uses_eagle_block_drop(vllm_config: VllmConfig) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return False
    use_eagle_block_drop = getattr(speculative_config, "use_eagle_block_drop", None)
    if callable(use_eagle_block_drop):
        return bool(use_eagle_block_drop())
    use_eagle = getattr(speculative_config, "use_eagle", None)
    return bool(use_eagle()) if callable(use_eagle) else False


def _is_lookup_enabled(vllm_config: VllmConfig) -> bool:
    transfer_config = vllm_config.kv_transfer_config
    consumer_is_to_load = transfer_config.kv_connector_extra_config.get("consumer_is_to_load", False)
    return transfer_config.kv_role != "kv_consumer" or consumer_is_to_load


def _is_store_enabled(vllm_config: VllmConfig) -> bool:
    transfer_config = vllm_config.kv_transfer_config
    consumer_is_to_put = transfer_config.kv_connector_extra_config.get("consumer_is_to_put", False)
    return transfer_config.kv_role in ("kv_producer", "kv_both") or consumer_is_to_put
