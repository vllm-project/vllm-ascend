"""Assemble AscendStore's domain graph at process boundaries."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_cache_spec

from .execution.io import BackendExistenceMissingFilter, BackendIO, IdentityMissingFilter, MissingFilter
from .execution.resources import KVResources
from .execution.timeline import AsynchronousLoadTimeline, LoadTimeline, StoreTimeline, SynchronousLoadTimeline
from .graph.projection import (
    BindingProjection,
    ConsumerProjection,
    ContiguousBindingProjection,
    IdentityConsumerProjection,
    KVProjection,
    PipelinePartitionConsumerProjection,
    StridedBindingProjection,
)
from .graph.reachability import HybridReachability, KVReachability, UnitaryReachability
from .graph.topology import KVTopology, resolve_kv_topology
from .kv_pool import KVPoolGraph
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
    if extra_config.get("load_async", False):
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


def build_kv_pool_graph(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> KVPoolGraph:
    """Assemble the fixed Lookup, Load and Store graph for one KV Pool participant."""

    topology = resolve_kv_topology(vllm_config, kv_cache_config)
    transfer_config = vllm_config.kv_transfer_config
    extra_config = transfer_config.kv_connector_extra_config
    consumer_partitions = topology.consumer_pipeline_partitions
    if consumer_partitions is not None and len(consumer_partitions) > 1 and topology.tp_partition.tp_mismatch:
        raise ValueError("Consumer pipeline projection cannot yet be composed with TP mismatch")
    resources = KVResources.create(
        vllm_config.parallel_config,
        extra_config,
        topology.kv_cache_groups,
        topology.hash_block_size,
        kv_cache_config.num_blocks,
        topology.consumer_pipeline_partitions,
    )
    reachability = _build_reachability(vllm_config, kv_cache_config, topology)
    binding_projection: BindingProjection
    if topology.tp_partition.tp_mismatch:
        binding_projection = StridedBindingProjection(resources.token_database, topology)
    else:
        binding_projection = ContiguousBindingProjection(resources.token_database)
    projection = KVProjection(resources.token_database, topology, binding_projection)
    consumer_projection: ConsumerProjection
    if consumer_partitions is not None and len(consumer_partitions) > 1:
        consumer_projection = PipelinePartitionConsumerProjection(resources.token_database, consumer_partitions)
    else:
        consumer_projection = IdentityConsumerProjection()
    backend_io = BackendIO(resources.backend)
    missing_filter: MissingFilter
    if resources.backend.requires_exists_before_put:
        missing_filter = BackendExistenceMissingFilter(resources.backend)
    else:
        missing_filter = IdentityMissingFilter()
    load_timeline: LoadTimeline
    if extra_config.get("load_async", False):
        load_timeline = AsynchronousLoadTimeline(backend_io.backend.set_device)
    else:
        load_timeline = SynchronousLoadTimeline()
    store_timeline = StoreTimeline(backend_io.backend.set_device) if _is_store_enabled(vllm_config) else None
    return KVPoolGraph(
        resources,
        topology,
        reachability,
        projection,
        consumer_projection,
        missing_filter,
        backend_io,
        load_timeline,
        store_timeline,
    )


def _build_reachability(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    topology: KVTopology,
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
