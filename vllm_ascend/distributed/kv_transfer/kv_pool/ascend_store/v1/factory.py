"""Assemble Scheduler and Worker owners from admitted AscendStore configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import vllm.envs as vllm_envs
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import uses_hybrid_kv_cache

from .backend import LayerwiseAccessKind, resolve_backend_spec
from .infrastructure.configuration.naming import resolve_key_naming_config
from .infrastructure.configuration.resolve import resolve_kv_pool_config, resolve_worker_topology
from .infrastructure.coordinator import Coordinator
from .infrastructure.protocol.rpc import LookupClient
from .infrastructure.resources import KVPoolResources
from .scheduler import (
    AsynchronousBulkScheduler,
    KVPoolScheduler,
    LayerwiseScheduler,
    SchedulerConfig,
    SynchronousBulkScheduler,
)
from .worker.base import KVPoolWorker
from .worker.bulk import BulkWorker
from .worker.layerwise import GVALayerwiseWorker, KeyRangeLayerwiseWorker
from .worker.layerwise.gva.collective import StoreCollective

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


def create_kv_pool_scheduler(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, lookup_address: str
) -> KVPoolScheduler:
    config = resolve_kv_pool_config(vllm_config, kv_cache_config)
    store_group_specs = (
        tuple(
            (
                group_id,
                config.group_specs[group_id],
                bool(getattr(kv_cache_config.kv_cache_groups[group_id], "is_eagle_group", False)),
            )
            for group_id in config.transfer_group_ids
        )
        if config.store_enabled and not config.use_layerwise
        else ()
    )
    scheduler_config = SchedulerConfig(
        cache_transfer_granularity=config.cache_transfer_granularity,
        hash_block_size=config.hash_block_size,
        transfer_group_ids=config.transfer_group_ids,
        align_state_group_ids=config.align_state_group_ids.intersection(config.transfer_group_ids),
        load_enabled=config.load_enabled,
        store_enabled=config.store_enabled,
        save_decode_cache=config.save_decode_cache,
        use_eagle_block_drop=config.use_eagle,
        has_private_state=config.has_private_state,
        expected_worker_count=vllm_config.parallel_config.world_size,
        store_group_specs=store_group_specs,
        retention_interval=config.retention_interval,
    )
    scheduler_type: type[KVPoolScheduler]
    if config.use_layerwise:
        scheduler_type = LayerwiseScheduler
    elif config.load_async:
        scheduler_type = AsynchronousBulkScheduler
    else:
        scheduler_type = SynchronousBulkScheduler
    return scheduler_type(scheduler_config, LookupClient(lookup_address, timeout_ms=config.lookup_rpc_timeout_ms))


def create_kv_pool_worker(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> KVPoolWorker:
    """Acquire resources only after common configuration and local topology are resolved."""
    config = resolve_kv_pool_config(vllm_config, kv_cache_config)
    topology = resolve_worker_topology(vllm_config, kv_cache_config, config)
    coordinator = Coordinator(
        topology.transfer_groups,
        topology.cache_transfer_granularity,
        topology.hash_block_size,
        vllm_config.model_config.max_model_len,
        use_eagle=config.use_eagle,
        retention_interval=config.retention_interval,
    )
    backend_spec = resolve_backend_spec(config.backend_name)
    if config.use_layerwise and backend_spec.layerwise_access is None:
        raise TypeError("Layerwise route requires a concrete Backend data plane")
    key_builder = None
    if config.use_layerwise and backend_spec.layerwise_key_factory is not None:
        naming_config = resolve_key_naming_config(
            vllm_config,
            kv_cache_config,
            model_name=topology.transfer_groups[0].key_metadata.model_name,
            use_hybrid=uses_hybrid_kv_cache(vllm_config.scheduler_config, kv_cache_config.kv_cache_groups)
            or len(kv_cache_config.kv_cache_groups) > 1,
            group_block_sizes=tuple(spec.block_size for spec in config.group_specs),
            requires_layout_identity=backend_spec.key_requires_layout_identity,
        )
        key_builder = backend_spec.layerwise_key_factory(naming_config)
    align_shared_storage = (
        config.use_layerwise
        and backend_spec.layerwise_access is LayerwiseAccessKind.GVA
        and uses_hybrid_kv_cache(vllm_config.scheduler_config, kv_cache_config.kv_cache_groups)
    )
    # Backend-private options remain owned by the selected Backend, not the common resolver.
    backend = backend_spec.create(
        vllm_config.parallel_config,
        extra_config=vllm_config.kv_transfer_config.kv_connector_extra_config,
    )
    resources: KVPoolResources | None = None
    try:
        resources = KVPoolResources(
            backend,
            backend_spec,
            topology.transfer_groups,
            align_shared_storage=align_shared_storage,
        )
        if config.use_layerwise:
            if backend_spec.layerwise_access is LayerwiseAccessKind.GVA:
                collective = None
                if config.store_enabled and topology.cp_partition.object_shard_count > 1:
                    collective = StoreCollective(topology.context_global_ranks)
                    resources.bind_execution_cleanup(collective.close)
                return GVALayerwiseWorker(
                    topology,
                    resources,
                    store_enabled=config.store_enabled,
                    coordinator=coordinator,
                    kv_cache_layout=vllm_envs.VLLM_KV_CACHE_LAYOUT,
                    layerwise_prefetch_layers=config.layerwise_prefetch_layers,
                    key_builder=key_builder,
                    store_collective=collective,
                )
            return KeyRangeLayerwiseWorker(
                topology,
                resources,
                store_enabled=config.store_enabled,
                coordinator=coordinator,
                kv_cache_layout=vllm_envs.VLLM_KV_CACHE_LAYOUT,
                layerwise_prefetch_layers=config.layerwise_prefetch_layers,
                key_builder=key_builder,
            )
        return BulkWorker(
            topology,
            resources,
            coordinator=coordinator,
            kv_cache_layout=vllm_envs.VLLM_KV_CACHE_LAYOUT,
            store_enabled=config.store_enabled,
            load_async=config.load_async,
        )
    except BaseException:
        try:
            if resources is not None:
                resources.close()
            else:
                backend.close()
        except BaseException:
            logger.exception("Failed to close KV Pool Backend after Worker initialization failed")
        raise
