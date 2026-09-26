"""Construct AscendStore v1 services from static configuration."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum, auto
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_cache_spec

from .scheduler.layout import resolve_scheduler_transfer_layout
from .scheduler.load import DeferredLoadScheduling, ImmediateLoadScheduling
from .scheduler.load import LoadService as SchedulerLoadService
from .scheduler.lookup import LookupService as SchedulerLookupService
from .scheduler.service import SchedulerService
from .scheduler.store import StoreService as SchedulerStoreService
from .worker.coordinator import (
    HybridKVTransferCoordinator,
    KVTransferCoordinator,
    UnitaryKVTransferCoordinator,
)
from .worker.layout import (
    KVCacheGroupLayout,
    StridedKVPartitioner,
    WorkerTransferLayout,
    resolve_worker_transfer_layout,
)
from .worker.load.async_executor import AsyncLoadExecutor
from .worker.load.executor import SynchronousLoadExecutor
from .worker.load.service import LoadService as WorkerLoadService
from .worker.load.task import ContiguousLoadTaskBuilder, StridedLoadTaskBuilder
from .worker.lookup import LookupService as WorkerLookupService
from .worker.lookup.executor import LookupExecutor
from .worker.lookup.task import LookupTaskBuilder
from .worker.resources import WorkerCacheResources
from .worker.service import WorkerService
from .worker.store import StoreService as WorkerStoreService
from .worker.store.executor import StoreExecutor
from .worker.store.task import ContiguousStoreTaskBuilder, StridedStoreTaskBuilder

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


class LoadExecutionMode(Enum):
    """Static Load execution choice shared by Scheduler and Worker assembly."""

    SYNCHRONOUS = auto()
    ASYNCHRONOUS = auto()


def build_scheduler_service(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, lookup_address: str
) -> SchedulerService:
    layout = resolve_scheduler_transfer_layout(vllm_config, kv_cache_config)
    load_execution_mode = _resolve_load_execution_mode(vllm_config)
    if load_execution_mode is LoadExecutionMode.ASYNCHRONOUS:
        load_scheduling = DeferredLoadScheduling()
    else:
        load_scheduling = ImmediateLoadScheduling()
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    lookup_service = SchedulerLookupService(
        lookup_address,
        transfer_group_ids=layout.transfer_group_ids,
        cache_transfer_granularity=layout.cache_transfer_granularity,
        discard_partial_chunks=layout.discard_partial_chunks,
        enabled=_is_lookup_enabled(vllm_config),
    )
    load_service = SchedulerLoadService(load_scheduling)
    store_service = SchedulerStoreService(
        cache_transfer_granularity=layout.cache_transfer_granularity,
        discard_partial_chunks=layout.discard_partial_chunks,
        save_decode_cache=extra_config.get("save_decode_cache", False),
        enabled=_is_store_enabled(vllm_config),
    )
    return SchedulerService(layout, lookup_service, load_service, store_service)


def build_worker_service(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> WorkerService:
    layout = resolve_worker_transfer_layout(vllm_config, kv_cache_config)
    parallel_config = vllm_config.parallel_config
    transfer_config = vllm_config.kv_transfer_config
    extra_config = transfer_config.kv_connector_extra_config
    cache_resources = WorkerCacheResources.create(
        parallel_config,
        extra_config,
        layout.kv_cache_groups,
        layout.hash_block_size,
        kv_cache_config.num_blocks,
    )
    kv_cache_group = layout.kv_cache_groups[layout.transfer_group_ids[0]]
    kv_partitioner = _build_strided_kv_partitioner(cache_resources, layout, kv_cache_group)
    coordinator = _build_kv_transfer_coordinator(vllm_config, kv_cache_config, layout)
    lookup_service = _build_worker_lookup_service(cache_resources, layout, coordinator)
    load_service = _build_worker_load_service(
        cache_resources,
        layout,
        kv_cache_group,
        kv_partitioner,
        _resolve_load_execution_mode(vllm_config),
        coordinator,
    )
    store_service = _build_worker_store_service(
        cache_resources,
        layout,
        kv_cache_group,
        kv_partitioner,
        transfer_config.kv_role,
        _is_store_enabled(vllm_config),
        coordinator,
    )
    return WorkerService(cache_resources, lookup_service, load_service, store_service)


def _build_strided_kv_partitioner(
    cache_resources: WorkerCacheResources,
    layout: WorkerTransferLayout,
    kv_cache_group: KVCacheGroupLayout,
) -> StridedKVPartitioner | None:
    if not layout.tp_partition.tp_mismatch:
        return None
    if len(layout.transfer_group_ids) != 1:
        raise ValueError("AscendStore v1 TP mismatch requires one transferable KV cache group")
    return StridedKVPartitioner(
        cache_resources.token_database,
        kv_cache_group.block_size,
        layout.tp_rank,
        layout.tp_partition.key_slices_per_rank,
    )


def _build_worker_lookup_service(
    cache_resources: WorkerCacheResources,
    layout: WorkerTransferLayout,
    coordinator: KVTransferCoordinator,
) -> WorkerLookupService:
    task_builder = LookupTaskBuilder(
        cache_resources.token_database,
        layout.tp_partition.key_rank_count,
        layout.pp_size,
        layout.dcp_size,
    )
    return WorkerLookupService(coordinator, task_builder, LookupExecutor(cache_resources.backend))


def _build_kv_transfer_coordinator(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    layout: WorkerTransferLayout,
) -> KVTransferCoordinator:
    layouts_by_group = {group.group_id: group for group in layout.kv_cache_groups}
    if len(layout.transfer_group_ids) == 1:
        group_id = layout.transfer_group_ids[0]
        return UnitaryKVTransferCoordinator(
            group_id,
            layouts_by_group[group_id].block_size,
            vllm_config.model_config.max_model_len,
            layout.cache_transfer_granularity,
        )

    transfer_groups = [
        replace(group, kv_cache_spec=resolve_dcp_kv_cache_spec(group.kv_cache_spec, layout.dcp_size))
        for group in kv_cache_config.transfer_groups
    ]
    return HybridKVTransferCoordinator(
        layout.transfer_group_ids,
        transfer_groups,
        scheduler_block_size=layout.cache_transfer_granularity,
        hash_block_size=layout.hash_block_size,
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


def _build_worker_load_service(
    cache_resources: WorkerCacheResources,
    layout: WorkerTransferLayout,
    kv_cache_group: KVCacheGroupLayout,
    kv_partitioner: StridedKVPartitioner | None,
    execution_mode: LoadExecutionMode,
    coordinator: KVTransferCoordinator,
) -> WorkerLoadService:
    if kv_partitioner is None:
        task_builder = ContiguousLoadTaskBuilder(
            cache_resources.token_database,
            {group.group_id: group.block_size for group in layout.kv_cache_groups},
            layout.tp_rank,
        )
    else:
        task_builder = StridedLoadTaskBuilder(
            cache_resources.token_database,
            kv_cache_group.block_size,
            layout.cache_transfer_granularity,
            kv_partitioner,
        )
    executor_type = AsyncLoadExecutor if execution_mode is LoadExecutionMode.ASYNCHRONOUS else SynchronousLoadExecutor
    return WorkerLoadService(
        coordinator,
        task_builder,
        executor_type(cache_resources.backend),
        layout.cache_transfer_granularity,
        uses_group_scoped_block_ids=len(layout.kv_cache_groups) > 1,
    )


def _build_worker_store_service(
    cache_resources: WorkerCacheResources,
    layout: WorkerTransferLayout,
    kv_cache_group: KVCacheGroupLayout,
    kv_partitioner: StridedKVPartitioner | None,
    kv_role: str,
    enabled: bool,
    coordinator: KVTransferCoordinator,
) -> WorkerStoreService | None:
    if not enabled:
        return None
    if kv_partitioner is None:
        task_builder = ContiguousStoreTaskBuilder(
            cache_resources.token_database,
            {group.group_id: group.block_size for group in layout.kv_cache_groups},
            layout.tp_rank,
            layout.pcp_rank,
            layout.pcp_size,
            layout.dcp_size,
            layout.put_step,
            kv_role,
        )
    else:
        task_builder = StridedStoreTaskBuilder(
            cache_resources.token_database,
            kv_cache_group.block_size,
            layout.pcp_rank,
            layout.pcp_size,
            kv_partitioner,
        )
    return WorkerStoreService(coordinator, task_builder, StoreExecutor(cache_resources.backend))


def _resolve_load_execution_mode(vllm_config: VllmConfig) -> LoadExecutionMode:
    load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
    return LoadExecutionMode.ASYNCHRONOUS if load_async else LoadExecutionMode.SYNCHRONOUS


def _is_lookup_enabled(vllm_config: VllmConfig) -> bool:
    transfer_config = vllm_config.kv_transfer_config
    consumer_is_to_load = transfer_config.kv_connector_extra_config.get("consumer_is_to_load", False)
    return transfer_config.kv_role != "kv_consumer" or consumer_is_to_load


def _is_store_enabled(vllm_config: VllmConfig) -> bool:
    transfer_config = vllm_config.kv_transfer_config
    consumer_is_to_put = transfer_config.kv_connector_extra_config.get("consumer_is_to_put", False)
    return transfer_config.kv_role in ("kv_producer", "kv_both") or consumer_is_to_put
