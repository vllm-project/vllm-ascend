"""Construct AscendStore v1 services from static configuration."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

from .scheduler.layout import resolve_scheduler_transfer_layout
from .scheduler.load import DeferredLoadScheduling, ImmediateLoadScheduling
from .scheduler.load import LoadService as SchedulerLoadService
from .scheduler.lookup import LookupService as SchedulerLookupService
from .scheduler.service import SchedulerService
from .scheduler.store import StoreService as SchedulerStoreService
from .worker.layout import resolve_worker_transfer_layout
from .worker.load.async_executor import AsyncLoadExecutor
from .worker.load.executor import LoadExecutor
from .worker.load.service import LoadService as WorkerLoadService
from .worker.lookup import LookupService as WorkerLookupService
from .worker.resources import WorkerCacheResources
from .worker.service import WorkerService
from .worker.store import StoreService as WorkerStoreService

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
    load_execution_mode = _resolve_load_execution_mode(vllm_config)
    load_executor_type = AsyncLoadExecutor if load_execution_mode is LoadExecutionMode.ASYNCHRONOUS else LoadExecutor
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    transfer_config = vllm_config.kv_transfer_config
    extra_config = transfer_config.kv_connector_extra_config
    cache_resources = WorkerCacheResources.create(
        parallel_config,
        extra_config,
        layout.key_metadata,
        layout.block_size,
        layout.hash_block_size,
        kv_cache_config.num_blocks,
    )
    backend = cache_resources.backend
    token_database = cache_resources.token_database
    lookup_service = WorkerLookupService(
        backend,
        token_database,
        layout.tp_size,
        layout.pp_size,
        layout.dcp_size,
        layout.num_kv_heads,
        model_config.max_model_len,
        layout.block_size,
    )
    load_service = WorkerLoadService(
        token_database,
        layout.block_size,
        layout.block_size,
        layout.tp_rank,
        load_executor_type(backend),
    )
    store_service = None
    if _is_store_enabled(vllm_config):
        store_service = WorkerStoreService(
            backend,
            token_database,
            layout.block_size,
            layout.tp_rank,
            layout.pcp_rank,
            layout.pcp_size,
            layout.dcp_size,
            layout.put_step,
            transfer_config.kv_role,
        )
    return WorkerService(cache_resources, lookup_service, load_service, store_service)


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
