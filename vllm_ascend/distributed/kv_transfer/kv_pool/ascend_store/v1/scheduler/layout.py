"""Static transfer layout shared by Scheduler operation services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import vllm.v1.core.kv_cache_utils as kv_cache_utils

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True, slots=True)
class SchedulerTransferLayout:
    cache_transfer_granularity: int
    hash_block_size: int
    transfer_group_ids: tuple[int, ...]
    discard_partial_chunks: bool


def resolve_scheduler_transfer_layout(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> SchedulerTransferLayout:
    cache_transfer_granularity, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    discard_partial_chunks = extra_config.get("discard_partial_chunks", True)
    transfer_group_ids = tuple(
        getattr(kv_cache_config, "transfer_group_ids", range(len(kv_cache_config.kv_cache_groups)))
    )
    return SchedulerTransferLayout(
        cache_transfer_granularity,
        hash_block_size,
        transfer_group_ids,
        discard_partial_chunks,
    )
