"""Static transfer layout shared by Scheduler operation services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import infer_group_block_sizes

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True, slots=True)
class SchedulerTransferLayout:
    cache_transfer_granularity: int
    hash_block_size: int
    discard_partial_chunks: bool


def resolve_scheduler_transfer_layout(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> SchedulerTransferLayout:
    dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)
    group_block_sizes = infer_group_block_sizes(vllm_config.cache_config.block_size, kv_cache_config.kv_cache_groups)
    original_block_size = group_block_sizes[0]
    cache_transfer_granularity = original_block_size * dcp_size
    requested_hash_block_size = vllm_config.cache_config.prefix_match_unit
    if isinstance(requested_hash_block_size, int):
        base_hash_block_size = requested_hash_block_size
    else:
        base_hash_block_size = original_block_size
    hash_block_size = base_hash_block_size * dcp_size
    if cache_transfer_granularity % hash_block_size != 0:
        raise ValueError("block_size must be divisible by hash_block_size")

    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    discard_partial_chunks = extra_config.get("discard_partial_chunks", True)
    return SchedulerTransferLayout(cache_transfer_granularity, hash_block_size, discard_partial_chunks)
