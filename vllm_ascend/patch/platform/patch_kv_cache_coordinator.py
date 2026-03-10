# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm
from vllm.v1.core.kv_cache_coordinator import (HybridKVCacheCoordinator,
                                               KVCacheCoordinator,
                                               KVCacheCoordinatorNoPrefixCache,
                                               UnitaryKVCacheCoordinator)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import \
    get_manager_for_kv_cache_spec
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.core.multi_block_pool import MultiBlockPool

USE_MULTI_BLOCK_POOL = False


class KVCacheCoordinatorWithMultiPool(KVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    To simplify `find_longest_cache_hit`, it only supports the combination of
    two types of KV cache groups, and one of them must be full attention.
    May extend to more general cases in the future.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[
            0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size

        # Needs special handling for find_longest_cache_hit if eagle is enabled
        self.use_eagle = use_eagle
        cache_num_blocks = [
            kv_cache_config.num_blocks for _ in kv_cache_config.kv_cache_groups
        ]
        self.block_pool = MultiBlockPool(
            cache_num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
            ) for i, kv_cache_group in enumerate(
                self.kv_cache_config.kv_cache_groups))
        self.num_single_type_manager = len(self.single_type_managers)

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager))
        return blocks, 0


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    if USE_MULTI_BLOCK_POOL:
        return KVCacheCoordinatorWithMultiPool(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
    )


vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator = get_kv_cache_coordinator
