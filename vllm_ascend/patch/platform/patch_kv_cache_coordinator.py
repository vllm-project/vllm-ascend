from vllm.v1.core.block_pool import BlockPool
import vllm.v1.core.kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
    spec_manager_map,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_ascend.kv_cache_interface import AscendSparseMLAAttentionSpec


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec, **kwargs
) -> SingleTypeKVCacheManager:
    spec_manager_map[AscendSparseMLAAttentionSpec] = FullAttentionManager
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager


# TODO(rjg-lyh): Remove this patch after the kv_cache_spec refactor
# This is only a temporary workaround
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

    self.block_pool = BlockPool(
        kv_cache_config.num_blocks,
        enable_caching,
        hash_block_size,
        enable_kv_cache_events,
        metrics_collector,
    )

    # Needs special handling for find_longest_cache_hit if eagle is enabled
    self.use_eagle = use_eagle
    self.single_type_managers = tuple(
        get_manager_for_kv_cache_spec(
            kv_cache_spec=kv_cache_group.kv_cache_spec,
            block_pool=self.block_pool,
            kv_cache_group_id=i,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
        )
        for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
    )


vllm.v1.core.kv_cache_coordinator.KVCacheCoordinator.__init__ = __init__
