# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical cache-pool accounting for Ascend layouts."""

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.kv_cache_interface import KVCacheGroupSpec

from vllm_ascend.models.glm5next.cache_config import (
    _get_glm5_next_cache_layout,
    get_glm5_next_pool_bytes_per_block,
)


class AscendKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    def _get_pool_bytes_per_block(self, vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroupSpec]) -> int | None:
        # GLM allocates main MLA/KDA and compressed-indexer/tail pages in
        # separate device tensors. The default one-backing probe cannot infer
        # their combined physical block cost from unequal tensor sizes.
        if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
            return get_glm5_next_pool_bytes_per_block(kv_cache_groups)
        return super()._get_pool_bytes_per_block(vllm_config, kv_cache_groups)
