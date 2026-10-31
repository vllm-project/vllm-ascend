# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical cache-pool accounting for Ascend layouts."""

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.models.glm5next.cache_config import (
    _build_glm5_next_kv_cache_config,
    _get_glm5_next_cache_layout,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.models.qwen4_exp.cache_config import (
    get_qwen4_exp_pool_bytes_per_block,
)


class AscendKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    def get_kv_cache_config_from_groups(
        self, vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroupSpec], num_blocks: int
    ) -> KVCacheConfig:
        if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
            # Core has already chosen and converged N across ranks. Keep it
            # exact; fixed alignment overhead was reserved by the worker.
            plan = _build_glm5_next_kv_cache_config(vllm_config, kv_cache_groups, num_blocks)
            plan.kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
            return plan
        return super().get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks)

    def _get_pool_bytes_per_block(self, vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroupSpec]) -> int | None:
        # GLM allocates main MLA/KDA and compressed-indexer/tail pages in
        # separate device tensors. The default one-backing probe cannot infer
        # their combined physical block cost from unequal tensor sizes.
        if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
            return get_glm5_next_pool_bytes_per_block(kv_cache_groups, vllm_config)
        qwen_bytes = get_qwen4_exp_pool_bytes_per_block(kv_cache_groups)
        if qwen_bytes is not None:
            return qwen_bytes
        return super()._get_pool_bytes_per_block(vllm_config, kv_cache_groups)
