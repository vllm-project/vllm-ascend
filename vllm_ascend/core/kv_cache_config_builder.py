# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical cache-pool accounting for Ascend layouts."""

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.core.kv_cache_physical_plan import get_physical_kv_cache_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_layerwise_reuse_config,
    get_layerwise_reuse_layout,
)
from vllm_ascend.models.glm5next.cache_config import (
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
        plan = get_physical_kv_cache_config(vllm_config, kv_cache_groups, num_blocks)
        if plan is not None:
            # Core has already chosen and converged N across ranks. Keep it
            # exact; fixed alignment overhead was reserved by the worker.
            plan.kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
            return plan
        plan = super().get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks)
        extra = get_layerwise_reuse_config(getattr(vllm_config, "kv_transfer_config", None))
        if extra is not None:
            base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
            layout = get_layerwise_reuse_layout(plan, base_layers, extra)
            # Generic models retain logical capacity accounting: the worker
            # scales the budget and materializes the scratch tensors later.
            # Mark scheduler residency without changing the builder's probe.
            plan.layerwise_kv_cache_scratch = layout.has_layer_reuse and len(layout.layer_cache_specs) >= base_layers
        return plan

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
