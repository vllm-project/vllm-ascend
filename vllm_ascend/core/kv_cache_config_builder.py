# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend KV cache planning hooks for vLLM's pluggable config builder.

vLLM plans the KV cache in ``vllm.v1.core.kv_cache_planning`` and resolves the
planner through ``Platform.get_kv_cache_config_builder_cls`` (see
``NPUPlatform.get_kv_cache_config_builder_cls``). Ascend-specific grouping and
pool accounting live here; materialization stays in the patched planning
functions (see ``patch_kv_cache_utils.py``), which
``DefaultKVCacheConfigBuilder.get_kv_cache_config_from_groups`` calls with the
Core-resolved ``num_blocks``.

DeepSeek V4's shared-tuple grouping and geometry are the single source of
truth in ``vllm_ascend.models.deepseek_v4.cache_config`` (RFC #21053): this
module and ``patch_kv_cache_utils.py`` both consume it so the two lanes
cannot drift apart.
"""

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    unify_hybrid_kv_cache_specs,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheSpec,
)

from vllm_ascend.models.deepseek_v4.cache_config import (
    _get_kv_cache_groups_uniform_groups,
    group_and_unify_kv_cache_specs,
)
from vllm_ascend.models.deepseek_v4.cache_config import (
    deepseek_v4_pool_bytes_per_block as _deepseek_v4_pool_bytes_per_block,
)
from vllm_ascend.models.deepseek_v4.cache_config import (
    is_deepseek_v4_groups as _is_deepseek_v4_groups,
)
from vllm_ascend.models.deepseek_v41.cache_config import (
    get_deepseek_v41_pool_bytes_per_block,
    group_cache_specs,
    is_deepseek_v41_cache,
    make_cache_groups,
)
from vllm_ascend.models.glm5next.cache_config import (
    _get_glm5_next_cache_layout,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.models.glm5next.kv_cache import is_glm5_next_cache_spec
from vllm_ascend.models.kimi_k3.cache_config import _get_kimi_k3_dspark_mixed_kv_cache_groups
from vllm_ascend.models.kimi_k3.paired_sfa_cache import paired_sfa_cache_groups, paired_sfa_pool_bytes
from vllm_ascend.models.qwen4_exp.cache_config import (
    get_qwen4_exp_pool_bytes_per_block,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache


class AscendKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    """Build Ascend cache layouts with vLLM's cross-worker planning flow."""

    def get_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        """Group layers, preferring the Ascend layout when one applies.

        Precedence: Kimi-K3.1, GLM-Next, then DSV4.1 / mixed-page-size shared slots, then
        the Kimi-K3 DSpark mixed grouping, then vLLM's generic grouping.
        """
        if vllm_config.model_config.hf_config.model_type == "kimi_k31":
            return paired_sfa_cache_groups(vllm_config, kv_cache_spec)
        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            # vLLM's generic grouping promotes local-attention specs before it
            # groups, so the Ascend layouts must see the same promoted specs.
            unify_hybrid_kv_cache_specs(kv_cache_spec)

        if any(is_glm5_next_cache_spec(spec) for spec in kv_cache_spec.values()):
            return get_glm5_next_kv_cache_groups(vllm_config, kv_cache_spec)

        if is_deepseek_v41_cache(kv_cache_spec):
            return make_cache_groups(group_cache_specs(kv_cache_spec))

        grouped_specs = group_and_unify_kv_cache_specs(kv_cache_spec)
        if grouped_specs is not None and len(grouped_specs) >= 2:
            # A uniform spec is not a mixed-page-size layout; grouping it here
            # would drop the C128 bucket the shared-tuple layout depends on.
            return _get_kv_cache_groups_uniform_groups(grouped_specs)

        kimi_k3_groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(kv_cache_spec)
        if kimi_k3_groups is not None:
            return kimi_k3_groups

        return super().get_kv_cache_groups(vllm_config, kv_cache_spec)

    def _get_pool_bytes_per_block(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> int:
        """One global block ID consumes sum(page sizes) x layer tuples.

        The default implementation materializes a one-block pool and requires a
        single device backing size, which the DSV4 shared-tuple, the TurboQuant
        packed, the DSV4.1, and the GLM-Next layouts all violate.
        """
        if vllm_config.model_config.hf_config.model_type == "kimi_k31":
            return paired_sfa_pool_bytes(kv_cache_groups)
        if is_deepseek_v41_cache(kv_cache_groups):
            return get_deepseek_v41_pool_bytes_per_block(kv_cache_groups)
        if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
            return get_glm5_next_pool_bytes_per_block(kv_cache_groups)
        qwen_bytes = get_qwen4_exp_pool_bytes_per_block(kv_cache_groups)
        if qwen_bytes is not None:
            return qwen_bytes
        if tq_cache.uses_turboquant_groups(kv_cache_groups):
            return tq_cache.pool_bytes_per_block(kv_cache_groups)
        if _is_deepseek_v4_groups(kv_cache_groups):
            return _deepseek_v4_pool_bytes_per_block(kv_cache_groups)
        return super()._get_pool_bytes_per_block(vllm_config, kv_cache_groups)
