# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter, defaultdict
from dataclasses import replace

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    _approximate_gcd,
    _pool_bytes_per_block,
    unify_hybrid_kv_cache_specs,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroup,
    KVCacheLayerConfig,
    MLAAttentionSpec,
)

from vllm_ascend.core.kv_cache_layers import configure_cache_transfer
from vllm_ascend.models.deepseek_v4.cache_config import _get_deepseek_v4_cache_layout, _place_deepseek_v4_cache
from vllm_ascend.models.deepseek_v41.cache_config import (
    get_deepseek_v41_kv_cache_config,
    get_deepseek_v41_pool_bytes_per_block,
    group_layer_configs,
    is_deepseek_v41_cache,
)
from vllm_ascend.models.glm5next.cache_config import (
    _get_glm5_next_cache_layout,
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.models.glm5next.kv_cache import is_glm5_next_cache_spec
from vllm_ascend.models.kimi_k3.cache_config import _get_kimi_k3_dspark_mixed_kv_cache_groups
from vllm_ascend.models.kimi_k3.paired_sfa_cache import (
    paired_sfa_cache_config,
    paired_sfa_cache_groups,
    paired_sfa_pool_bytes,
)
from vllm_ascend.models.qwen4_exp.cache_config import (
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_kv_cache_groups,
    get_qwen4_exp_pool_bytes_per_block,
)
from vllm_ascend.models.step5.cache_config import (
    get_step5_kv_cache_config_from_groups,
    get_step5_kv_cache_groups,
    get_step5_pool_bytes_per_block,
    is_step5_groups,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import TURBOQUANT_CACHE_DTYPE
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache


def _is_deepseek_v4(specs):
    return any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in specs)


class AscendKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    """Apply Ascend layouts with native block-table groups and manager policy."""

    def get_kv_cache_groups(
        self, vllm_config: VllmConfig, layer_configs: dict[str, KVCacheLayerConfig]
    ) -> list[KVCacheGroup]:
        speculative = getattr(vllm_config, "speculative_config", None)
        drafts = (
            {name for name, config in layer_configs.items() if config.is_draft}
            if speculative is not None and speculative.method == "dspark"
            else None
        )
        if vllm_config.model_config.hf_config.model_type == "kimi_k31":
            groups = paired_sfa_cache_groups(vllm_config, layer_configs)
        else:
            if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
                unify_hybrid_kv_cache_specs(layer_configs)
            specs = [config.spec for config in layer_configs.values()]
            if any(is_glm5_next_cache_spec(spec) for spec in specs):
                groups = get_glm5_next_kv_cache_groups(vllm_config, layer_configs)
            elif is_deepseek_v41_cache(specs):
                groups = group_layer_configs(layer_configs)
            elif _is_deepseek_v4(specs):
                groups = _group_deepseek_v4_layers(layer_configs)
            else:
                default_groups = super().get_kv_cache_groups
                groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(layer_configs)
                if groups is None:
                    groups = get_step5_kv_cache_groups(vllm_config, layer_configs, default_groups)
                if groups is None:
                    groups = get_qwen4_exp_kv_cache_groups(vllm_config, layer_configs, default_groups)
                if groups is None:
                    groups = default_groups(vllm_config, layer_configs)
        if drafts is not None:
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_layout import (
                draft_only_group_ids,
            )

            # PP projection discards remote names. The prefix store needs the global IDs.
            vllm_config.dspark_draft_only_group_ids = draft_only_group_ids(groups, drafts)
        return groups

    def get_kv_cache_config_from_groups(
        self, vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroup], num_blocks: int
    ) -> KVCacheConfig:
        config = self._place(vllm_config, kv_cache_groups, num_blocks)
        configure_cache_transfer(vllm_config, config)
        return config

    def _place(self, vllm_config, groups, num_blocks):
        if vllm_config.model_config.hf_config.model_type == "kimi_k31":
            return paired_sfa_cache_config(vllm_config, groups, num_blocks=num_blocks)
        if is_step5_groups(groups):
            return get_step5_kv_cache_config_from_groups(vllm_config, groups, num_blocks=num_blocks)
        pools = _replicated_draft_pools(groups)
        if pools:
            sizes = [_pool_bytes_per_block(vllm_config, pool) * num_blocks for pool in pools]
            offset = 0
            tensors = []
            for pool, size in zip(pools, sizes):
                placement = super().get_kv_cache_config_from_groups(vllm_config, pool, num_blocks)
                tensors.extend(
                    replace(t, size=sum(sizes), offset=offset + t.offset) for t in placement.kv_cache_tensors
                )
                offset += size
        else:
            qwen = get_qwen4_exp_kv_cache_config(vllm_config, groups, num_blocks=num_blocks)
            if qwen is not None:
                return qwen
            if is_deepseek_v41_cache(groups):
                return get_deepseek_v41_kv_cache_config(vllm_config, groups, num_blocks=num_blocks)
            if _get_glm5_next_cache_layout(groups) is not None:
                return get_glm5_next_kv_cache_config(vllm_config, groups, num_blocks=num_blocks)
            if tq_cache.uses_turboquant_groups(groups):
                _, tensors = tq_cache.cache_config(vllm_config, groups, num_blocks=num_blocks)
            elif _is_deepseek_v4(s for g in groups for s in g.distinct_specs):
                tensors = _place_deepseek_v4_cache(groups, num_blocks)
            else:
                return super().get_kv_cache_config_from_groups(vllm_config, groups, num_blocks)
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=tensors,
            kv_cache_groups=groups,
            prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
        )

    def _get_pool_bytes_per_block(self, vllm_config, groups):
        if vllm_config.model_config.hf_config.model_type == "kimi_k31":
            return paired_sfa_pool_bytes(groups)
        if is_step5_groups(groups):
            return get_step5_pool_bytes_per_block(groups)
        pools = _replicated_draft_pools(groups)
        if pools:
            return sum(_pool_bytes_per_block(vllm_config, pool) for pool in pools)
        if is_deepseek_v41_cache(groups):
            return get_deepseek_v41_pool_bytes_per_block(groups)
        if _get_glm5_next_cache_layout(groups) is not None:
            return get_glm5_next_pool_bytes_per_block(groups)
        if tq_cache.uses_turboquant_groups(groups):
            return tq_cache.pool_bytes_per_block(groups)
        if _is_deepseek_v4(s for g in groups for s in g.distinct_specs):
            sizes, _, _, _, slots = _get_deepseek_v4_cache_layout(groups)
            return sum(sizes) * slots
        return get_qwen4_exp_pool_bytes_per_block(groups)


def _replicated_draft_pools(groups):
    if not any(issubclass(g.manager_cls, MambaManager) for g in groups):
        return None
    target, drafts = [], []
    for g in groups:
        if (
            g.manager_cls is FullAttentionManager
            and not g.dcp_sharded
            and all(isinstance(s, AttentionSpec) and not isinstance(s, MLAAttentionSpec) for s in g.distinct_specs)
        ):
            drafts.append([g])
        else:
            target.append(g)
    return [target, *drafts] if target and drafts else None


def _group_deepseek_v4_layers(layer_configs: dict[str, KVCacheLayerConfig]) -> list[KVCacheGroup]:
    full = defaultdict(dict)
    local = defaultdict(dict)
    for name, config in layer_configs.items():
        if config.manager_cls is SlidingWindowManager:
            local[config.spec.block_size, config.manager_config, config.spec.dcp_sharded][name] = config
        else:
            full[config.spec.block_size, config.manager_config, config.spec.dcp_sharded][name] = config
    full_groups = [layers for _, layers in sorted(full.items(), key=lambda item: item[0][0])]
    grouped_layers = [*full_groups, *local.values()]
    if any(
        getattr(config.spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE for config in layer_configs.values()
    ):
        return tq_cache.group_layers(grouped_layers)

    if not full_groups:
        return [KVCacheGroup.from_layers(group) for group in grouped_layers]
    page_sizes = {config.spec.page_size_bytes for config in full_groups[0].values()}
    counts = [
        max(Counter(config.spec.page_size_bytes for config in group.values()).values()) for group in grouped_layers
    ]
    num_tuples = _approximate_gcd(counts, lower_bound=counts[0])
    groups = [KVCacheGroup.from_layers(group) for group in full_groups]
    for layers in local.values():
        buckets = defaultdict(list)
        for name, config in layers.items():
            current_size = config.spec.page_size_bytes
            page_size = min(size for size in page_sizes if size >= current_size)
            if page_size != current_size:
                config = replace(config, spec=replace(config.spec, page_size_padded=page_size))
            buckets[page_size].append((name, config))
        assert len({len(bucket) for bucket in buckets.values()}) == 1
        tuples = list(zip(*buckets.values(), strict=True))
        num_groups = cdiv(len(tuples), num_tuples)
        for i in range(num_groups):
            groups.append(KVCacheGroup.from_layers(dict(pair for slot in tuples[i::num_groups] for pair in slot)))
    return groups
