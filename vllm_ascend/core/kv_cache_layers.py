# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer and group interfaces for the two supported vLLM cache planners."""

from dataclasses import dataclass, replace

from vllm.v1 import kv_cache_interface
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import MLAAttentionSpec

HAS_LAYER_CONFIG = hasattr(kv_cache_interface, "KVCacheLayerConfig")

if HAS_LAYER_CONFIG:
    CacheRequirement = kv_cache_interface.KVCacheLayerConfig
    from vllm.v1.core.kv_cache_planning import can_share_block_table
    from vllm.v1.kv_cache_interface import KVCacheGroup as CacheGroup

    @dataclass
    class AscendCacheGroup(CacheGroup):
        # A mixed target/draft allocation needs per-layer ownership after PP projection.
        draft_layers: frozenset[str] = frozenset()
        target_specs: tuple[kv_cache_interface.KVCacheSpec, ...] = ()

        @classmethod
        def from_layers(cls, layer_configs, **kwargs):
            return super().from_layers(layer_configs, **cls.ownership(layer_configs), **kwargs)

        @staticmethod
        def ownership(configs):
            return dict(
                draft_layers=frozenset(n for n, c in configs.items() if c.is_draft),
                target_specs=tuple(dict.fromkeys(c.spec for c in configs.values() if not c.is_draft)),
            )

    def make_cache_group(configs, specs, *, uniform=True):
        members = {name: replace(configs[name], spec=spec) for name, spec in specs.items()}
        if not can_share_block_table(members):
            return None
        return AscendCacheGroup.from_layers(members)

    def group_max_blocks(group, config):
        return group.max_allocated_blocks_per_request(config)

    class FullAttentionCacheLayer:
        def get_kv_cache_config(self, vllm_config):
            spec = self._get_cache_layout(vllm_config)
            return FullAttentionManager.layer_config(spec) if spec is not None else None

    class SlidingMLACacheLayer:
        def get_kv_cache_config(self, vllm_config):
            layout = self._get_cache_layout(vllm_config)
            return SlidingWindowManager.layer_config(layout, sliding_window=self._cache_window)

    def get_layer_spec(layer, vllm_config):
        config = layer.get_kv_cache_config(vllm_config)
        return config.spec if config is not None else None

    def group_layer_specs(group):
        return group.kv_cache_specs

    MLA_LAYOUT_TYPES = (MLAAttentionSpec,)

    def is_plain_full_attention(spec, config):
        return type(spec) is kv_cache_interface.AttentionSpec and config.manager_cls is FullAttentionManager

    def group_params(group):
        return group.manager_config.params

    def group_manager(group):
        return group.manager_cls

    def group_prefix_cacheable(group):
        return group.prefix_cacheable

    def group_sliding_reach(group):
        return group.sliding_window_reach

    def group_distinct_specs(group):
        return group.distinct_specs

    def group_block_size(group):
        return group.block_size

else:
    CacheRequirement = kv_cache_interface.KVCacheSpec
    from vllm.utils.math_utils import cdiv
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec as CacheGroup
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    def make_cache_group(configs, specs, *, uniform=True):
        if uniform:
            spec = UniformTypeKVCacheSpecs.from_specs(specs)
        else:
            first = next(iter(specs.values()))
            spec = first.merge(list(specs.values()))
        return CacheGroup(list(specs), spec) if spec is not None else None

    def group_max_blocks(group, config):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            return spec.max_memory_usage_pages(config)
        return cdiv(spec.max_memory_usage_bytes(config), spec.page_size_bytes)

    class FullAttentionCacheLayer:
        def get_kv_cache_spec(self, vllm_config):
            return self._get_cache_layout(vllm_config)

    class SlidingMLACacheLayer:
        def get_kv_cache_spec(self, vllm_config):
            return self._get_cache_layout(vllm_config)

    def get_layer_spec(layer, vllm_config):
        return layer.get_kv_cache_spec(vllm_config)

    def group_layer_specs(group):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            # A release group may list shared consumers alongside the local cache owners.
            return {name: spec.kv_cache_specs[name] for name in group.layer_names if name in spec.kv_cache_specs}
        return dict.fromkeys(group.layer_names, spec)

    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    MLA_LAYOUT_TYPES = (MLAAttentionSpec, SlidingWindowMLASpec)

    def is_plain_full_attention(spec, config):
        return type(spec) is kv_cache_interface.FullAttentionSpec

    def group_params(group):
        return group_distinct_specs(group)[0]

    def group_manager(group):
        return KVCacheSpecRegistry.get_manager_class(group_params(group))

    def group_prefix_cacheable(group):
        from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable

        return is_prefix_cacheable(group.kv_cache_spec)

    def group_sliding_reach(group):
        return group_params(group).num_retained_tokens

    def group_distinct_specs(group):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            return tuple(dict.fromkeys(spec.kv_cache_specs.values()))
        return (spec,)

    def group_block_size(group):
        return group.kv_cache_spec.block_size


def cache_layouts(requirements: dict[str, CacheRequirement]) -> dict[str, kv_cache_interface.KVCacheSpec]:
    if HAS_LAYER_CONFIG:
        return {name: config.spec for name, config in requirements.items()}
    return requirements


def group_draft_layers(group):
    if HAS_LAYER_CONFIG:
        return group.draft_layers if isinstance(group, AscendCacheGroup) else frozenset()
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return {name for name, layout in spec.kv_cache_specs.items() if layout.is_draft}
    return set(group.layer_names) if spec.is_draft else set()


def group_target_specs(group):
    if HAS_LAYER_CONFIG:
        return group.target_specs if isinstance(group, AscendCacheGroup) else group.distinct_specs
    spec = group.kv_cache_spec
    layouts = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
    return tuple(layout for layout in layouts if not layout.is_draft)


def configure_cache_transfer(vllm_config, kv_cache_config):
    kv_cache_config.kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None and speculative_config.method == "dspark":
        from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import resident_mla_context_group_ids
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_layout import draft_only_group_ids

        # Release scheduler conversion keeps only the first spec in a uniform
        # group, losing both residency and per-layer is_draft ownership. Keep
        # the PD context and external prefix-store classifications separate:
        # a mixed group needs draft block IDs AND ordinary target hit keys.
        # Remove these snapshots once upstream preserves per-layer ownership.
        kv_cache_config.dspark_context_group_ids = resident_mla_context_group_ids(kv_cache_config.kv_cache_groups)
        kv_cache_config.dspark_draft_only_group_ids = draft_only_group_ids(kv_cache_config.kv_cache_groups)
