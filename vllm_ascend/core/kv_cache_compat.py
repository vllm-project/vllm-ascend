# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Temporary Ascend-only bridge for vLLM #61452 legacy cache consumers.

Remove this bridge after Ascend's native manager-config migration lands.
"""

from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

from vllm.v1 import kv_cache_interface as kvi

from vllm_ascend.utils import vllm_version_is

if not TYPE_CHECKING and vllm_version_is("0.34.0"):
    from vllm.v1.kv_cache_interface import ChunkedLocalAttentionSpec as ChunkedLocalAttentionSpec
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec as CircularBufferSpec,
    )
    from vllm.v1.kv_cache_interface import (
        CrossAttentionSpec as CrossAttentionSpec,
    )
    from vllm.v1.kv_cache_interface import (
        EncoderOnlyAttentionSpec as EncoderOnlyAttentionSpec,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec as FullAttentionSpec,
    )
    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec as KVCacheGroupSpec,
    )
    from vllm.v1.kv_cache_interface import (
        RSWASpec as RSWASpec,
    )
    from vllm.v1.kv_cache_interface import (
        SlidingWindowMLASpec as SlidingWindowMLASpec,
    )
    from vllm.v1.kv_cache_interface import (
        SlidingWindowSpec as SlidingWindowSpec,
    )
    from vllm.v1.kv_cache_interface import (
        UniformTypeKVCacheSpecs as UniformTypeKVCacheSpecs,
    )
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry as KVCacheSpecRegistry

    LegacyMambaSpec = kvi.MambaSpec
    LegacyMLAAttentionSpec = kvi.MLAAttentionSpec
else:

    @dataclass(frozen=True, kw_only=True)
    class LegacyMambaSpec(kvi.MambaSpec):
        mamba_cache_mode: str = "none"
        num_speculative_blocks: int = 0
        num_prefill_checkpoint_blocks: int = 0
        prefill_checkpoint_alignment: int | None = None
        is_draft: bool = field(default=False, compare=False)

        def max_memory_usage_bytes(self, vllm_config) -> int:
            return legacy_allocation_blocks(self, vllm_config) * self.page_size_bytes

        def max_num_blocks_per_req(self, vllm_config, max_len: int) -> int:
            return legacy_block_table_width(self, vllm_config, max_len)

    @dataclass(frozen=True, kw_only=True)
    class LegacyMLAAttentionSpec(kvi.MLAAttentionSpec):
        is_draft: bool = field(default=False, compare=False)
        non_causal_multi_token_decode: bool = False

    @dataclass(frozen=True, kw_only=True)
    class FullAttentionSpec(kvi.AttentionSpec):
        sliding_window: int | None = None
        attention_chunk_size: int | None = None
        is_draft: bool = field(default=False, compare=False)
        non_causal: bool = False
        non_causal_multi_token_decode: bool = False

        def max_num_blocks_per_req(self, vllm_config, max_len: int) -> int:
            return legacy_block_table_width(self, vllm_config, max_len)

    @dataclass(frozen=True, kw_only=True)
    class SlidingWindowSpec(kvi.AttentionSpec):
        sliding_window: int
        extra_retained_tokens: int = 0
        bounded_replay: bool = False

        @property
        def num_retained_tokens(self) -> int:
            return self.sliding_window - 1 + self.extra_retained_tokens

        def max_memory_usage_bytes(self, vllm_config) -> int:
            return legacy_allocation_blocks(self, vllm_config) * self.page_size_bytes

        def max_num_blocks_per_req(self, vllm_config, max_len: int) -> int:
            return legacy_block_table_width(self, vllm_config, max_len)

    @dataclass(frozen=True, kw_only=True)
    class SlidingWindowMLASpec(SlidingWindowSpec):
        head_size_v: int = 0
        cache_dtype_str: str | None = None
        alignment: int | None = None
        model_version: str | None = None

    @dataclass(frozen=True, kw_only=True)
    class CircularBufferSpec(kvi.AttentionSpec):
        dcp_sharded: bool = False

        @property
        def uses_slot_mapping(self) -> bool:
            return False

        @property
        def prefix_cacheable(self) -> bool:
            return False

        def max_memory_usage_bytes(self, _vllm_config) -> int:
            return self.page_size_bytes

        def max_num_blocks_per_req(self, _vllm_config, _max_len: int) -> int:
            return 1

        @classmethod
        def merge(cls, specs):
            first = specs[0]
            assert all(spec == first for spec in specs), "All circular layers must use the same cache layout."
            return first

    @dataclass(frozen=True, kw_only=True)
    class CrossAttentionSpec(kvi.AttentionSpec):
        pass

    @dataclass(frozen=True, kw_only=True)
    class ChunkedLocalAttentionSpec(kvi.AttentionSpec):
        attention_chunk_size: int

    @dataclass(frozen=True, kw_only=True)
    class EncoderOnlyAttentionSpec(kvi.AttentionSpec):
        pass

    @dataclass(frozen=True, kw_only=True)
    class RSWASpec(FullAttentionSpec):
        rswa_window: int

    @dataclass(frozen=True)
    class UniformTypeKVCacheSpecs(kvi.KVCacheSpec):
        kv_cache_specs: dict[str, kvi.KVCacheSpec]

        @property
        def first_spec(self) -> kvi.KVCacheSpec:
            return next(iter(self.kv_cache_specs.values()))

        @property
        def page_size_bytes(self) -> int:
            return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

        @property
        def prefix_cacheable(self) -> bool:
            for spec in self.kv_cache_specs.values():
                policy = legacy_manager_config(spec)
                if not policy.manager_cls.prefix_cacheable(policy):
                    return False
            return True

        def max_memory_usage_pages(self, vllm_config) -> int:
            return max(legacy_allocation_blocks(spec, vllm_config) for spec in self.kv_cache_specs.values())

        def max_memory_usage_bytes(self, vllm_config) -> int:
            return self.max_memory_usage_pages(vllm_config) * self.page_size_bytes

        def max_num_blocks_per_req(self, vllm_config, max_len: int) -> int:
            widths = {legacy_block_table_width(spec, vllm_config, max_len) for spec in self.kv_cache_specs.values()}
            assert len(widths) == 1, "All layers in a KV cache group must need the same number of block table entries."
            return next(iter(widths))

        def get_max_layers_per_page_size(self) -> int:
            return Counter(spec.page_size_bytes for spec in self.kv_cache_specs.values()).most_common(1)[0][1]

        @classmethod
        def from_specs(cls, specs: dict[str, kvi.KVCacheSpec]):
            if not specs:
                return None
            first = next(iter(specs.values()))
            if any(
                (spec.block_size, spec.dcp_sharded) != (first.block_size, first.dcp_sharded) for spec in specs.values()
            ):
                return None
            return cls(first.block_size, specs, dcp_sharded=first.dcp_sharded)

    @dataclass
    class KVCacheGroupSpec:
        layer_names: list[str]
        kv_cache_spec: kvi.KVCacheSpec
        is_eagle_group: bool = False
        host_resident: bool = False
        enable_kv_transfer: bool = True
        role: kvi.KVCacheGroupRole = kvi.KVCacheGroupRole.DEFAULT

        @property
        def kv_cache_specs(self) -> dict[str, kvi.KVCacheSpec]:
            """Per-layer physical layouts exposed to cache config consumers."""
            if isinstance(self.kv_cache_spec, UniformTypeKVCacheSpecs):
                return {name: self.kv_cache_spec.kv_cache_specs[name] for name in self.layer_names}
            return dict.fromkeys(self.layer_names, self.kv_cache_spec)

        @property
        def block_size(self) -> int:
            return self.kv_cache_spec.block_size

        @property
        def dcp_sharded(self) -> bool:
            return self.kv_cache_spec.dcp_sharded

        @property
        def page_size_bytes(self) -> int:
            return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

        @property
        def distinct_specs(self) -> tuple[kvi.KVCacheSpec, ...]:
            spec = self.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                return tuple(dict.fromkeys(spec.kv_cache_specs.values()))
            return (spec,)

        @property
        def manager_config(self) -> kvi.ManagerConfig:
            return legacy_manager_config(self.distinct_specs[0])

        @property
        def manager_cls(self):
            return self.manager_config.manager_cls

        @property
        def prefix_cacheable(self) -> bool:
            return self.manager_cls.prefix_cacheable(self.manager_config)

        @property
        def prefix_replay_tokens(self) -> int:
            return self.manager_cls.prefix_replay_tokens(self.manager_config)

        @property
        def sliding_window_reach(self) -> int:
            return self.manager_cls.sliding_window_reach(self.manager_config)

        @property
        def uses_slot_mapping(self) -> bool:
            return self.manager_cls.uses_slot_mapping and any(spec.uses_slot_mapping for spec in self.distinct_specs)

        def max_allocated_blocks_per_request(self, vllm_config) -> int:
            return self.manager_cls.max_allocated_blocks_per_request(
                self.manager_config, self.block_size, self.dcp_sharded, vllm_config
            )

        def max_num_blocks_per_req(self, vllm_config, max_len=None) -> int:
            if max_len is None:
                max_len = vllm_config.model_config.max_model_len
            return self.manager_cls.max_num_blocks_per_req(
                self.manager_config, self.block_size, self.dcp_sharded, vllm_config, max_len
            )

    class KVCacheSpecRegistry:
        """Keep registrations local to Ascend's legacy spec consumers."""

        _managers: dict[type, type] = {}
        _uniform_types: dict[type, type] = {}

        @classmethod
        def register(cls, kvcache_spec_cls, manager_class, uniform_type_base_spec=None):
            cls._managers[kvcache_spec_cls] = manager_class
            cls._uniform_types[kvcache_spec_cls] = uniform_type_base_spec or kvcache_spec_cls

        @classmethod
        def get_manager_class(cls, spec):
            for base in type(spec).__mro__:
                if base in cls._managers:
                    return cls._managers[base]
            return None

        @classmethod
        def get_uniform_type_base_spec(cls, spec):
            for base in type(spec).__mro__:
                if base in cls._uniform_types:
                    return cls._uniform_types[base]
            return None


if not vllm_version_is("0.34.0"):

    def legacy_manager_config(spec: kvi.KVCacheSpec) -> kvi.ManagerConfig:
        """Resolve a legacy declaration through the main cache managers."""
        from vllm.v1.core.single_type_kv_cache_manager import (
            ChunkedLocalAttentionManager,
            CircularBufferManager,
            CrossAttentionManager,
            FullAttentionManager,
            MambaManager,
            RSWAManager,
            SlidingWindowManager,
        )

        from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec, AscendIndexerKPoolTailSpec

        if isinstance(spec, AscendIndexerKPoolTailSpec):
            from vllm_ascend.models.glm5next.kv_cache import KpoolTailManager

            return KpoolTailManager.config()
        if isinstance(spec, AscendCircularBufferSpec):
            from vllm_ascend.core.single_type_kv_cache_manager import AscendCircularBufferManager

            return AscendCircularBufferManager.config()
        if isinstance(spec, CircularBufferSpec):
            return CircularBufferManager.config()
        if isinstance(spec, SlidingWindowSpec):
            return SlidingWindowManager.config(
                sliding_window=spec.sliding_window,
                extra_retained_tokens=spec.extra_retained_tokens,
                bounded_replay=spec.bounded_replay,
            )
        if isinstance(spec, ChunkedLocalAttentionSpec):
            return ChunkedLocalAttentionManager.config(attention_chunk_size=spec.attention_chunk_size)
        if isinstance(spec, RSWASpec):
            return RSWAManager.config(rswa_window=spec.rswa_window)
        if isinstance(spec, CrossAttentionSpec):
            return CrossAttentionManager.config()
        if isinstance(spec, LegacyMambaSpec):
            return MambaManager.config(
                enable_prefix_caching=spec.mamba_cache_mode == "align",
                num_speculative_blocks=spec.num_speculative_blocks,
                num_prefill_checkpoint_blocks=spec.num_prefill_checkpoint_blocks,
                prefill_checkpoint_alignment=spec.prefill_checkpoint_alignment,
            )
        if isinstance(spec, kvi.MambaSpec):
            return MambaManager.config()
        if isinstance(spec, kvi.AttentionSpec) and not isinstance(spec, EncoderOnlyAttentionSpec):
            return FullAttentionManager.config()
        raise TypeError(f"Unsupported legacy KV cache spec: {type(spec).__name__}")

    def legacy_allocation_blocks(spec: kvi.KVCacheSpec, vllm_config) -> int:
        policy = legacy_manager_config(spec)
        return policy.manager_cls.max_allocated_blocks_per_request(
            policy, spec.block_size, spec.dcp_sharded, vllm_config
        )

    def legacy_block_table_width(spec: kvi.KVCacheSpec, vllm_config, max_len: int) -> int:
        policy = legacy_manager_config(spec)
        return policy.manager_cls.max_num_blocks_per_req(
            policy, spec.block_size, spec.dcp_sharded, vllm_config, max_len
        )

    def _legacy_group_spec(group: kvi.KVCacheGroup, spec: kvi.KVCacheSpec) -> kvi.KVCacheSpec:
        if not isinstance(spec, kvi.MambaSpec):
            return spec
        from vllm.v1.core.single_type_kv_cache_manager import MambaManager

        params = group.manager_config.params
        assert isinstance(params, MambaManager.Params)
        layout = {field.name: getattr(spec, field.name) for field in fields(kvi.MambaSpec) if field.init}
        return LegacyMambaSpec(
            **layout,
            mamba_cache_mode="align" if params.enable_prefix_caching else "none",
            num_speculative_blocks=params.num_speculative_blocks,
            num_prefill_checkpoint_blocks=params.num_prefill_checkpoint_blocks,
            prefill_checkpoint_alignment=params.prefill_checkpoint_alignment,
        )

    def bridge_attention_layout(config: kvi.KVCacheLayerConfig) -> kvi.KVCacheSpec:
        """Retain the layout tags read by Ascend's pre-#61452 cache consumers."""
        from vllm.v1.core.single_type_kv_cache_manager import (
            CircularBufferManager,
            FullAttentionManager,
            SlidingWindowManager,
        )

        spec = config.spec
        if type(spec) not in (kvi.AttentionSpec, SlidingWindowSpec):
            return spec
        params = config.manager_config.params
        layout = {field.name: getattr(spec, field.name) for field in fields(kvi.AttentionSpec) if field.init}
        if config.manager_cls is FullAttentionManager:
            return FullAttentionSpec(**layout, is_draft=config.is_draft)
        if isinstance(params, SlidingWindowManager.Params):
            return SlidingWindowSpec(
                **layout,
                sliding_window=params.sliding_window,
                extra_retained_tokens=params.extra_retained_tokens,
                bounded_replay=params.bounded_replay,
            )
        if config.manager_cls is CircularBufferManager:
            return CircularBufferSpec(**layout)
        return spec


def register_all_kvcache_specs(vllm_config) -> None:
    """Seed the Ascend registry used by remaining spec-based consumers."""
    if vllm_version_is("0.34.0"):
        from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs as register_release_specs

        register_release_specs(vllm_config)
        return

    from vllm.v1.core.single_type_kv_cache_manager import (
        ChunkedLocalAttentionManager,
        CircularBufferManager,
        CrossAttentionManager,
        FullAttentionManager,
        MambaManager,
        RSWAManager,
        SlidingWindowManager,
    )

    for spec, manager in (
        (FullAttentionSpec, FullAttentionManager),
        (kvi.MLAAttentionSpec, FullAttentionManager),
        (kvi.MambaSpec, MambaManager),
        (SlidingWindowSpec, SlidingWindowManager),
        (SlidingWindowMLASpec, SlidingWindowManager),
        (CircularBufferSpec, CircularBufferManager),
        (RSWASpec, RSWAManager),
        (ChunkedLocalAttentionSpec, ChunkedLocalAttentionManager),
        (CrossAttentionSpec, CrossAttentionManager),
    ):
        KVCacheSpecRegistry.register(spec, manager)


def create_legacy_cache_groups(
    specs: dict[str, kvi.KVCacheSpec],
    grouped_layer_names: list[list[str]],
) -> list[KVCacheGroupSpec]:
    """Keep the Ascend-only grouping helper after vLLM removes the old one."""
    groups = []
    for names in grouped_layer_names:
        layouts = [specs[name] for name in names]
        groups.append(KVCacheGroupSpec(names, _merge_legacy_layouts(layouts)))
    return groups


def _merge_legacy_layouts(layouts: list[kvi.KVCacheSpec]) -> kvi.KVCacheSpec:
    if vllm_version_is("0.34.0"):
        return layouts[0].merge(layouts)
    from vllm_ascend.core.kv_cache_interface import (
        AscendIndexerKPoolTailSpec,
        AscendMLAAttentionSpec,
        AscendSFAIndexerCacheSpec,
        AscendSlidingWindowMLASpec,
    )

    first = layouts[0]
    if isinstance(
        first,
        (
            AscendIndexerKPoolTailSpec,
            AscendMLAAttentionSpec,
            AscendSFAIndexerCacheSpec,
            AscendSlidingWindowMLASpec,
        ),
    ):
        return first.merge(layouts)
    assert all(layout == first for layout in layouts), "A uniform group must use the same layout"
    return first


def is_kv_cache_spec_uniform(specs: dict[str, kvi.KVCacheSpec]) -> bool:
    if not specs:
        return True
    try:
        _merge_legacy_layouts(list(specs.values()))
    except AssertionError:
        return False
    return True


def get_group_layer_layouts(group, *, excluded_layers: Collection[str] = ()) -> dict[str, kvi.KVCacheSpec]:
    """Read per-layer layouts from an upstream group or an Ascend legacy group."""
    if vllm_version_is("0.34.0") or not isinstance(group, kvi.KVCacheGroup):
        spec = group.kv_cache_spec
        layer_names = (name for name in group.layer_names if name not in excluded_layers)
        if isinstance(spec, UniformTypeKVCacheSpecs):
            return {name: spec.kv_cache_specs[name] for name in layer_names}
        return dict.fromkeys(layer_names, spec)
    return {
        name: _legacy_group_spec(group, spec)
        for name, spec in group.kv_cache_specs.items()
        if name not in excluded_layers
    }


def get_group_layout(group) -> kvi.KVCacheSpec:
    """Present a legacy layout to Ascend code that plans physical storage."""
    if vllm_version_is("0.34.0") or not isinstance(group, kvi.KVCacheGroup):
        return group.kv_cache_spec
    distinct = group.distinct_specs
    if len(distinct) == 1:
        return _legacy_group_spec(group, distinct[0])
    return UniformTypeKVCacheSpecs(
        block_size=group.block_size, dcp_sharded=group.dcp_sharded, kv_cache_specs=get_group_layer_layouts(group)
    )


def legacy_group_view(group) -> KVCacheGroupSpec:
    """Keep the native group intact while an Ascend layout planner reads it."""
    if vllm_version_is("0.34.0") or not isinstance(group, kvi.KVCacheGroup):
        return group
    return KVCacheGroupSpec(
        layer_names=group.layer_names,
        kv_cache_spec=get_group_layout(group),
        host_resident=group.host_resident,
        enable_kv_transfer=group.enable_kv_transfer,
        role=group.role,
    )
