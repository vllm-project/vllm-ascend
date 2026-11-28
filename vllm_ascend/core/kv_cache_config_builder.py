# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend KV cache planning rules for vLLM's pluggable config builder.

vLLM plans the KV cache in ``vllm.v1.core.kv_cache_planning`` and resolves the
planner through ``Platform.get_kv_cache_config_builder_cls`` (see
``NPUPlatform.get_kv_cache_config_builder_cls``). Ascend-specific grouping,
pool accounting, and materialization live here, so the planning module itself
no longer needs a monkey patch for these decisions.
"""

from collections import defaultdict
from dataclasses import dataclass, field, fields, replace

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    _approximate_gcd,
    unify_hybrid_kv_cache_specs,
)
from vllm.v1.core.kv_cache_planning import _pool_bytes_per_block as _upstream_pool_bytes_per_block
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups as _upstream_get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_ascend.core.kv_cache_compat import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    LegacyMambaSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_group_layer_layouts,
    legacy_group_view,
)
from vllm_ascend.core.kv_cache_interface import may_override_num_blocks
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
    get_layerwise_offload_config,
    get_spec_reuse_rules,
    plan_layerwise_cache_reuse,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LAYERWISE_REUSE_BACKING_BYTES,
    get_layerwise_reuse_backing_bytes,
)
from vllm_ascend.models.deepseek_v41.cache_config import (
    get_deepseek_v41_kv_cache_config,
    get_deepseek_v41_pool_bytes_per_block,
    group_cache_specs,
    group_layer_configs,
    is_deepseek_v41_cache,
    make_cache_groups,
    place_deepseek_v41_cache,
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
    is_step5_kv_cache_spec,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import TURBOQUANT_CACHE_DTYPE
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache
from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.34.0"):
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
    from vllm.v1.kv_cache_interface import KVCacheGroup

    @dataclass
    class _DSparkKVCacheGroup(KVCacheGroup):
        # Keep global ownership when Core projects the group onto a PP rank.
        draft_prefix_only: bool = field(default=False, compare=False)
        resident_context: bool = field(default=False, compare=False)


class AscendKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    """Build Ascend cache layouts with vLLM's cross-worker planning flow."""

    def get_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        """Group layers, preferring the Ascend layout when one applies.

        Precedence: Kimi-K3.1, GLM-Next, Qwen4-Exp MTP, then DSV4.1 /
        mixed-page-size shared slots, then the Kimi-K3 DSpark mixed grouping,
        then vLLM's generic grouping.
        """
        if not vllm_version_is("0.34.0"):
            groups = self._get_native_groups(vllm_config, kv_cache_spec)
            speculative = getattr(vllm_config, "speculative_config", None)
            if speculative is not None and speculative.method == "dspark":
                return _annotate_dspark_groups(groups, kv_cache_spec)
            return groups
        if vllm_config.model_config.hf_config.model_type == "kimi_k31":
            return paired_sfa_cache_groups(vllm_config, kv_cache_spec)
        step5_groups = get_step5_kv_cache_groups(vllm_config, kv_cache_spec, super().get_kv_cache_groups)
        if step5_groups is not None:
            return step5_groups
        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            # vLLM's generic grouping promotes local-attention specs before it
            # groups, so the Ascend layouts must see the same promoted specs.
            unify_hybrid_kv_cache_specs(kv_cache_spec)

        if any(is_glm5_next_cache_spec(spec) for spec in kv_cache_spec.values()):
            return get_glm5_next_kv_cache_groups(vllm_config, kv_cache_spec)

        qwen4_exp_groups = get_qwen4_exp_kv_cache_groups(vllm_config, kv_cache_spec, super().get_kv_cache_groups)
        if qwen4_exp_groups is not None:
            return qwen4_exp_groups

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

        groups = super().get_kv_cache_groups(vllm_config, kv_cache_spec)
        return _apply_hisparse_ascendstore_admission(vllm_config, groups)

    def _get_native_groups(self, vllm_config, configs):
        attention = getattr(vllm_config, "attention_config", None)
        if getattr(attention, "hisparse_config", None) is not None:
            groups = super().get_kv_cache_groups(vllm_config, configs)
            return _apply_hisparse_ascendstore_admission(vllm_config, groups)
        specs = {name: config.spec for name, config in configs.items()}
        if is_deepseek_v41_cache(specs):
            return group_layer_configs(configs)

        def default_groups(config, layouts):
            native = super(AscendKVCacheConfigBuilder, self).get_kv_cache_groups(
                config, {name: replace(configs[name], spec=spec) for name, spec in layouts.items()}
            )
            return [legacy_group_view(group) for group in native]

        if is_step5_kv_cache_spec(specs):
            groups = get_step5_kv_cache_groups(vllm_config, specs, default_groups)
        elif vllm_config.model_config.hf_config.model_type == "kimi_k31":
            groups = paired_sfa_cache_groups(vllm_config, specs, layer_configs=configs)
        elif any(is_glm5_next_cache_spec(spec) for spec in specs.values()):
            groups = get_glm5_next_kv_cache_groups(vllm_config, specs)
        else:
            mixed_specs = group_and_unify_kv_cache_specs(specs)
            if mixed_specs is not None and len(mixed_specs) >= 2:
                groups = _get_kv_cache_groups_uniform_groups(mixed_specs)
            else:
                groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(specs, layer_configs=configs)
                if groups is None:
                    groups = get_qwen4_exp_kv_cache_groups(vllm_config, specs, default_groups)
        if groups is None:
            return super().get_kv_cache_groups(vllm_config, configs)
        return [
            KVCacheGroup.from_layers(
                {name: replace(configs[name], spec=layout) for name, layout in get_group_layer_layouts(group).items()},
                host_resident=group.host_resident,
                enable_kv_transfer=group.enable_kv_transfer,
                role=group.role,
            )
            for group in groups
        ]

    def get_kv_cache_config_from_groups(self, vllm_config, kv_cache_groups, num_blocks) -> KVCacheConfig:
        if vllm_version_is("0.34.0"):
            bytes_per_block = self._get_native_pool_bytes_per_block(vllm_config, kv_cache_groups)
            if bytes_per_block is None:
                return _attach_transfer_and_speculation_metadata(
                    vllm_config,
                    super().get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks),
                )
            return get_ascend_kv_cache_config_from_groups(
                vllm_config,
                kv_cache_groups,
                num_blocks,
            )
        legacy_groups = [legacy_group_view(group) for group in kv_cache_groups]
        config = None
        if is_deepseek_v41_cache(kv_cache_groups):
            config = place_deepseek_v41_cache(vllm_config, kv_cache_groups, num_blocks)
        elif vllm_config.model_config.hf_config.model_type == "kimi_k31":
            config = paired_sfa_cache_config(vllm_config, legacy_groups, num_blocks)
        elif _get_glm5_next_cache_layout(legacy_groups) is not None:
            config = get_glm5_next_kv_cache_config(vllm_config, legacy_groups, num_blocks)
        elif is_step5_groups(legacy_groups):
            config = get_step5_kv_cache_config_from_groups(vllm_config, legacy_groups, num_blocks)
        elif (pools := _get_native_replicated_draft_pools(kv_cache_groups)) is not None:
            config = self._place_native_replicated_draft_pools(vllm_config, kv_cache_groups, pools, num_blocks)
        elif _has_deepseek_v4_layout(legacy_groups):
            config = _place_deepseek_v4_cache(vllm_config, kv_cache_groups, legacy_groups, num_blocks)
        elif (qwen_pool_bytes := get_qwen4_exp_pool_bytes_per_block(legacy_groups)) is not None:
            config = get_qwen4_exp_kv_cache_config(vllm_config, legacy_groups, num_blocks * qwen_pool_bytes)
        if config is None:
            config = super().get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks)
        config.kv_cache_groups = kv_cache_groups
        return _attach_transfer_and_speculation_metadata(vllm_config, config)

    def _place_native_replicated_draft_pools(self, vllm_config, groups, pools, num_blocks):
        build_pool = super().get_kv_cache_config_from_groups
        configs = [build_pool(vllm_config, pool, num_blocks) for pool in pools]
        assert all(config.num_blocks == num_blocks for config in configs)
        pool_sizes = [max(tensor.size for tensor in config.kv_cache_tensors) for config in configs]
        size = sum(pool_sizes)
        tensors: list[KVCacheTensor] = []
        offset = 0
        for config, pool_size in zip(configs, pool_sizes):
            tensors.extend(
                replace(tensor, size=size, offset=offset + tensor.offset) for tensor in config.kv_cache_tensors
            )
            offset += pool_size
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=tensors,
            kv_cache_groups=groups,
            prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
        )

    def _get_pool_bytes_per_block(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> int | None:
        """Return the device memory cost per global block ID."""
        extra_config = get_layerwise_offload_config(getattr(vllm_config, "kv_transfer_config", None))
        if extra_config is not None and kv_cache_groups and get_spec_reuse_rules(vllm_config) is not None:
            probe = self.get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks=1)
            backing_bytes = get_layerwise_reuse_backing_bytes(probe)
            if backing_bytes is not None:
                return backing_bytes
        return self._get_native_pool_bytes_per_block(vllm_config, kv_cache_groups)

    def _get_native_pool_bytes_per_block(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups,
    ) -> int | None:
        """Return the pool cost used to build the native descriptors."""
        native_groups = kv_cache_groups
        kv_cache_groups = [legacy_group_view(group) for group in kv_cache_groups]
        if vllm_config.model_config.hf_config.model_type == "kimi_k31":
            return paired_sfa_pool_bytes(kv_cache_groups)
        pools = _get_replicated_draft_pools(kv_cache_groups) if vllm_version_is("0.34.0") else None
        if pools is not None:
            return sum(_upstream_pool_bytes_per_block(vllm_config, pool) for pool in pools)
        if is_step5_groups(kv_cache_groups):
            return get_step5_pool_bytes_per_block(kv_cache_groups)
        if not vllm_version_is("0.34.0") and (pools := _get_native_replicated_draft_pools(native_groups)) is not None:
            return sum(_upstream_pool_bytes_per_block(vllm_config, pool) for pool in pools)
        if is_deepseek_v41_cache(kv_cache_groups):
            return get_deepseek_v41_pool_bytes_per_block(kv_cache_groups)
        if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
            return get_glm5_next_pool_bytes_per_block(kv_cache_groups)
        qwen_bytes = get_qwen4_exp_pool_bytes_per_block(kv_cache_groups)
        if qwen_bytes is not None:
            return qwen_bytes
        if tq_cache.uses_turboquant_groups(kv_cache_groups):
            return tq_cache.pool_bytes_per_block(kv_cache_groups)
        if _is_deepseek_v4_groups(kv_cache_groups) or (
            not vllm_version_is("0.34.0") and _has_deepseek_v4_layout(kv_cache_groups)
        ):
            return _deepseek_v4_pool_bytes_per_block(kv_cache_groups)
        return super()._get_pool_bytes_per_block(vllm_config, native_groups)


def _apply_hisparse_ascendstore_admission(vllm_config, groups):
    if vllm_config.attention_config.hisparse_config is None:
        return groups
    transfer_config = vllm_config.kv_transfer_config
    if transfer_config is None or not transfer_config.has_connector("AscendStoreConnector"):
        return groups

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hisparse.admission import (
        AscendStoreHiSparseResidentManager,
        AscendStoreHiSparseResidentSpec,
    )

    if vllm_version_is("0.34.0"):
        from vllm.v1.kv_cache_interface import HiSparseResidentSpec

        for i, group in enumerate(groups):
            spec = group.kv_cache_spec
            if isinstance(spec, HiSparseResidentSpec):
                resident_spec = AscendStoreHiSparseResidentSpec(
                    **{field.name: getattr(spec, field.name) for field in fields(spec) if field.init}
                )
                groups[i] = replace(group, kv_cache_spec=resident_spec)
    else:
        from vllm.v1.core.single_type_kv_cache_manager import HiSparseResidentManager

        for i, group in enumerate(groups):
            if group.manager_cls is HiSparseResidentManager:
                resident_specs = {
                    name: AscendStoreHiSparseResidentSpec(
                        **{field.name: getattr(spec, field.name) for field in fields(spec) if field.init}
                    )
                    for name, spec in group.kv_cache_specs.items()
                }
                groups[i] = replace(
                    group,
                    kv_cache_specs=resident_specs,
                    manager_config=AscendStoreHiSparseResidentManager.config(),
                )
    return groups


def _annotate_dspark_groups(groups, layer_configs):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import resident_mla_context_group_ids

    resident_groups = set(resident_mla_context_group_ids(groups))
    return [
        _DSparkKVCacheGroup(
            **{f.name: getattr(group, f.name) for f in fields(KVCacheGroup)},
            draft_prefix_only=all(layer_configs[name].is_draft for name in group.layer_names),
            resident_context=i in resident_groups,
        )
        for i, group in enumerate(groups)
    ]


def _get_native_replicated_draft_pools(groups):
    """Reserve a disjoint backing region for each replicated dense draft group."""
    if not any(issubclass(group.manager_cls, MambaManager) for group in groups):
        return None
    target_groups = []
    draft_pools = []
    for group in groups:
        if (
            not group.dcp_sharded
            and issubclass(group.manager_cls, FullAttentionManager)
            and all(not isinstance(spec, MLAAttentionSpec) for spec in group.distinct_specs)
            and group.layer_names
        ):
            draft_pools.append([group])
        else:
            target_groups.append(group)
    if draft_pools and any(group.layer_names for group in target_groups):
        return [target_groups, *draft_pools]
    return None


def _get_replicated_draft_pools(
    groups: list[KVCacheGroupSpec],
) -> list[list[KVCacheGroupSpec]] | None:
    """Keep contiguous dense draft K/V outside the target/state alias pool."""
    has_mamba = False
    target_groups = []
    draft_pools = []
    for group in groups:
        spec = group.kv_cache_spec
        specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        has_mamba |= any(isinstance(layout, (MambaSpec, LegacyMambaSpec)) for layout in specs)
        if all(
            isinstance(s, FullAttentionSpec) and not isinstance(s, MLAAttentionSpec) and not s.dcp_sharded
            for s in specs
        ):
            draft_pools.append([group])
        else:
            target_groups.append(group)
    if has_mamba and target_groups and draft_pools:
        return [target_groups, *draft_pools]
    return None


def _page_sizes(spec: UniformTypeKVCacheSpecs) -> set[int]:
    """Distinct page sizes across a group's cache specs."""
    return {s.page_size_bytes for s in spec.kv_cache_specs.values()}


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """Group the KV cache specs and unify each group into one
    UniformTypeKVCacheSpecs. Currently, this is only used for DeepseekV4.
    """
    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    logical_block_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            logical_block_specs[spec.block_size][name] = spec

    mla_uniform_specs = []
    for block_size in sorted(logical_block_specs):
        spec_dict = logical_block_specs[block_size]
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """Generate the KV cache groups from the grouped specs."""
    assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
    if any(
        getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE
        for group in grouped_specs
        for spec in group.kv_cache_specs.values()
    ):
        return tq_cache.group_specs(grouped_specs)
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    full_mla_c128_spec = grouped_specs[1]

    assert all(isinstance(spec, MLAAttentionSpec) for spec in full_mla_spec.kv_cache_specs.values())
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )
    full_mla_c128_group = KVCacheGroupSpec(
        layer_names=list(full_mla_c128_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_c128_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [g_spec.get_max_layers_per_page_size() for g_spec in grouped_specs]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

    # TODO(cmq): this is not general enough
    swa_mla_specs = grouped_specs[2:]

    assert all(
        isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
    # Possibly padding layer tuples for this.
    # Additionally, we also pad KV blocks in each SWA layer, to align the page size
    # with the corresponding layer in the full-MLA group.
    all_page_sizes = _page_sizes(full_mla_spec)
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = _page_sizes(sm_spec)
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            group_layer_specs = {name: sm_spec.kv_cache_specs[name] for name in group_layer_names}
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, full_mla_c128_group, *swa_mla_groups]


def _get_deepseek_v4_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[list[int], list[dict[int, list[str]]], list[str], int, int]:
    """Return the geometry shared by DSV4 planning and memory accounting.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream so every layer's page_size matches one of the
    full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each layer
    at tuple_idx = position-within-bucket.
    """
    page_sizes = sorted({spec.page_size_bytes for spec in get_group_layer_layouts(kv_cache_groups[0]).values()})

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        specs = get_group_layer_layouts(group)
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            padded = getattr(specs[name], "page_size_padded", None)
            page_size = padded if padded is not None else specs[name].page_size_bytes
            if "mtp" not in name:
                b[page_size].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = page_size
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    return page_sizes, bucketed, mtp_layer_names, mtp_page_size, num_layer_tuples


def _is_deepseek_v4_groups(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    if (
        is_deepseek_v41_cache(kv_cache_groups)
        or not kv_cache_groups
        or not all(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in kv_cache_groups)
    ):
        return False
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        assert isinstance(group_spec, UniformTypeKVCacheSpecs)
        specs = group_spec.kv_cache_specs.values()
        if any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in specs):
            return True
    return False


def _deepseek_v4_pool_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Bytes one global block ID consumes across the shared tuple slots.

    vLLM re-plans the ranks with more KV memory against this value. The
    standardized per-group layout has a different divisor from Ascend's
    shared-tuple layout, so deriving it from the tuple geometry keeps every
    rank's ``num_blocks`` consistent.
    """
    page_sizes, _, _, _, num_layer_tuples = _get_deepseek_v4_cache_layout(kv_cache_groups)
    return sum(page_sizes) * num_layer_tuples


def _get_kv_cache_config_deepseek_v4_main(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    num_blocks: int,
) -> KVCacheConfig:
    tensors: list[KVCacheTensor]
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        _, tensors = tq_cache.cache_config(kv_cache_groups, num_blocks)
    else:
        (
            page_sizes,
            bucketed,
            mtp_layer_names,
            mtp_page_size,
            num_tuple_slots,
        ) = _get_deepseek_v4_cache_layout(kv_cache_groups)

        bytes_per_tuple = sum(page_sizes)

        tuple_stride = bytes_per_tuple * num_blocks
        backing_size = tuple_stride * num_tuple_slots

        # Within every tuple slot, page-size buckets are placed consecutively.
        page_offsets: dict[int, int] = {}
        page_prefix = 0
        for page_size in page_sizes:
            page_offsets[page_size] = page_prefix * num_blocks
            page_prefix += page_size

        tensors = []

        # Keep each descriptor inside one cache group. Descriptors from different
        # groups alias corresponding tuple slots by using identical geometry.
        for group_buckets in bucketed:
            for page_size in page_sizes:
                layer_names = group_buckets.get(page_size)
                if not layer_names:
                    continue

                tensors.append(
                    KVCacheTensor(
                        size=backing_size,
                        layers=list(layer_names),
                        offset=page_offsets[page_size],
                        layer_stride=tuple_stride,
                        block_stride=page_size,
                    )
                )

        # MTP layers receive trailing tuple slots. Unused page buckets remain
        # padding so num_blocks and memory accounting retain the existing contract.
        normal_tuple_slots = num_tuple_slots - len(mtp_layer_names)
        for index, layer_name in enumerate(mtp_layer_names):
            slot = normal_tuple_slots + index
            tensors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=[layer_name],
                    offset=slot * tuple_stride + page_offsets[mtp_page_size],
                    layer_stride=0,
                    block_stride=mtp_page_size,
                )
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )


def get_ascend_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    num_blocks: int,
) -> KVCacheConfig:
    """Describe Core's block count and preserve scheduler transfer metadata.

    Qwen4 retains its budget-based planner. Independent generic draft pools
    retain their existing override behavior.

    The transfer and speculation metadata is attached by
    ``_attach_transfer_and_speculation_metadata`` before returning; see it for
    why the PD role must ride on the config object.
    """
    pools = _get_replicated_draft_pools(kv_cache_groups)
    if not kv_cache_groups:
        kv_cache_config = DefaultKVCacheConfigBuilder().get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, num_blocks
        )
    elif vllm_config.model_config.hf_config.model_type == "kimi_k31":
        kv_cache_config = paired_sfa_cache_config(vllm_config, kv_cache_groups, num_blocks)
    elif is_step5_groups(kv_cache_groups):
        kv_cache_config = get_step5_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks)
    elif pools is not None:
        # Preserve #21345's independent-pool override and placement behavior;
        # this path is separate from the model-planner API migration.
        pool_bytes = [_upstream_pool_bytes_per_block(vllm_config, pool) for pool in pools]
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        total_size = num_blocks * sum(pool_bytes)
        kv_cache_tensors: list[KVCacheTensor] = []
        offset = 0
        for pool, pool_bytes_per_block in zip(pools, pool_bytes):
            pool_size = num_blocks * pool_bytes_per_block
            pool_config = _upstream_get_kv_cache_config_from_groups(vllm_config, pool, pool_size)
            assert pool_config.num_blocks == num_blocks
            kv_cache_tensors.extend(
                replace(tensor, size=total_size, offset=offset + tensor.offset)
                for tensor in pool_config.kv_cache_tensors
            )
            offset += pool_size
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
        )
    elif is_deepseek_v41_cache(kv_cache_groups):
        kv_cache_config = get_deepseek_v41_kv_cache_config(vllm_config, kv_cache_groups, num_blocks)
    elif _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        kv_cache_config = get_glm5_next_kv_cache_config(vllm_config, kv_cache_groups, num_blocks)
    elif (qwen_pool_bytes := get_qwen4_exp_pool_bytes_per_block(kv_cache_groups)) is not None:
        kv_cache_config = get_qwen4_exp_kv_cache_config(vllm_config, kv_cache_groups, num_blocks * qwen_pool_bytes)
        assert kv_cache_config is not None
    elif not _is_deepseek_v4_groups(kv_cache_groups):
        kv_cache_config = DefaultKVCacheConfigBuilder().get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, num_blocks
        )
    else:
        kv_cache_config = _get_kv_cache_config_deepseek_v4_main(vllm_config, kv_cache_groups, num_blocks)
    return _attach_transfer_and_speculation_metadata(vllm_config, kv_cache_config)


def _attach_transfer_and_speculation_metadata(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> KVCacheConfig:
    """Keep transfer and speculation metadata on every materialization path.

    Attach ``kv_transfer_config`` onto the built config: ``KVCacheConfig`` has
    no native field, and the PD prefill-producer role read back by the kv-cache
    coordinator (see ``patch_kv_cache_coordinator.py``) must ride on the config
    object - the coordinator factory receives ``KVCacheConfig`` but not
    ``VllmConfig``. The attribute survives the scheduler-side ``copy.deepcopy``
    in ``generate_scheduler_kv_cache_config`` and is dropped by worker pickle
    IPC, which never reads it. The hybrid cache coordinator treats a config
    without the role as standalone and would skip the EAGLE replay block drop
    used for external cache loads, so the upstream default materializer's
    result carries it too.
    """
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    kv_cache_config.kv_transfer_config = kv_transfer_config
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None and speculative_config.method == "dspark":
        groups = kv_cache_config.kv_cache_groups
        if vllm_version_is("0.34.0"):
            from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
                resident_mla_context_group_ids,
            )
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_layout import (
                draft_only_group_ids,
            )

            # The release scheduler conversion does not preserve per-layer
            # ownership. Snapshot the distinct PD and prefix-store roles.
            kv_cache_config.dspark_context_group_ids = resident_mla_context_group_ids(groups)
            kv_cache_config.dspark_draft_only_group_ids = draft_only_group_ids(groups)
        else:
            kv_cache_config.dspark_context_group_ids = tuple(
                i for i, group in enumerate(groups) if group.resident_context
            )
            kv_cache_config.dspark_draft_only_group_ids = tuple(
                i for i, group in enumerate(groups) if group.draft_prefix_only
            )
    reuse_layout = plan_layerwise_cache_reuse(kv_cache_config, vllm_config)
    if reuse_layout is not None:
        assert kv_transfer_config is not None
        # Core budgets these bytes and the scheduler uses the value to detect reuse.
        kv_cache_config.kv_transfer_config = replace(
            kv_transfer_config,
            kv_connector_extra_config={
                **kv_transfer_config.kv_connector_extra_config,
                LAYERWISE_REUSE_BACKING_BYTES: reuse_layout.kv_cache_tensors[0].size,
            },
        )
    return kv_cache_config


def _has_deepseek_v4_layout(groups: list[KVCacheGroupSpec]) -> bool:
    return any(
        isinstance(spec, MLAAttentionSpec) and spec.model_version == "deepseek_v4"
        for group in groups
        for spec in get_group_layer_layouts(group).values()
    )


def _place_deepseek_v4_cache(vllm_config, groups, layouts, num_blocks: int) -> KVCacheConfig:
    if tq_cache.uses_turboquant_groups(layouts):
        _, tensors = tq_cache.cache_config(layouts, num_blocks)
    else:
        sizes, buckets, draft_names, draft_size, slots = _get_deepseek_v4_cache_layout(layouts)
        stride = sum(sizes) * num_blocks
        offsets = {}
        offset = 0
        for size in sizes:
            offsets[size] = offset
            offset += size * num_blocks
        tensors = [
            KVCacheTensor(
                size=stride * slots, layers=list(names), offset=offsets[size], layer_stride=stride, block_stride=size
            )
            for group in buckets
            for size, names in group.items()
        ]
        tensors.extend(
            KVCacheTensor(
                size=stride * slots,
                layers=[name],
                offset=(slots - len(draft_names) + idx) * stride + offsets[draft_size],
                layer_stride=0,
                block_stride=draft_size,
            )
            for idx, name in enumerate(draft_names)
        )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )
