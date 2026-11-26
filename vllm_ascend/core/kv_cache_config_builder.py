# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend KV cache planning rules for vLLM's pluggable config builder.

vLLM plans the KV cache in ``vllm.v1.core.kv_cache_planning`` and resolves the
planner through ``Platform.get_kv_cache_config_builder_cls`` (see
``NPUPlatform.get_kv_cache_config_builder_cls``). Ascend-specific grouping,
pool accounting, and materialization live here, so the planning module itself
no longer needs a monkey patch for these decisions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    _approximate_gcd,
    may_override_num_blocks,
    unify_hybrid_kv_cache_specs,
    unify_kv_cache_spec_page_size,
)
from vllm.v1.core.kv_cache_planning import _pool_bytes_per_block as _upstream_pool_bytes_per_block
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups as _upstream_get_kv_cache_config_from_groups
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
)

from vllm_ascend.core.kv_cache_interface import KVCacheGroup
from vllm_ascend.utils import vllm_version_is

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheLayerConfig

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
    is_deepseek_v41_cache,
    make_cache_groups,
)
from vllm_ascend.models.deepseek_v41.cache_config import (
    group_layer_configs as group_deepseek_v41_layer_configs,
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


class AscendKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    """Build Ascend cache layouts with vLLM's cross-worker planning flow."""

    def get_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        layer_configs: dict[str, Any],
    ) -> list[KVCacheGroup]:
        """Group layers, preferring the Ascend layout when one applies.

        Precedence: Kimi-K3.1, Step5, GLM-Next, Qwen4-Exp MTP, DSV4.1,
        DSV4 mixed-page-size shared slots, Kimi-K3 DSpark mixed grouping,
        then vLLM's generic grouping.
        """
        if vllm_version_is("0.34.0"):
            return self._build_kv_cache_groups(vllm_config, layer_configs)
        draft_names = frozenset(name for name, lc in layer_configs.items() if lc.is_draft)
        vllm_config._ascend_draft_layer_names = draft_names
        groups = self._build_kv_cache_groups(vllm_config, layer_configs)
        speculative_config = getattr(vllm_config, "speculative_config", None)
        if speculative_config is not None and speculative_config.method == "dspark" and draft_names:
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_layout import (
                draft_only_group_ids,
            )

            vllm_config._ascend_dspark_draft_only_group_ids = draft_only_group_ids(groups, draft_names)
        return groups

    def _build_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        layer_configs: dict[str, Any],
    ) -> list[KVCacheGroup]:
        if vllm_config.model_config.hf_config.model_type == "kimi_k31":
            return paired_sfa_cache_groups(vllm_config, layer_configs)
        step5_groups = get_step5_kv_cache_groups(vllm_config, layer_configs, super().get_kv_cache_groups)
        if step5_groups is not None:
            return step5_groups
        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            # vLLM's generic grouping promotes local-attention layers before it
            # groups, so the Ascend layouts must see the same promoted configs.
            unify_hybrid_kv_cache_specs(layer_configs)

        specs_by_layer = (
            layer_configs if vllm_version_is("0.34.0") else {name: lc.spec for name, lc in layer_configs.items()}
        )
        if any(is_glm5_next_cache_spec(spec) for spec in specs_by_layer.values()):
            return get_glm5_next_kv_cache_groups(vllm_config, layer_configs)

        qwen4_exp_groups = get_qwen4_exp_kv_cache_groups(vllm_config, layer_configs, super().get_kv_cache_groups)
        if qwen4_exp_groups is not None:
            return qwen4_exp_groups

        if is_deepseek_v41_cache(specs_by_layer):
            if vllm_version_is("0.34.0"):
                return make_cache_groups(group_cache_specs(layer_configs))
            return group_deepseek_v41_layer_configs(layer_configs)

        if vllm_version_is("0.34.0"):
            grouped_specs = group_and_unify_kv_cache_specs(layer_configs)
            if grouped_specs is not None and len(grouped_specs) >= 2:
                # A uniform spec is not a mixed-page-size layout; grouping it here
                # would drop the C128 bucket the shared-tuple layout depends on.
                return _get_kv_cache_groups_uniform_groups(grouped_specs)
        else:
            dsv4_groups = _group_deepseek_v4_layers(layer_configs)
            if dsv4_groups is not None:
                return dsv4_groups

        kimi_k3_groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(layer_configs)
        if kimi_k3_groups is not None:
            return kimi_k3_groups
        if not vllm_version_is("0.34.0") and any(
            issubclass(lc.manager_cls, MambaManager) for lc in layer_configs.values()
        ):
            # Retry after unifying page sizes in case draft/target attention
            # layers only share page sizes once padded.
            try:
                unified = unify_kv_cache_spec_page_size(dict(layer_configs))
            except NotImplementedError:
                unified = None
            if unified is not None:
                kimi_k3_groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(unified)
                if kimi_k3_groups is not None:
                    layer_configs.update(unified)
                    return kimi_k3_groups

        groups = super().get_kv_cache_groups(vllm_config, layer_configs)
        if vllm_config.attention_config.hisparse_config is None:
            return groups
        transfer_config = vllm_config.kv_transfer_config
        if transfer_config is None or not transfer_config.has_connector("AscendStoreConnector"):
            return groups

        if vllm_version_is("0.34.0"):
            from vllm.v1.kv_cache_interface import HiSparseResidentSpec

            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hisparse.admission import (
                AscendStoreHiSparseResidentSpec,
            )

            for i, group in enumerate(groups):
                spec = group.kv_cache_spec  # type: ignore[attr-defined]
                if isinstance(spec, HiSparseResidentSpec):
                    resident_spec = AscendStoreHiSparseResidentSpec(
                        **{field.name: getattr(spec, field.name) for field in fields(spec) if field.init}
                    )
                    groups[i] = replace(group, kv_cache_spec=resident_spec)  # type: ignore[call-arg]
            return groups

        from vllm.v1.core.single_type_kv_cache_manager import HiSparseResidentManager

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hisparse.admission import (
            AscendStoreHiSparseResidentManager,
        )

        for i, group in enumerate(groups):
            if group.manager_cls is HiSparseResidentManager:
                groups[i] = replace(
                    group,
                    manager_config=replace(group.manager_config, manager_cls=AscendStoreHiSparseResidentManager),
                )
        return groups

    def get_kv_cache_config_from_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroup],
        num_blocks: int,
    ) -> KVCacheConfig:
        """Materialize the Ascend layouts for exactly ``num_blocks`` blocks.

        The default implementation derives the pool size from a one-block
        placement, which the shared-slot layouts here violate. Use the native
        pool cost so the model planners materialize exactly the requested block
        count before layerwise reuse is planned.
        """
        bytes_per_block = self._get_native_pool_bytes_per_block(vllm_config, kv_cache_groups)
        if bytes_per_block is None:
            return _attach_transfer_and_speculation_metadata(
                vllm_config,
                super().get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks),
            )
        return get_ascend_kv_cache_config_from_groups(
            vllm_config,
            kv_cache_groups,
            num_blocks * bytes_per_block,
        )

    def _get_pool_bytes_per_block(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroup],
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
        kv_cache_groups: list[KVCacheGroup],
    ) -> int | None:
        """Return the pool cost used to build the native descriptors."""
        if vllm_config.model_config.hf_config.model_type == "kimi_k31":
            draft_layer_names = getattr(vllm_config, "_ascend_draft_layer_names", None)
            if draft_layer_names is None:
                return paired_sfa_pool_bytes(kv_cache_groups)
            return paired_sfa_pool_bytes(kv_cache_groups, draft_layer_names=draft_layer_names)
        if is_step5_groups(kv_cache_groups):
            return get_step5_pool_bytes_per_block(kv_cache_groups)
        qwen_bytes = get_qwen4_exp_pool_bytes_per_block(kv_cache_groups)
        if qwen_bytes is not None:
            return qwen_bytes
        if is_deepseek_v41_cache(kv_cache_groups):
            return get_deepseek_v41_pool_bytes_per_block(kv_cache_groups)
        if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
            return get_glm5_next_pool_bytes_per_block(kv_cache_groups)
        pools = _get_replicated_draft_pools(kv_cache_groups)
        if pools is not None:
            return sum(_upstream_pool_bytes_per_block(vllm_config, pool) for pool in pools)
        if tq_cache.uses_turboquant_groups(kv_cache_groups):
            return tq_cache.pool_bytes_per_block(kv_cache_groups)
        if _is_deepseek_v4_groups(kv_cache_groups):
            return _deepseek_v4_pool_bytes_per_block(kv_cache_groups)
        return super()._get_pool_bytes_per_block(vllm_config, kv_cache_groups)


def _get_replicated_draft_pools(
    groups: list[KVCacheGroup],
) -> list[list[KVCacheGroup]] | None:
    """Keep contiguous dense draft K/V outside the target/state alias pool."""
    if not any(issubclass(group.manager_cls, MambaManager) for group in groups):
        return None
    target_groups = []
    draft_pools = []
    for group in groups:
        if (
            issubclass(group.manager_cls, FullAttentionManager)
            and not group.dcp_sharded
            and all(
                isinstance(spec, AttentionSpec) and not isinstance(spec, MLAAttentionSpec)
                for spec in group.distinct_specs
            )
        ):
            draft_pools.append([group])
        else:
            target_groups.append(group)
    if target_groups and draft_pools:
        return [target_groups, *draft_pools]
    return None


def _group_page_sizes(group: KVCacheGroup) -> set[int]:
    return {spec.page_size_bytes for spec in group.kv_cache_specs.values()}


def _group_max_layers_per_page_size(group: KVCacheGroup) -> int:
    counts: dict[int, int] = defaultdict(int)
    for spec in group.kv_cache_specs.values():
        counts[spec.page_size_bytes] += 1
    return max(counts.values())


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[Any] | None:
    """Group and unify DeepSeek-V4 specs into UniformTypeKVCacheSpecs (v0.34.0)."""
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec, UniformTypeKVCacheSpecs

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

    swa_uniform_specs: list[Any] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[Any],
) -> list[Any]:
    """Generate DeepSeek-V4 KV cache groups from UniformTypeKVCacheSpecs (v0.34.0)."""
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, SlidingWindowMLASpec, UniformTypeKVCacheSpecs

    assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
    if any(
        getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE
        for group in grouped_specs
        for spec in group.kv_cache_specs.values()
    ):
        return tq_cache.group_specs(grouped_specs)
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

    num_layer_tuples_per_group: list[int] = [g_spec.get_max_layers_per_page_size() for g_spec in grouped_specs]
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
    num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

    swa_mla_specs = grouped_specs[2:]
    assert all(
        isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
    )

    all_page_sizes = {s.page_size_bytes for s in full_mla_spec.kv_cache_specs.values()}
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = {s.page_size_bytes for s in sm_spec.kv_cache_specs.values()}
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        size_to_candidate = {ps: min(x for x in all_page_sizes if x >= ps) for ps in sm_page_sizes}
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        assert len({len(layers) for layers in layers_per_size.values()}) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
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


def _group_deepseek_v4_layers(
    layer_configs: dict[str, KVCacheLayerConfig],
) -> list[KVCacheGroup] | None:
    """Group DeepSeek-V4 full-MLA and sliding-window MLA layers."""
    if not any(
        issubclass(lc.manager_cls, SlidingWindowManager) and isinstance(lc.spec, MLAAttentionSpec)
        for lc in layer_configs.values()
    ):
        return None

    full_by_block: dict[int, dict[str, KVCacheLayerConfig]] = defaultdict(dict)
    swa_by_block: dict[int, dict[str, KVCacheLayerConfig]] = defaultdict(dict)
    for name, lc in layer_configs.items():
        if issubclass(lc.manager_cls, SlidingWindowManager) and isinstance(lc.spec, MLAAttentionSpec):
            swa_by_block[lc.spec.block_size][name] = lc
        elif isinstance(lc.spec, MLAAttentionSpec):
            full_by_block[lc.spec.block_size][name] = lc

    full_groups = [KVCacheGroup.from_layers(full_by_block[bs]) for bs in sorted(full_by_block)]
    swa_groups = [KVCacheGroup.from_layers(bucket) for bucket in swa_by_block.values()]
    grouped = [*full_groups, *swa_groups]
    if len(grouped) < 2:
        return None

    if any(
        getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE
        for group in grouped
        for spec in group.distinct_specs
    ):
        return tq_cache.group_layers(grouped)

    # For now, we restrict the first grouped_spec to be full MLA.
    full_mla_group = full_groups[0]
    full_mla_c128_group = full_groups[1]

    # We define a layer tuple as a group of layers with different page sizes, and
    # one KV cache group contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other KV cache groups will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group = [_group_max_layers_per_page_size(g) for g in grouped]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])

    # Split each SWA group into smaller groups to align their #(layer tuples),
    # possibly padding layer tuples for this.
    # Additionally, we also pad KV blocks in each SWA layer to align the page size
    # with the corresponding layer in the full-MLA group.
    all_page_sizes = _group_page_sizes(full_mla_group)
    split_swa_groups: list[KVCacheGroup] = []
    for sm_group in swa_groups:
        sm_page_sizes = _group_page_sizes(sm_group)
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate = {ps: min(x for x in all_page_sizes if x >= ps) for ps in sm_page_sizes}
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_group.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): for now, inside a group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len({len(layers) for layers in layers_per_size.values()}) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each group for aligned #(layers).
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            sub_configs = {name: layer_configs[name] for name in group_layer_names}
            split_swa_groups.append(KVCacheGroup.from_layers(sub_configs))

    return [full_mla_group, full_mla_c128_group, *split_swa_groups]


def _get_deepseek_v4_cache_layout(
    kv_cache_groups: list[KVCacheGroup],
) -> tuple[list[int], list[dict[int, list[str]]], list[str], int, int]:
    """Return the geometry shared by DSV4 planning and memory accounting.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream so every layer's page_size matches one of the
    full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each layer
    at tuple_idx = position-within-bucket.
    """
    page_sizes = sorted(_group_page_sizes(kv_cache_groups[0]))

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names: list[str] = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        specs = group.kv_cache_specs
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
    # per-page-size bucket; for SWA sub-groups this equals the sub-group
    # size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)
    return page_sizes, bucketed, mtp_layer_names, mtp_page_size, num_layer_tuples


def _is_deepseek_v4_groups(kv_cache_groups: list[KVCacheGroup]) -> bool:
    if is_deepseek_v41_cache(kv_cache_groups) or not kv_cache_groups:
        return False
    if vllm_version_is("0.34.0"):
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

        if not all(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in kv_cache_groups):  # type: ignore[attr-defined]
            return False
    return any(
        getattr(spec, "model_version", None) == "deepseek_v4"
        for group in kv_cache_groups
        for spec in group.distinct_specs
    )


def _deepseek_v4_pool_bytes_per_block(kv_cache_groups: list[KVCacheGroup]) -> int:
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
    kv_cache_groups: list[KVCacheGroup],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        return tq_cache.cache_config(vllm_config, kv_cache_groups, available_memory)
    (
        page_sizes,
        bucketed,
        mtp_layer_names,
        mtp_page_size,
        num_tuple_slots,
    ) = _get_deepseek_v4_cache_layout(kv_cache_groups)

    bytes_per_tuple = sum(page_sizes)
    num_blocks = available_memory // (bytes_per_tuple * num_tuple_slots)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    tuple_stride = bytes_per_tuple * num_blocks
    backing_size = tuple_stride * num_tuple_slots

    page_offsets: dict[int, int] = {}
    page_prefix = 0
    for page_size in page_sizes:
        page_offsets[page_size] = page_prefix * num_blocks
        page_prefix += page_size

    tensors: list[KVCacheTensor] = []
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

    return num_blocks, tensors


def get_ascend_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroup],
    available_memory: int,
) -> KVCacheConfig:
    """Materialize an Ascend layout from the memory budget Core resolved.

    The transfer and speculation metadata is attached by
    ``_attach_transfer_and_speculation_metadata`` before returning; see it for
    why the PD role must ride on the config object.
    """
    qwen_config = get_qwen4_exp_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    if vllm_config.model_config.hf_config.model_type == "kimi_k31":
        kv_cache_config = paired_sfa_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif is_step5_groups(kv_cache_groups):
        kv_cache_config = get_step5_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)
    elif qwen_config is not None:
        kv_cache_config = qwen_config
    elif is_deepseek_v41_cache(kv_cache_groups):
        kv_cache_config = get_deepseek_v41_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        kv_cache_config = get_glm5_next_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif (pools := _get_replicated_draft_pools(kv_cache_groups)) is not None:
        # Every physical pool shares the scheduler's block count, but owns a
        # disjoint byte range. Use the same aggregate divisor as admission and
        # rank re-planning, then retain upstream's per-pool stride descriptors.
        pool_bytes = [_upstream_pool_bytes_per_block(vllm_config, pool) for pool in pools]
        bytes_per_block = sum(pool_bytes)
        num_blocks = may_override_num_blocks(vllm_config, available_memory // bytes_per_block)
        total_size = num_blocks * bytes_per_block
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
    elif not _is_deepseek_v4_groups(kv_cache_groups):
        kv_cache_config = _upstream_get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)
    else:
        num_blocks, kv_cache_tensors = _get_kv_cache_config_deepseek_v4_main(
            vllm_config,
            kv_cache_groups,
            available_memory,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
        )
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
        from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import resident_mla_context_group_ids

        # Keep the PD context and external prefix-store classifications
        # separate: a mixed group needs draft block IDs AND ordinary target
        # hit keys. Remove these snapshots once upstream preserves per-layer
        # draft ownership on KVCacheGroup.
        kv_cache_config.dspark_context_group_ids = resident_mla_context_group_ids(kv_cache_config.kv_cache_groups)
        if vllm_version_is("0.34.0"):
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_layout import (
                draft_only_group_ids,
            )

            kv_cache_config.dspark_draft_only_group_ids = draft_only_group_ids(kv_cache_config.kv_cache_groups)
        else:
            kv_cache_config.dspark_draft_only_group_ids = getattr(vllm_config, "_ascend_dspark_draft_only_group_ids", ())
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
