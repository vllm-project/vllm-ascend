# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend KV cache planning, owned by the pluggable config builder.

vLLM plans the KV cache in ``vllm.v1.core.kv_cache_planning`` and resolves the
planner through ``Platform.get_kv_cache_config_builder_cls`` (see
``NPUPlatform.get_kv_cache_config_builder_cls``). All Ascend-specific grouping,
materialization, and accounting rules therefore live here, in one owner, rather
than in a monkey patch of the planning module.
"""

from collections import defaultdict

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    _annotate_eagle_groups,
    _approximate_gcd,
    _unify_hybrid_kv_cache_specs,
    _uses_trailing_mtp_layers,
    _warn_if_unannotated_eagle_mamba,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.models.deepseek_v41.cache_config import (
    get_deepseek_v41_kv_cache_config,
    get_deepseek_v41_pool_bytes_per_block,
    group_cache_specs,
    is_deepseek_v41_cache,
    make_cache_groups,
)
from vllm_ascend.models.glm5next.cache_config import (
    _get_glm5_next_cache_layout,
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_max_memory_usage,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.models.glm5next.kv_cache import is_glm5_next_cache_spec
from vllm_ascend.quantization.methods.kv_cache.turboquant import TURBOQUANT_CACHE_DTYPE
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache

_KIMI_K3_TARGET_LAYER_PREFIX = "language_model.model.layers."
_KIMI_K3_DRAFT_LAYER_PREFIX = "model.layers."


class AscendKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    """Build Ascend cache layouts with vLLM's cross-worker planning flow."""

    def get_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        """Group layers, preferring the Ascend layout when one applies.

        Precedence: GLM-Next, then DSV4.1 / mixed-page-size shared slots, then
        the Kimi-K3 DSpark mixed grouping, then vLLM's generic grouping.
        """
        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            # vLLM's generic grouping promotes local-attention specs before it
            # groups, so the Ascend layouts must see the same promoted specs.
            _unify_hybrid_kv_cache_specs(kv_cache_spec)

        if any(is_glm5_next_cache_spec(spec) for spec in kv_cache_spec.values()):
            return get_glm5_next_kv_cache_groups(vllm_config, kv_cache_spec)

        grouped_groups = _get_ascend_grouped_groups(vllm_config, kv_cache_spec)
        if grouped_groups is not None:
            return grouped_groups

        kimi_k3_groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(kv_cache_spec)
        if kimi_k3_groups is not None:
            return kimi_k3_groups

        return super().get_kv_cache_groups(vllm_config, kv_cache_spec)

    def get_kv_cache_config_from_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        num_blocks: int,
    ) -> KVCacheConfig:
        if is_deepseek_v41_cache(kv_cache_groups):
            kv_cache_config = get_deepseek_v41_kv_cache_config(
                vllm_config,
                kv_cache_groups,
                num_blocks=num_blocks,
            )
        elif _get_glm5_next_cache_layout(kv_cache_groups) is not None:
            kv_cache_config = get_glm5_next_kv_cache_config(
                vllm_config,
                kv_cache_groups,
                num_blocks=num_blocks,
            )
        elif _is_turboquant_or_deepseek_v4_groups(kv_cache_groups):
            kv_cache_config = _get_kv_cache_config_deepseek_v4_main(
                vllm_config,
                kv_cache_groups,
                num_blocks,
            )
        else:
            kv_cache_config = super().get_kv_cache_config_from_groups(
                vllm_config,
                kv_cache_groups,
                num_blocks,
            )
        # KVCacheConfig has no native field for the transfer config, and the
        # coordinator factory receives the config but not VllmConfig. The
        # attribute survives the scheduler-side deepcopy and is dropped by
        # worker pickle IPC, which never reads it.
        kv_cache_config.kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        return kv_cache_config

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
        if is_deepseek_v41_cache(kv_cache_groups):
            return get_deepseek_v41_pool_bytes_per_block(kv_cache_groups)
        if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
            return get_glm5_next_pool_bytes_per_block(kv_cache_groups)
        if tq_cache.uses_turboquant_groups(kv_cache_groups):
            return tq_cache.pool_bytes_per_block(kv_cache_groups)
        if _is_deepseek_v4_groups(kv_cache_groups):
            return _deepseek_v4_pool_bytes_per_block(kv_cache_groups)
        return super()._get_pool_bytes_per_block(vllm_config, kv_cache_groups)

    def _get_max_memory_usage_bytes_from_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> int:
        """Only GLM-Next needs a custom admission formula.

        The default formula (``pool bytes per block x requested blocks``) is
        exact for every other Ascend layout once
        :meth:`_get_pool_bytes_per_block` is overridden, because those layouts
        are uniform-type groups charged once per requested page.
        """
        if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
            return get_glm5_next_max_memory_usage(vllm_config, kv_cache_groups)
        return super()._get_max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)


def _get_ascend_grouped_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Group DSV4.1 or mixed C4/C128/SWA-MLA layers into shared slots.

    Returns None when the spec is not an Ascend multi-page-size layout, so the
    caller can keep using vLLM's generic grouping.
    """
    if is_deepseek_v41_cache(kv_cache_spec):
        groups = make_cache_groups(group_cache_specs(kv_cache_spec))
    else:
        grouped_specs = group_and_unify_kv_cache_specs(kv_cache_spec)
        if grouped_specs is None or len(grouped_specs) < 2:
            # A uniform spec is not a mixed-page-size layout; grouping it here
            # would drop the C128 bucket the shared-tuple layout depends on.
            return None
        groups = _get_kv_cache_groups_uniform_groups(grouped_specs)

    _annotate_eagle_groups(
        vllm_config,
        kv_cache_spec,
        groups,
        use_trailing_layer_fallback=_uses_trailing_mtp_layers(vllm_config),
    )
    _warn_if_unannotated_eagle_mamba(vllm_config, groups)
    return groups


def _page_sizes(spec: UniformTypeKVCacheSpecs) -> set[int]:
    """Distinct page sizes across a group's cache specs."""
    return {s.page_size_bytes for s in spec.kv_cache_specs.values()}


def _is_turboquant_or_deepseek_v4_groups(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> bool:
    return tq_cache.uses_turboquant_groups(kv_cache_groups) or _is_deepseek_v4_groups(kv_cache_groups)


def _get_kimi_k3_dspark_mixed_kv_cache_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Build topology-independent Kimi K3 DSpark scheduler groups.

    Target and causal draft attention layers require the same full-sequence
    block ownership. Putting them in one UniformType group lets them share one
    scheduler block table while preserving a separate physical page per layer.
    Recurrent layers are split into the fewest balanced groups whose size does
    not exceed the attention group. This minimizes scheduler groups while
    keeping the recurrent groups balanced.

    Block and page sizes are resolved by the runtime and intentionally not
    fixed here: TP8 and TP16 produce different sizes but the same ownership
    relation. An unrecognized or incompatible signature falls back to vLLM's
    generic hybrid grouping.
    """
    target_attention_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
    }
    draft_attention_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_DRAFT_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
    }
    mamba_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and isinstance(spec, MambaSpec)
    }

    matched_layer_count = len(target_attention_specs) + len(draft_attention_specs) + len(mamba_specs)
    if (
        not target_attention_specs
        or not draft_attention_specs
        or not mamba_specs
        or matched_layer_count != len(kv_cache_spec)
    ):
        return None

    attention_specs = [*target_attention_specs.values(), *draft_attention_specs.values()]
    all_specs = [*attention_specs, *mamba_specs.values()]
    # Only attention layers share a scheduler block table. Mamba keeps its
    # own groups and may use max_model_len as block_size when cache_mode=none.
    if (
        len({spec.block_size for spec in attention_specs}) != 1
        or len({spec.page_size_bytes for spec in all_specs}) != 1
    ):
        return None

    first_mamba_spec = next(iter(mamba_specs.values()))
    if any(spec != first_mamba_spec for spec in mamba_specs.values()):
        return None

    # Insert target attention first. generate_scheduler_kv_cache_config unwraps a
    # UniformType group to its first spec, and this representative is registered
    # with the FullAttentionManager needed by both target and draft attention.
    mixed_attention_specs = {**target_attention_specs, **draft_attention_specs}
    mixed_attention_spec = UniformTypeKVCacheSpecs.from_specs(mixed_attention_specs)
    if mixed_attention_spec is None:
        return None

    groups = [
        KVCacheGroupSpec(
            layer_names=list(mixed_attention_specs),
            kv_cache_spec=mixed_attention_spec,
        )
    ]
    mamba_layer_names = list(mamba_specs)
    mamba_group_count = cdiv(len(mamba_layer_names), len(mixed_attention_specs))
    for group_idx in range(mamba_group_count):
        layer_names = mamba_layer_names[group_idx::mamba_group_count]
        group_specs = {name: mamba_specs[name] for name in layer_names}
        uniform_mamba_spec = UniformTypeKVCacheSpecs.from_specs(group_specs)
        assert uniform_mamba_spec is not None
        groups.append(
            KVCacheGroupSpec(
                layer_names=layer_names,
                kv_cache_spec=uniform_mamba_spec,
            )
        )

    logger.info(
        "Using Kimi K3 DSpark mixed KV grouping: %d target + %d draft attention layers, followed by Mamba groups %s",
        len(target_attention_specs),
        len(draft_attention_specs),
        [len(group.layer_names) for group in groups[1:]],
    )
    return groups


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
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(_page_sizes(full_mla_spec))

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            if "mtp" not in name:
                b[specs[name].page_size_bytes].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = specs[name].page_size_bytes
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
    """Materialize DSV4 tuple slots aliased across one global block-ID pool."""
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        tensors = tq_cache.cache_tensors(kv_cache_groups, num_blocks)
    else:
        tensors = _get_deepseek_v4_cache_tensors(kv_cache_groups, num_blocks)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )


def _get_deepseek_v4_cache_tensors(
    kv_cache_groups: list[KVCacheGroupSpec],
    num_blocks: int,
) -> list[KVCacheTensor]:
    """Describe one backing tensor per page-size bucket and tuple slot."""
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

    tensors: list[KVCacheTensor] = []

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

    return tensors
