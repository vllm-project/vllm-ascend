# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections import defaultdict
from dataclasses import replace
from typing import Any

# Both supported versions use the KV-cache planner and builder API from #61183/#42309.
import vllm.v1.core.kv_cache_planning as _kv_cache_planning
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_planning import _approximate_gcd, may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
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
from vllm_ascend.models.kimi_k3.cache_config import _get_kimi_k3_dspark_mixed_kv_cache_groups
from vllm_ascend.models.kimi_k3.paired_sfa_cache import (
    paired_sfa_cache_config,
    paired_sfa_cache_groups,
    paired_sfa_max_memory,
    paired_sfa_pool_bytes,
)
from vllm_ascend.models.qwen4_exp.cache_config import (
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_kv_cache_groups,
    get_qwen4_exp_max_memory_usage,
    get_qwen4_exp_pool_bytes_per_block,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import TURBOQUANT_CACHE_DTYPE
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache
from vllm_ascend.utils import vllm_version_is

_orig_get_kv_cache_groups_uniform_page_size = _kv_cache_planning._get_kv_cache_groups_uniform_page_size
_orig_get_kv_cache_groups = _kv_cache_planning.get_kv_cache_groups
_orig_get_packed_kv_cache_groups = _kv_cache_planning._get_packed_kv_cache_groups
_orig_annotate_eagle_groups = getattr(_kv_cache_planning, "_annotate_eagle_groups", None)
_orig_get_kv_cache_config_from_groups = _kv_cache_planning.get_kv_cache_config_from_groups
_orig_max_memory_usage_bytes_from_groups = _kv_cache_planning._max_memory_usage_bytes_from_groups
_orig_pool_bytes_per_block = _kv_cache_planning._pool_bytes_per_block


def _page_sizes(spec: UniformTypeKVCacheSpecs) -> set[int]:
    """Distinct page sizes across a group's cache specs.

    vLLM #51718 removed ``UniformTypeKVCacheSpecs.get_page_sizes()`` on main;
    the underlying ``kv_cache_specs``/``page_size_bytes`` are unchanged on both
    lanes, so compute it inline.
    """
    return {s.page_size_bytes for s in spec.kv_cache_specs.values()}


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    kimi_k3_groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(kv_cache_spec)
    if kimi_k3_groups is not None:
        return kimi_k3_groups
    return _orig_get_kv_cache_groups_uniform_page_size(kv_cache_spec)


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
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
    """
    Generate the KV cache groups from the grouped specs.
    """
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
    num_layer_tuples_per_group: list[int] = [_get_max_layers_per_page_size(g_spec) for g_spec in grouped_specs]
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
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
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


def _get_max_layers_per_page_size(spec: UniformTypeKVCacheSpecs) -> int:
    """Bridge the UniformTypeKVCacheSpecs helper renamed by vLLM #53896."""
    return spec.get_max_layers_per_page_size()


def _uses_qwen4_exp_mtp(vllm_config: VllmConfig) -> bool:
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return False
    draft_model_config = getattr(
        speculative_config,
        "draft_model_config",
        None,
    )
    architectures = tuple(getattr(draft_model_config, "architectures", ()) or ())
    if "Qwen4ExpMTP" in architectures:
        return True
    text_config = getattr(draft_model_config, "hf_text_config", None)
    text_architectures = tuple(getattr(text_config, "architectures", ()) or ())
    return getattr(text_config, "model_type", None) == "qwen4_exp_mtp" or ("Qwen4ExpMTP" in text_architectures)


def _ascend_annotate_eagle_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    kv_cache_groups: list[KVCacheGroupSpec],
    use_trailing_layer_fallback: bool = False,
) -> None:
    """Annotate Qwen4Exp's actual MTP cache group, not every group.

    Qwen4Exp's MTP attention spec is otherwise indistinguishable from the
    target's spec. Its predictor is registered at the target layer count, so
    use that model-scoped position to find only the group containing the real
    draft layer. Leaving every group unannotated triggers vLLM's flag-all
    fallback, which incorrectly treats target Mamba groups as draft state and
    disables cross-request prefix reuse.
    """
    if _orig_annotate_eagle_groups is not None:
        _orig_annotate_eagle_groups(
            vllm_config,
            kv_cache_spec,
            kv_cache_groups,
            use_trailing_layer_fallback=use_trailing_layer_fallback,
        )
    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is None
        or not speculative_config.use_eagle_block_drop()
        or not _uses_qwen4_exp_mtp(vllm_config)
    ):
        return

    text_config = vllm_config.model_config.hf_text_config
    first_draft_layer = int(text_config.num_hidden_layers)
    num_draft_layers = int(
        getattr(
            getattr(
                speculative_config.draft_model_config,
                "hf_text_config",
                None,
            ),
            "mtp_num_hidden_layers",
            1,
        )
        or 1
    )
    draft_markers = tuple(
        f".layers.{layer_idx}."
        for layer_idx in range(
            first_draft_layer,
            first_draft_layer + num_draft_layers,
        )
    )
    for group in kv_cache_groups:
        if any(marker in f".{layer_name}." for marker in draft_markers for layer_name in group.layer_names):
            group.is_eagle_group = True

    if not any(group.is_eagle_group for group in kv_cache_groups):
        raise ValueError(f"Qwen4Exp MTP is enabled, but no KV cache group contains its draft layer(s) {draft_markers}.")


def _ascend_get_packed_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Preserve Ascend's DSV4 grouping on the live packed-group hook."""
    if is_deepseek_v41_cache(kv_cache_spec):
        grouped_specs = group_cache_specs(kv_cache_spec)
        groups = make_cache_groups(grouped_specs)
    else:
        grouped_specs = group_and_unify_kv_cache_specs(kv_cache_spec)
        if grouped_specs is None:
            assert _orig_get_packed_kv_cache_groups is not None
            return _orig_get_packed_kv_cache_groups(vllm_config, kv_cache_spec)
        groups = _get_kv_cache_groups_uniform_groups(grouped_specs)
    return groups


def _get_deepseek_v4_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[list[int], list[dict[int, list[str]]], list[str], int, int]:
    """Return the geometry shared by DSV4 planning and rank normalization.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
    every layer's page_size matches one of the full-MLA bucket sizes.

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


def _get_kv_cache_config_deepseek_v4_main(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
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

    return num_blocks, tensors


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


def _get_replicated_draft_pools(
    groups: list[KVCacheGroupSpec],
) -> list[list[KVCacheGroupSpec]] | None:
    """Keep contiguous dense draft K/V outside the target/state alias pool."""
    if not any(get_kv_cache_spec_kind(group.kv_cache_spec) == KVCacheSpecKind.MAMBA for group in groups):
        return None
    target_groups = []
    draft_pools = []
    for group in groups:
        spec = group.kv_cache_spec
        specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        if all(
            isinstance(s, FullAttentionSpec) and not isinstance(s, MLAAttentionSpec) and not s.dcp_sharded
            for s in specs
        ):
            draft_pools.append([group])
        else:
            target_groups.append(group)
    if target_groups and draft_pools:
        return [target_groups, *draft_pools]
    return None


def _ascend_pool_bytes_per_block(  # type: ignore[misc]
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Use the same DSV4 divisor as Ascend's shared-tuple planner.

    vLLM #51718 re-plans ranks with more KV memory using
    ``min_num_blocks * _pool_bytes_per_block(groups)``. Its standardized
    per-group layout has a different divisor from Ascend's DSV4 shared-tuple
    layout, so using the upstream value changes ``num_blocks`` during the
    re-plan and leaves ranks inconsistent.
    """
    if vllm_config.model_config.hf_config.model_type == "kimi_k31":
        return paired_sfa_pool_bytes(kv_cache_groups)
    pools = _get_replicated_draft_pools(kv_cache_groups)
    if pools is not None:
        return sum(_orig_pool_bytes_per_block(vllm_config, pool) for pool in pools)
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        return tq_cache.pool_bytes_per_block(kv_cache_groups)
    if is_deepseek_v41_cache(kv_cache_groups):
        return get_deepseek_v41_pool_bytes_per_block(kv_cache_groups)
    if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        return get_glm5_next_pool_bytes_per_block(kv_cache_groups)
    qwen_bytes = get_qwen4_exp_pool_bytes_per_block(kv_cache_groups)
    if qwen_bytes is not None:
        return qwen_bytes
    if not _is_deepseek_v4_groups(kv_cache_groups):
        return _orig_pool_bytes_per_block(vllm_config, kv_cache_groups)

    page_sizes, _, _, _, num_layer_tuples = _get_deepseek_v4_cache_layout(kv_cache_groups)
    return sum(page_sizes) * num_layer_tuples


def _ascend_max_memory_usage_bytes_from_groups(
    builder: Any,
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Keep Ascend admission accounting with the upstream builder API."""
    if vllm_config.model_config.hf_config.model_type == "kimi_k31":
        return paired_sfa_max_memory(vllm_config, kv_cache_groups)
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        return tq_cache.max_memory_usage(vllm_config, kv_cache_groups)
    if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        return get_glm5_next_max_memory_usage(vllm_config, kv_cache_groups)
    qwen_bytes = get_qwen4_exp_max_memory_usage(vllm_config, kv_cache_groups)
    if qwen_bytes is not None:
        return qwen_bytes
    if not _is_deepseek_v4_groups(kv_cache_groups):
        return _orig_max_memory_usage_bytes_from_groups(builder, vllm_config, kv_cache_groups)

    assert all(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in kv_cache_groups)
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    layer_tuple_bytes = sum(_page_sizes(full_mla_spec))
    num_layer_tuples = max(
        _get_max_layers_per_page_size(group.kv_cache_spec)
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    )
    return sum(
        num_layer_tuples * group.kv_cache_spec.max_memory_usage_pages(vllm_config) * layer_tuple_bytes
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    )


def _get_ascend_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    if vllm_config.model_config.hf_config.model_type == "kimi_k31":
        return paired_sfa_cache_groups(vllm_config, kv_cache_spec)
    if any(is_glm5_next_cache_spec(spec) for spec in kv_cache_spec.values()):
        return get_glm5_next_kv_cache_groups(vllm_config, kv_cache_spec)
    groups = get_qwen4_exp_kv_cache_groups(vllm_config, kv_cache_spec, _orig_get_kv_cache_groups)
    if groups is not None:
        return groups
    return _orig_get_kv_cache_groups(vllm_config, kv_cache_spec)


def _ascend_get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Restore Ascend-specific cache planners while retaining transfer metadata.

    Attach ``kv_transfer_config`` onto every built config: ``KVCacheConfig``
    has no native field, and the PD prefill-producer role read back by the
    kv-cache coordinator (see ``patch_kv_cache_coordinator.py``) must ride on
    the config object - the coordinator factory receives ``KVCacheConfig``
    but not ``VllmConfig``. The attribute survives the scheduler-side
    ``copy.deepcopy`` in ``generate_scheduler_kv_cache_config`` and is
    dropped by worker pickle IPC, which never reads it.
    """
    qwen_config = (
        None
        if vllm_version_is("0.28.0")
        else get_qwen4_exp_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    )
    pools = _get_replicated_draft_pools(kv_cache_groups)
    if vllm_config.model_config.hf_config.model_type == "kimi_k31":
        kv_cache_config = paired_sfa_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif pools is not None:
        # Every physical pool shares the scheduler's block count, but owns a
        # disjoint byte range. Use the same aggregate divisor as admission and
        # rank re-planning, then retain upstream's per-pool stride descriptors.
        pool_bytes = [_orig_pool_bytes_per_block(vllm_config, pool) for pool in pools]
        bytes_per_block = sum(pool_bytes)
        num_blocks = may_override_num_blocks(vllm_config, available_memory // bytes_per_block)
        total_size = num_blocks * bytes_per_block
        tensors: list[KVCacheTensor] = []
        offset = 0
        for pool, pool_bytes_per_block in zip(pools, pool_bytes):
            pool_size = num_blocks * pool_bytes_per_block
            pool_config = _orig_get_kv_cache_config_from_groups(vllm_config, pool, pool_size)
            assert pool_config.num_blocks == num_blocks
            tensors.extend(
                replace(tensor, size=total_size, offset=offset + tensor.offset)
                for tensor in pool_config.kv_cache_tensors
            )
            offset += pool_size
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
        )
    elif qwen_config is not None:
        kv_cache_config = qwen_config
    elif is_deepseek_v41_cache(kv_cache_groups):
        kv_cache_config = get_deepseek_v41_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        kv_cache_config = get_glm5_next_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif not _is_deepseek_v4_groups(kv_cache_groups):
        kv_cache_config = _orig_get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)
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
    kv_cache_config.kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None and speculative_config.method == "dspark":
        from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import resident_mla_context_group_ids
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_layout import draft_only_group_ids

        # Upstream scheduler conversion keeps only the first spec in a uniform
        # group, losing both residency and per-layer is_draft ownership. Keep
        # the PD context and external prefix-store classifications separate:
        # a mixed group needs draft block IDs AND ordinary target hit keys.
        # Remove these snapshots once upstream preserves per-layer ownership.
        kv_cache_config.dspark_context_group_ids = resident_mla_context_group_ids(kv_cache_config.kv_cache_groups)
        kv_cache_config.dspark_draft_only_group_ids = draft_only_group_ids(kv_cache_config.kv_cache_groups)
    return kv_cache_config


assert _orig_get_packed_kv_cache_groups is not None
if _orig_annotate_eagle_groups is not None:
    _kv_cache_planning._annotate_eagle_groups = _ascend_annotate_eagle_groups
_kv_cache_planning._get_packed_kv_cache_groups = _ascend_get_packed_kv_cache_groups
_kv_cache_planning._get_kv_cache_groups_uniform_page_size = _get_kv_cache_groups_uniform_page_size
_kv_cache_planning.get_kv_cache_groups = _get_ascend_kv_cache_groups
_kv_cache_planning.get_kv_cache_config_from_groups = _ascend_get_kv_cache_config_from_groups
_kv_cache_planning._max_memory_usage_bytes_from_groups = _ascend_max_memory_usage_bytes_from_groups
_kv_cache_planning._pool_bytes_per_block = _ascend_pool_bytes_per_block
