# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from collections import defaultdict

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_utils import _approximate_gcd, may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.six_region_kv_cache_layout import (
    HIDDEN,
    build_six_region_kv_cache_layout,
)

_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes

_orig_get_kv_cache_groups = vllm.v1.core.kv_cache_utils.get_kv_cache_groups
_orig_get_kv_cache_config_from_groups = vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups
_orig_max_memory_usage_bytes_from_groups = vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support DCP.
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with CP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp to maintain alignment.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp
        return bs, bs

    if dcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by the CP factors for proper alignment.
        group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp
        if not cache_config.enable_prefix_caching:
            return scheduler_block_size, scheduler_block_size
        hash_block_size = math.gcd(*group_block_sizes)
        return scheduler_block_size, hash_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def _try_get_full_allocation_fallback_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    if any(isinstance(spec, HiddenStateCacheSpec) for spec in kv_cache_spec.values()):
        return None
    if any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    has_mla = any(isinstance(spec, MLAAttentionSpec) for spec in kv_cache_spec.values())
    has_regular_swa = any(isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values())
    if not (has_mla and has_regular_swa):
        return None

    full_block_sizes = {spec.block_size for spec in kv_cache_spec.values() if isinstance(spec, FullAttentionSpec)}
    full_attention_block_size = next(iter(full_block_sizes)) if len(full_block_sizes) == 1 else None
    promoted_specs = kv_cache_spec.copy()
    for layer_name, spec in kv_cache_spec.items():
        if not isinstance(spec, SlidingWindowSpec):
            continue
        page_size_padded = None
        block_size = full_attention_block_size or spec.block_size
        if spec.page_size_padded is not None:
            unpadded_page_size = spec.unpadded_page_size_bytes * block_size // spec.block_size
            page_size_padded = max(spec.page_size_padded, unpadded_page_size)
        promoted_specs[layer_name] = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            head_size_v=spec.head_size_v,
            dtype=spec.dtype,
            kv_quant_mode=spec.kv_quant_mode,
            page_size_padded=page_size_padded,
            indexes_kv_by_block_stride=spec.indexes_kv_by_block_stride,
            sliding_window=spec.sliding_window,
        )

    uniform_spec = UniformTypeKVCacheSpecs.from_specs(promoted_specs)
    if uniform_spec is None:
        return None
    vllm.v1.core.kv_cache_utils.logger.warning(
        "KV cache page sizes cannot be unified; treating sliding-window "
        "layers as full attention for cache allocation. Sliding-window "
        "attention compute is unchanged."
    )
    return vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_type(uniform_spec)


def _qsa_source_name(layer_name: str, suffix: str) -> str:
    if not layer_name.endswith(suffix):
        raise ValueError(f"Invalid QSA cache owner {layer_name!r}")
    return layer_name[: -len(suffix)]


def _prepare_qwen4_exp_qsa_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> tuple[set[str], set[str], set[str]] | None:
    """Align QSA geometry before grouping and identify composite owners."""
    from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec

    raw = {
        _qsa_source_name(name, ".indexer.raw_key_cache"): (name, spec)
        for name, spec in kv_cache_spec.items()
        if name.endswith(".indexer.raw_key_cache")
    }
    if not raw:
        return None
    main = {
        _qsa_source_name(name, ".attn"): (name, spec)
        for name, spec in kv_cache_spec.items()
        if name.endswith(".attn") and isinstance(spec, FullAttentionSpec)
    }
    compressed = {
        _qsa_source_name(name, ".indexer.compressed_key_cache"): (name, spec)
        for name, spec in kv_cache_spec.items()
        if name.endswith(".indexer.compressed_key_cache")
    }
    if set(main) != set(raw) or set(main) != set(compressed):
        raise ValueError("QSA main/raw/compressed owners must have identical source layers")
    gdn_specs = [spec for spec in kv_cache_spec.values() if isinstance(spec, MambaSpec) and len(spec.shapes) == 2]
    if not gdn_specs:
        raise ValueError("QSA six-slab layout requires GDN state specs")
    max_ssm_bytes = max(math.prod(spec.shapes[1]) * get_dtype_size(spec.dtypes[1]) for spec in gdn_specs)
    main_names: set[str] = set()
    compressed_names: set[str] = set()
    raw_names: set[str] = set()
    for source in sorted(main):
        main_name, main_spec = main[source]
        compressed_name, compressed_spec = compressed[source]
        raw_name, raw_spec = raw[source]
        if not isinstance(main_spec, FullAttentionSpec):
            raise ValueError(f"{main_name} is not FullAttentionSpec")
        if not isinstance(compressed_spec, MLAAttentionSpec):
            raise ValueError(f"{compressed_name} is not MLAAttentionSpec")
        if not isinstance(raw_spec, AscendCircularBufferSpec):
            raise ValueError(f"{raw_name} is not AscendCircularBufferSpec")
        k_token_bytes = main_spec.num_kv_heads * main_spec.head_size * get_dtype_size(main_spec.dtype)
        if max_ssm_bytes % k_token_bytes:
            raise ValueError(f"GDN SSM page {max_ssm_bytes} is not integral QSA K tokens")
        target = max(main_spec.block_size, max_ssm_bytes // k_token_bytes)
        if target % 128 or target % compressed_spec.compress_ratio:
            raise ValueError(f"QSA aligned block_size={target} violates kernel/compression alignment")
        object.__setattr__(main_spec, "block_size", target)
        object.__setattr__(compressed_spec, "block_size", target)
        if raw_spec.block_size % compressed_spec.compress_ratio:
            raise ValueError(f"{raw_name} capacity={raw_spec.block_size} is not ratio-aligned")
        main_names.add(main_name)
        compressed_names.add(compressed_name)
        raw_names.add(raw_name)
    return main_names, compressed_names, raw_names


def _merge_qsa_composite_group(
    groups: list[KVCacheGroupSpec],
    kv_cache_spec: dict[str, KVCacheSpec],
    main_names: set[str],
    compressed_names: set[str],
    raw_names: set[str],
) -> list[KVCacheGroupSpec]:
    """Give main K/V and compressed index state one block table/physical ID."""
    composite_names = main_names | compressed_names
    consumed: list[int] = []
    eagle = False
    for index, group in enumerate(groups):
        members = set(group.layer_names)
        overlap = members & composite_names
        if not overlap:
            continue
        if not members <= composite_names:
            raise ValueError("QSA composite owners were mixed with another cache role")
        consumed.append(index)
        eagle = eagle or group.is_eagle_group
    covered = set().union(*(set(groups[i].layer_names) for i in consumed))
    if covered != composite_names:
        raise ValueError("QSA composite grouping lost main or compressed owners")
    ordered = [name for name in kv_cache_spec if name in composite_names]
    uniform = UniformTypeKVCacheSpecs.from_specs({name: kv_cache_spec[name] for name in ordered})
    if uniform is None:
        raise ValueError("QSA main/compressed owners do not have one lifetime")
    merged = KVCacheGroupSpec(ordered, uniform, is_eagle_group=eagle)
    first = min(consumed)
    result = [
        merged if index == first else group
        for index, group in enumerate(groups)
        if index == first or index not in consumed
    ]

    raw_consumed = [
        index
        for index, group in enumerate(result)
        if set(group.layer_names) <= raw_names and set(group.layer_names) & raw_names
    ]
    covered_raw = set().union(*(set(result[index].layer_names) for index in raw_consumed))
    if covered_raw != raw_names:
        raise ValueError("QSA raw grouping lost circular owners")
    ordered_raw = [name for name in kv_cache_spec if name in raw_names]
    raw_uniform = UniformTypeKVCacheSpecs.from_specs({name: kv_cache_spec[name] for name in ordered_raw})
    if raw_uniform is None:
        raise ValueError("QSA raw owners do not have one circular lifetime")
    raw_eagle = any(result[index].is_eagle_group for index in raw_consumed)
    raw_merged = KVCacheGroupSpec(ordered_raw, raw_uniform, is_eagle_group=raw_eagle)
    raw_first = min(raw_consumed)
    result = [
        raw_merged if index == raw_first else group
        for index, group in enumerate(result)
        if index == raw_first or index not in raw_consumed
    ]
    return result


def get_kv_cache_groups(vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
    qsa_roles = _prepare_qwen4_exp_qsa_groups(kv_cache_spec)
    try:
        groups = _orig_get_kv_cache_groups(vllm_config, kv_cache_spec)
    except NotImplementedError as exc:
        fallback_groups = _try_get_full_allocation_fallback_groups(kv_cache_spec)
        if fallback_groups is None:
            spec_summary = sorted(
                {
                    (type(spec).__name__, spec.block_size, spec.page_size_bytes, spec.page_size_padded)
                    for spec in kv_cache_spec.values()
                }
            )
            raise NotImplementedError(
                "KV cache page-size unification failed and the vllm-ascend "
                "full-allocation fallback could not build a uniform KV cache "
                "group. Detected specs are listed as "
                "(type, block_size, page_size_bytes, page_size_padded): "
                f"{spec_summary}."
            ) from exc
        groups = fallback_groups
    if qsa_roles is None:
        return groups
    return _merge_qsa_composite_group(groups, kv_cache_spec, qsa_roles[0], qsa_roles[1], qsa_roles[2])


def _group_member_specs(group: KVCacheGroupSpec) -> dict[str, KVCacheSpec]:
    if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
        return group.kv_cache_spec.kv_cache_specs
    return {name: group.kv_cache_spec for name in group.layer_names}


def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Account for heterogeneous pages inside the QSA composite group."""
    probe = build_six_region_kv_cache_layout(kv_cache_groups, num_blocks=1)
    if probe is None:
        return _orig_max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
    hidden_bytes_per_block = sum(owner.spec.page_size_bytes for owner in probe.owners if owner.role == HIDDEN)
    bytes_per_pool_block = probe.slot_count * sum(r.page_size_bytes for r in probe.regions) + hidden_bytes_per_block
    required_pool_blocks = sum(
        cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            group.kv_cache_spec.page_size_bytes,
        )
        for group in kv_cache_groups
    )
    return required_pool_blocks * bytes_per_pool_block


def _get_qwen4_exp_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig | None:
    """Build one six-slab backing per ordinal layer slot."""
    probe = build_six_region_kv_cache_layout(kv_cache_groups, num_blocks=1)
    if probe is None:
        return None

    hidden_owners = [owner for owner in probe.owners if owner.role == HIDDEN]
    hidden_bytes_per_block = sum(owner.spec.page_size_bytes for owner in hidden_owners)
    slab_bytes_per_block = sum(region.page_size_bytes for region in probe.regions)
    bytes_per_block = probe.slot_count * slab_bytes_per_block + hidden_bytes_per_block
    candidate = available_memory // bytes_per_block
    while candidate > 0:
        candidate_layout = build_six_region_kv_cache_layout(kv_cache_groups, num_blocks=candidate)
        assert candidate_layout is not None
        required = candidate_layout.slot_count * candidate_layout.slot_backing_size + hidden_bytes_per_block * candidate
        if required <= available_memory:
            break
        candidate -= 1
    num_blocks = may_override_num_blocks(vllm_config, candidate)
    layout = build_six_region_kv_cache_layout(kv_cache_groups, num_blocks=num_blocks)
    assert layout is not None

    tensors = [
        KVCacheTensor(
            size=layout.slot_backing_size,
            shared_by=layout.slot_shared_by(slot),
        )
        for slot in range(layout.slot_count)
    ]
    tensors.extend(
        KVCacheTensor(
            size=owner.spec.page_size_bytes * num_blocks,
            shared_by=[owner.layer_name],
        )
        for owner in hidden_owners
    )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=kv_cache_groups,
    )


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    qwen_config = _get_qwen4_exp_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    if qwen_config is not None:
        return qwen_config
    return _orig_get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    ratio_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            ratio_specs[spec.compress_ratio][name] = spec

    mla_uniform_specs = []
    for ratio in sorted(ratio_specs, key=lambda r: (r != 4, r)):
        spec_dict = ratio_specs[ratio]
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
    num_layer_tuples_per_group: list[int] = [g_spec.get_num_layer_tuples() for g_spec in grouped_specs]
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
    all_page_sizes = full_mla_spec.get_page_sizes()
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = sm_spec.get_page_sizes()
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


def _get_kv_cache_config_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """DeepseekV4 KV cache tensor layout planning.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
    every layer's page_size matches one of the full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each
    layer at tuple_idx = position-within-bucket. Emit one KVCacheTensor
    per (tuple_idx, bucket) whose shared_by is the union of per-group
    layers at that slot.
    """
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(full_mla_spec.get_page_sizes())
    layer_tuple_page_bytes = sum(page_sizes)

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

    num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    kv_cache_tensors: list[KVCacheTensor] = []
    for tuple_idx in range(num_layer_tuples - len(mtp_layer_names)):
        for ps in page_sizes:
            shared_by: list[str] = []
            for b in bucketed:
                bucket = b.get(ps)
                if bucket is not None and tuple_idx < len(bucket):
                    shared_by.append(bucket[tuple_idx])
            kv_cache_tensors.append(KVCacheTensor(size=ps * num_blocks, shared_by=shared_by))
    for i in range(len(mtp_layer_names)):
        kv_cache_tensors.append(KVCacheTensor(size=mtp_page_size * num_blocks, shared_by=[mtp_layer_names[i]]))

    return num_blocks, kv_cache_tensors


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.core.kv_cache_utils.get_kv_cache_groups = get_kv_cache_groups
vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups = _max_memory_usage_bytes_from_groups
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups
# vLLM v0.24.0 renamed _get_kv_cache_config_deepseek_v4 to _get_kv_cache_config_packed and
# get_kv_cache_config_from_groups now calls _get_kv_cache_config_packed directly, bypassing
# the alias patch above. Patch the canonical name so Ascend's non-packed layout is used.
vllm.v1.core.kv_cache_utils._get_kv_cache_config_packed = _get_kv_cache_config_deepseek_v4

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
