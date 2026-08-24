# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from collections import defaultdict

import torch
import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, round_up
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

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes

_orig_get_kv_cache_groups = vllm.v1.core.kv_cache_utils.get_kv_cache_groups

_orig_get_kv_cache_config_from_groups = vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups

_orig_pool_bytes_per_block = vllm.v1.core.kv_cache_utils._pool_bytes_per_block

_orig_max_memory_usage_bytes_from_groups = vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups

_orig_get_max_concurrency_for_kv_cache_config = vllm.v1.core.kv_cache_utils.get_max_concurrency_for_kv_cache_config

_orig_generate_scheduler_kv_cache_config = vllm.v1.core.kv_cache_utils.generate_scheduler_kv_cache_config


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


def _is_kimi_k3_dspark_config(vllm_config: VllmConfig) -> bool:
    """Return whether the config selects the supported K3 DSpark v1 path."""
    if getattr(vllm_config, "use_v2_model_runner", False):
        return False

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if getattr(scheduler_config, "disable_hybrid_kv_cache_manager", False):
        return False

    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None or getattr(speculative_config, "method", None) != "dspark":
        return False

    model_config = getattr(vllm_config, "model_config", None)
    target_text_config = getattr(model_config, "hf_text_config", None)
    target_hf_config = getattr(model_config, "hf_config", None)
    is_kimi_k3_target = (
        getattr(target_text_config, "model_type", None) == "kimi_linear"
        and getattr(target_text_config, "attn_res_block_size", None) is not None
    ) or getattr(target_hf_config, "model_type", None) == "kimi_k3"
    if not is_kimi_k3_target:
        return False

    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    draft_hf_config = getattr(draft_model_config, "hf_config", None)
    return getattr(draft_hf_config, "model_type", None) == "k3_dspark"


def _is_kimi_k3_c8_target_spec(spec: KVCacheSpec, main_page_size: int) -> bool:
    return (
        isinstance(spec, AscendMLAAttentionSpec)
        and spec.dtype == torch.int8
        and spec.compress_ratio == 1
        and spec.model_version is None
        and spec.sliding_window is None
        and spec.attention_chunk_size is None
        and not spec.indexes_kv_by_block_stride
        and spec.page_size_bytes == main_page_size
        and spec.page_size_padded == main_page_size
        and spec.unpadded_page_size_bytes < main_page_size
    )


def _is_kimi_k3_bf16_draft_spec(spec: KVCacheSpec, main_page_size: int) -> bool:
    return (
        isinstance(spec, AscendMLAAttentionSpec)
        and spec.dtype == torch.bfloat16
        and spec.compress_ratio == 1
        and spec.model_version is None
        and spec.sliding_window is None
        and spec.attention_chunk_size is None
        and not spec.indexes_kv_by_block_stride
        and spec.page_size_bytes > main_page_size
        and spec.page_size_padded is None
    )


def _get_kimi_k3_c8_dspark_spec_partition(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> tuple[int, dict[str, KVCacheSpec], dict[str, KVCacheSpec]] | None:
    """Validate and partition the narrow K3 C8-target/BF16-draft layout."""
    if not _is_kimi_k3_dspark_config(vllm_config) or not kv_cache_spec:
        return None

    if any(not isinstance(spec, (AscendMLAAttentionSpec, MambaSpec)) for spec in kv_cache_spec.values()):
        return None

    block_sizes = {spec.block_size for spec in kv_cache_spec.values()}
    if len(block_sizes) != 1:
        return None

    mamba_specs = [spec for spec in kv_cache_spec.values() if isinstance(spec, MambaSpec)]
    if not mamba_specs:
        return None
    mamba_page_sizes = {spec.page_size_bytes for spec in mamba_specs}
    if len(mamba_page_sizes) != 1:
        return None
    main_page_size = next(iter(mamba_page_sizes))
    if any(spec.page_size_padded != main_page_size or spec.mamba_cache_mode != "align" for spec in mamba_specs):
        return None

    target_specs: dict[str, KVCacheSpec] = {}
    draft_specs: dict[str, KVCacheSpec] = {}
    for layer_name, spec in kv_cache_spec.items():
        if isinstance(spec, MambaSpec):
            continue
        if _is_kimi_k3_c8_target_spec(spec, main_page_size):
            target_specs[layer_name] = spec
        elif _is_kimi_k3_bf16_draft_spec(spec, main_page_size):
            draft_specs[layer_name] = spec
        else:
            return None

    if not target_specs or not draft_specs:
        return None
    return main_page_size, target_specs, draft_specs


def _try_get_kimi_k3_c8_dspark_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    partition = _get_kimi_k3_c8_dspark_spec_partition(vllm_config, kv_cache_spec)
    if partition is None:
        return None
    _, target_specs, draft_specs = partition

    # Keep the target/Mamba page layout intact. The draft layers share the
    # target's logical block table, but remain distinct physical page buckets.
    base_specs = {layer_name: spec for layer_name, spec in kv_cache_spec.items() if layer_name not in draft_specs}
    groups = vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size(base_specs)
    target_group_idx = next(
        (
            idx
            for idx, group in enumerate(groups)
            if isinstance(group.kv_cache_spec, AscendMLAAttentionSpec)
            and any(layer_name in target_specs for layer_name in group.layer_names)
        ),
        None,
    )
    if target_group_idx is None:
        return None

    target_group = groups[target_group_idx]
    logical_layer_names = [*target_group.layer_names, *draft_specs]
    logical_specs = {layer_name: kv_cache_spec[layer_name] for layer_name in logical_layer_names}
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(logical_specs)
    if uniform_spec is None:
        return None
    groups[target_group_idx] = KVCacheGroupSpec(
        layer_names=logical_layer_names,
        kv_cache_spec=uniform_spec,
        is_eagle_group=True,
    )
    vllm.v1.core.kv_cache_utils.logger.info_once(
        "Using the Kimi-K3 C8 target/BF16 DSpark hybrid KV cache layout: "
        "draft layers share a target block table and use independent page buckets."
    )
    return groups


def get_kv_cache_groups(vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
    kimi_k3_groups = _try_get_kimi_k3_c8_dspark_groups(vllm_config, kv_cache_spec)
    if kimi_k3_groups is not None:
        return kimi_k3_groups

    try:
        return _orig_get_kv_cache_groups(vllm_config, kv_cache_spec)
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
        return fallback_groups


def _get_kimi_k3_c8_dspark_main_page_size(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    *,
    allow_scheduler_layout: bool = False,
) -> int | None:
    """Recognize the global or PP-projected K3 hybrid group structure."""
    if not _is_kimi_k3_dspark_config(vllm_config) or not kv_cache_groups:
        return None

    block_sizes = {group.kv_cache_spec.block_size for group in kv_cache_groups}
    if len(block_sizes) != 1:
        return None

    mamba_specs = [group.kv_cache_spec for group in kv_cache_groups if isinstance(group.kv_cache_spec, MambaSpec)]
    if not mamba_specs:
        return None
    main_page_sizes = {spec.page_size_bytes for spec in mamba_specs}
    if len(main_page_sizes) != 1:
        return None
    main_page_size = next(iter(main_page_sizes))
    if any(spec.page_size_padded != main_page_size or spec.mamba_cache_mode != "align" for spec in mamba_specs):
        return None

    has_uniform_group = False
    has_eagle_attention_group = False
    has_mla_spec = False
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, MambaSpec):
            continue
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            has_uniform_group = True
            if not set(group.layer_names).issubset(group_spec.kv_cache_specs):
                return None
            inner_specs = group_spec.kv_cache_specs.values()
            for inner_spec in inner_specs:
                if not (
                    _is_kimi_k3_c8_target_spec(inner_spec, main_page_size)
                    or _is_kimi_k3_bf16_draft_spec(inner_spec, main_page_size)
                ):
                    return None
                has_mla_spec = True
            continue
        is_target_spec = _is_kimi_k3_c8_target_spec(group_spec, main_page_size)
        is_draft_scheduler_representative = (
            allow_scheduler_layout and group.is_eagle_group and _is_kimi_k3_bf16_draft_spec(group_spec, main_page_size)
        )
        if not (is_target_spec or is_draft_scheduler_representative):
            return None
        has_eagle_attention_group |= group.is_eagle_group
        has_mla_spec = True

    is_supported_layout = has_uniform_group or (allow_scheduler_layout and has_eagle_attention_group)
    return main_page_size if is_supported_layout and has_mla_spec else None


def _get_kimi_k3_c8_dspark_page_buckets(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> list[tuple[int, list[list[str]]]] | None:
    """Return ordinary, independently allocated page buckets in layout order."""
    main_page_size = _get_kimi_k3_c8_dspark_main_page_size(vllm_config, kv_cache_groups)
    if main_page_size is None:
        return None

    buckets = vllm.v1.core.kv_cache_utils._bucket_layers_by_page_size(kv_cache_groups)
    if not buckets:
        return None
    if any(page_size < main_page_size for page_size in buckets):
        return None

    # Keep the existing target/Mamba family first for layout compatibility,
    # followed by larger draft buckets. These are ordinary tensors, not aliases
    # into a packed backing.
    ordered_page_sizes = []
    if main_page_size in buckets:
        ordered_page_sizes.append(main_page_size)
    ordered_page_sizes.extend(sorted(page_size for page_size in buckets if page_size != main_page_size))
    return [(page_size, buckets[page_size]) for page_size in ordered_page_sizes]


def _kimi_k3_c8_dspark_pool_bytes_per_block(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int | None:
    buckets = _get_kimi_k3_c8_dspark_page_buckets(vllm_config, kv_cache_groups)
    if buckets is None:
        return None
    return sum(page_size * len(slots) for page_size, slots in buckets)


def _pool_bytes_per_block(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    bytes_per_block = _kimi_k3_c8_dspark_pool_bytes_per_block(vllm_config, kv_cache_groups)
    if bytes_per_block is not None:
        return bytes_per_block
    return _orig_pool_bytes_per_block(vllm_config, kv_cache_groups)


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    buckets = _get_kimi_k3_c8_dspark_page_buckets(vllm_config, kv_cache_groups)
    if buckets is None:
        return _orig_get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)

    bytes_per_block = sum(page_size * len(slots) for page_size, slots in buckets)
    num_blocks = may_override_num_blocks(vllm_config, available_memory // bytes_per_block)
    kv_cache_tensors = [
        KVCacheTensor(
            size=page_size * num_blocks,
            shared_by=shared_by,
        )
        for page_size, slots in buckets
        for shared_by in slots
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def _kimi_k3_c8_dspark_blocks_per_request(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    blocks_per_request = 0
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            blocks_per_request += spec.max_memory_usage_pages(vllm_config)
        else:
            blocks_per_request += cdiv(
                spec.max_memory_usage_bytes(vllm_config),
                spec.page_size_bytes,
            )
    return blocks_per_request


def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    bytes_per_block = _kimi_k3_c8_dspark_pool_bytes_per_block(vllm_config, kv_cache_groups)
    if bytes_per_block is None:
        return _orig_max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
    blocks_per_request = _kimi_k3_c8_dspark_blocks_per_request(vllm_config, kv_cache_groups)
    return bytes_per_block * blocks_per_request


def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> float:
    if (
        _get_kimi_k3_c8_dspark_main_page_size(
            vllm_config,
            kv_cache_config.kv_cache_groups,
            allow_scheduler_layout=True,
        )
        is None
    ):
        return _orig_get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)
    blocks_per_request = _kimi_k3_c8_dspark_blocks_per_request(
        vllm_config,
        kv_cache_config.kv_cache_groups,
    )
    return kv_cache_config.num_blocks / blocks_per_request


def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """Preserve an EAGLE group present on any PP worker projection."""
    scheduler_config = _orig_generate_scheduler_kv_cache_config(kv_cache_configs)
    num_groups = len(scheduler_config.kv_cache_groups)
    assert all(len(config.kv_cache_groups) == num_groups for config in kv_cache_configs)
    for group_id, scheduler_group in enumerate(scheduler_config.kv_cache_groups):
        scheduler_group.is_eagle_group = any(
            config.kv_cache_groups[group_id].is_eagle_group for config in kv_cache_configs
        )
    return scheduler_config


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
vllm.v1.core.kv_cache_utils._pool_bytes_per_block = _pool_bytes_per_block
vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups = _max_memory_usage_bytes_from_groups
vllm.v1.core.kv_cache_utils.get_max_concurrency_for_kv_cache_config = get_max_concurrency_for_kv_cache_config
vllm.v1.core.kv_cache_utils.generate_scheduler_kv_cache_config = generate_scheduler_kv_cache_config
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups
# vLLM v0.24.0 renamed _get_kv_cache_config_deepseek_v4 to _get_kv_cache_config_packed and
# get_kv_cache_config_from_groups now calls _get_kv_cache_config_packed directly, bypassing
# the alias patch above. Patch the canonical name so Ascend's non-packed layout is used.
vllm.v1.core.kv_cache_utils._get_kv_cache_config_packed = _get_kv_cache_config_deepseek_v4

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.engine.core.generate_scheduler_kv_cache_config = generate_scheduler_kv_cache_config
