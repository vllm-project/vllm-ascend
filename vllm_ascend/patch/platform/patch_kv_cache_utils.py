# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from collections import defaultdict
from dataclasses import replace

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_utils import _approximate_gcd, may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendDCPReplicatedDraftAttentionSpec,
)

_KIMI_K3_TARGET_LAYER_PREFIX = "language_model.model.layers."
_KIMI_K3_DRAFT_LAYER_PREFIX = "model.layers."
_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes
_orig_unify_kv_cache_spec_page_size = vllm.v1.core.kv_cache_utils.unify_kv_cache_spec_page_size
_orig_get_kv_cache_groups_uniform_page_size = vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size
_orig_get_kv_cache_config_from_groups = vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups


if UniformTypeKVCacheSpecs.max_num_blocks_per_req is KVCacheSpec.max_num_blocks_per_req:

    def _uniform_type_max_num_blocks_per_req(
        self: UniformTypeKVCacheSpecs,
        vllm_config: VllmConfig,
        max_len: int,
    ) -> int:
        """Preserve the inner spec's block-table width."""
        widths = {spec.max_num_blocks_per_req(vllm_config, max_len) for spec in self.kv_cache_specs.values()}
        assert len(widths) == 1, (
            "All layers in the same KV cache group must need the same number "
            f"of block table entries, got {sorted(widths)}."
        )
        return next(iter(widths))

    UniformTypeKVCacheSpecs.max_num_blocks_per_req = (  # type: ignore[method-assign]
        _uniform_type_max_num_blocks_per_req
    )


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
    if len({spec.block_size for spec in attention_specs}) != 1:
        return None

    base_page_sizes = {spec.page_size_bytes for spec in [*target_attention_specs.values(), *mamba_specs.values()]}
    if len(base_page_sizes) != 1:
        return None
    base_page_size = next(iter(base_page_sizes))
    for spec in draft_attention_specs.values():
        if isinstance(spec, AscendDCPReplicatedDraftAttentionSpec):
            if spec.page_size_bytes != (base_page_size * spec.dcp_replication_size):
                return None
        elif spec.page_size_bytes != base_page_size:
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


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    kimi_k3_groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(kv_cache_spec)
    if kimi_k3_groups is not None:
        return kimi_k3_groups
    return _orig_get_kv_cache_groups_uniform_page_size(kv_cache_spec)


def _unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """Align each replicated draft lane, preserving non-rectangular pages."""
    replicated_specs = {
        name: spec for name, spec in kv_cache_spec.items() if isinstance(spec, AscendDCPReplicatedDraftAttentionSpec)
    }
    if replicated_specs:
        # First ask vLLM to align only the target's ordinary pages.  The draft's
        # DCP multiplier describes additional physical storage, not a larger
        # scheduler page.  Feeding either the multiplied draft page or its GQA
        # page into the generic unifier can change the logical block size.
        ordinary_specs = {name: spec for name, spec in kv_cache_spec.items() if name not in replicated_specs}
        try:
            aligned_ordinary_specs = _orig_unify_kv_cache_spec_page_size(ordinary_specs)
        except NotImplementedError:
            aligned_ordinary_specs = {}
        ordinary_page_sizes = {spec.page_size_bytes for spec in aligned_ordinary_specs.values()}
        if len(ordinary_page_sizes) == 1:
            base_page_size = next(iter(ordinary_page_sizes))
            aligned_specs = {
                name: (
                    replace(
                        kv_cache_spec[name],
                        page_size_padded=base_page_size,
                    )
                    if name in replicated_specs
                    else spec
                )
                for name, spec in {
                    **aligned_ordinary_specs,
                    **replicated_specs,
                }.items()
            }
            if _get_kimi_k3_dspark_mixed_kv_cache_groups(aligned_specs) is not None:
                return aligned_specs
    return _orig_unify_kv_cache_spec_page_size(kv_cache_spec)


def _kv_cache_config_has_mamba_layers(self: KVCacheConfig) -> bool:
    """Recognize Mamba layers nested in UniformType cache groups."""
    return any(get_kv_cache_spec_kind(group.kv_cache_spec) == KVCacheSpecKind.MAMBA for group in self.kv_cache_groups)


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


def _get_kimi_k3_replicated_dspark_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig | None:
    """Plan the minimal K3 target-sharded / DSpark-replicated layout.

    Target attention pages keep aliasing the balanced Mamba groups. Each
    replicated draft layer gets one independent DCP-sized tensor, avoiding a
    rectangular allocation that would pad every target layer to draft size.
    """
    if not kv_cache_groups:
        return None
    first_group_spec = kv_cache_groups[0].kv_cache_spec
    if not isinstance(first_group_spec, UniformTypeKVCacheSpecs):
        return None
    first_specs = first_group_spec.kv_cache_specs
    draft_layers = [
        name
        for name in kv_cache_groups[0].layer_names
        if isinstance(
            first_specs[name],
            AscendDCPReplicatedDraftAttentionSpec,
        )
    ]
    if not draft_layers:
        return None
    target_layers = [name for name in kv_cache_groups[0].layer_names if name not in draft_layers]
    if not target_layers or any(not isinstance(first_specs[name], FullAttentionSpec) for name in target_layers):
        return None

    bytes_per_block = sum(first_specs[name].page_size_bytes for name in kv_cache_groups[0].layer_names)
    if any(
        sum(
            group.kv_cache_spec.kv_cache_specs[name].page_size_bytes
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else group.kv_cache_spec.page_size_bytes
            for name in group.layer_names
        )
        > bytes_per_block
        for group in kv_cache_groups[1:]
    ):
        return None
    num_blocks = may_override_num_blocks(
        vllm_config,
        available_memory // bytes_per_block,
    )

    recurrent_layers_by_group = [list(group.layer_names) for group in kv_cache_groups[1:]]
    kv_cache_tensors: list[KVCacheTensor] = []
    for layer_idx, target_layer in enumerate(target_layers):
        page_size = first_specs[target_layer].page_size_bytes
        layers = [target_layer]
        for recurrent_layers in recurrent_layers_by_group:
            if layer_idx < len(recurrent_layers):
                layers.append(recurrent_layers[layer_idx])
        kv_cache_tensors.append(
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=layers,
            )
        )
    for draft_layer in draft_layers:
        page_size = first_specs[draft_layer].page_size_bytes
        kv_cache_tensors.append(
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=[draft_layer],
            )
        )

    logger.info(
        "Using Kimi K3 DCP-replicated DSpark KV layout: %d logical blocks, "
        "%d target tensors and %d replicated draft tensors",
        num_blocks,
        len(target_layers),
        len(draft_layers),
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=(vllm_config.cache_config.prefix_cache_retention_interval),
    )


def _get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    config = _get_kimi_k3_replicated_dspark_kv_cache_config(
        vllm_config,
        kv_cache_groups,
        available_memory,
    )
    if config is not None:
        return config
    return _orig_get_kv_cache_config_from_groups(
        vllm_config,
        kv_cache_groups,
        available_memory,
    )


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.core.kv_cache_utils.unify_kv_cache_spec_page_size = _unify_kv_cache_spec_page_size
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size = _get_kv_cache_groups_uniform_page_size
vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups = _get_kv_cache_config_from_groups
# vLLM v0.24.0 renamed _get_kv_cache_config_deepseek_v4 to _get_kv_cache_config_packed and
# get_kv_cache_config_from_groups now calls _get_kv_cache_config_packed directly, bypassing
# the alias patch above. Patch the canonical name so Ascend's non-packed layout is used.
vllm.v1.core.kv_cache_utils._get_kv_cache_config_packed = _get_kv_cache_config_deepseek_v4
KVCacheConfig.has_mamba_layers = property(  # type: ignore[assignment]
    _kv_cache_config_has_mamba_layers
)

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
