# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Shared-tuple cache planning geometry for DeepSeek V4.

RFC #21053 (standardize KV cache allocation process) asks the Planning stage
to expose one explicit layout that every downstream consumer reads from,
instead of each caller re-deriving the same geometry. This module is that
single source of truth for DeepSeek V4's shared-tuple page layout: the
full-MLA group, the C128 group, and every SWA sub-group place their layers
into physical tuple slots that are shared across page-size buckets, with
MTP/draft layers occupying trailing slots of their own.

Both supported lanes build on this layout:
- ``vllm_ascend/core/kv_cache_config_builder.py`` (the pluggable
  ``AscendKVCacheConfigBuilder``, used on vLLM main);
- ``vllm_ascend/patch/platform/patch_kv_cache_utils.py`` (the monkey-patch
  lane used on the release branch).

so the grouping, detection, and pool-accounting logic can never drift
between the two lanes. KVCacheTensor descriptor materialization (the actual
offsets/strides into the physical backing) stays in
``patch_kv_cache_utils.py``: it is Planning-stage *output*, built once per
process from this layout, not duplicated geometry itself.

This module intentionally does not touch Kimi-K3's cache planning: Kimi-K3
uses its own mixed-group layout (``vllm_ascend.models.kimi_k3``) and never
constructs or consumes a ``DeepseekV4CacheLayout``.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_planning import _approximate_gcd
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.models.deepseek_v41.cache_config import is_deepseek_v41_cache
from vllm_ascend.quantization.methods.kv_cache.turboquant import TURBOQUANT_CACHE_DTYPE
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache


@dataclass(frozen=True)
class DeepseekV4CacheLayout:
    """Shared-tuple geometry for a DeepSeek V4 set of KV cache groups.

    Every group's layers are bucketed by physical ``page_size_bytes`` (the
    page-size classes shared across groups, padded upstream so non-full-MLA
    groups line up with a full-MLA bucket); within a bucket, a layer's index
    is its tuple slot. MTP/draft layers are not bucketed by page size - they
    occupy the trailing tuple slots, after every ordinary bucket, at a
    single page size of their own.
    """

    #: Distinct physical page sizes, sorted ascending. These are the
    #: page-size buckets every group's layers are partitioned into.
    page_sizes: tuple[int, ...]
    #: Per-group bucketing: ``bucketed[g][page_size]`` is the ordered tuple
    #: of layer names that group places in that page-size bucket.
    bucketed: tuple[Mapping[int, tuple[str, ...]], ...]
    #: MTP/draft layer names, in registration order, placed at the
    #: trailing tuple slots.
    mtp_layer_names: tuple[str, ...]
    #: The (single) physical page size MTP/draft layers use.
    mtp_page_size: int
    #: Total tuple slot count: the longest per-page-size bucket across every
    #: group, plus one trailing slot per MTP/draft layer.
    num_layer_tuples: int

    @property
    def bytes_per_tuple(self) -> int:
        """Bytes one physical tuple slot occupies across every page bucket."""
        return sum(self.page_sizes)

    @property
    def pool_bytes_per_block(self) -> int:
        """Bytes one global block ID consumes across the shared tuple slots.

        vLLM re-plans the ranks with more KV memory against this value. The
        standardized per-group layout has a different divisor from Ascend's
        shared-tuple layout, so deriving it from the tuple geometry keeps
        every rank's ``num_blocks`` consistent.
        """
        return self.bytes_per_tuple * self.num_layer_tuples


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


def get_deepseek_v4_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> DeepseekV4CacheLayout:
    """Return the geometry shared by DSV4 planning and memory accounting.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see ``_get_kv_cache_groups_uniform_groups``)
    so every layer's page_size matches one of the full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each layer
    at tuple_idx = position-within-bucket.
    """
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    sorted_page_sizes = sorted(_page_sizes(full_mla_spec))

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names: list[str] = []
    mtp_page_size = 0
    bucketed: list[Mapping[int, tuple[str, ...]]] = []
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
        bucketed.append(MappingProxyType({page_size: tuple(names) for page_size, names in b.items()}))

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_max_layers_per_page_size()); for SWA
    # sub-groups this equals the sub-group size (each has a single
    # page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    return DeepseekV4CacheLayout(
        page_sizes=tuple(sorted_page_sizes),
        bucketed=tuple(bucketed),
        mtp_layer_names=tuple(mtp_layer_names),
        mtp_page_size=mtp_page_size,
        num_layer_tuples=num_layer_tuples,
    )


def is_deepseek_v4_groups(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
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


def deepseek_v4_pool_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Bytes one global block ID consumes across the shared tuple slots.

    vLLM re-plans the ranks with more KV memory against this value. The
    standardized per-group layout has a different divisor from Ascend's
    shared-tuple layout, so deriving it from the tuple geometry keeps every
    rank's ``num_blocks`` consistent.
    """
    return get_deepseek_v4_cache_layout(kv_cache_groups).pool_bytes_per_block
