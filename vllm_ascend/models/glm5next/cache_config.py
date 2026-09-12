# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-Next cache groups and source-compatible physical pool layout.

The main MLA cache and compressed indexer cache share scheduler block IDs,
while compressor state and every KDA/Mamba group allocate IDs independently.
Physical storage uses standard unpacked KV cache descriptors with two page-size
classes: main MLA/KDA pages and compressed-indexer/state pages.
"""

from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.kv_cache_utils import (
    create_kv_cache_group_specs,
    may_override_num_blocks,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import get_kv_cache_compression_ratio


@dataclass(frozen=True)
class _Glm5NextCacheLayout:
    full_group: KVCacheGroupSpec
    state_group: KVCacheGroupSpec
    mamba_groups: tuple[KVCacheGroupSpec, ...]
    mla_names: tuple[str, ...]
    indexer_names: tuple[str, ...]
    state_names: tuple[str, ...]
    main_page_size: int
    small_page_size: int
    main_slot_count: int
    small_slot_count: int


def _is_glm5_next_spec(spec: KVCacheSpec) -> bool:
    return getattr(spec, "model_version", None) == "glm5_next"


def _unpadded_page_size(spec: KVCacheSpec) -> int:
    if hasattr(spec, "unpadded_page_size_bytes"):
        return spec.unpadded_page_size_bytes
    if hasattr(spec, "real_page_size_bytes"):
        return spec.real_page_size_bytes
    return spec.page_size_bytes


def _sorted_layer_names(layer_names: list[str]) -> tuple[str, ...]:
    try:
        return tuple(sorted(layer_names, key=extract_layer_index))
    except ValueError:
        # Synthetic layer names used by callers or tests need not contain an
        # integer model-layer index. Preserve their registration order.
        return tuple(layer_names)


def _layer_indices(layer_names: tuple[str, ...]) -> tuple[int, ...] | None:
    try:
        return tuple(extract_layer_index(name) for name in layer_names)
    except ValueError:
        return None


def _is_glm5_next_main_spec(spec: KVCacheSpec) -> bool:
    return isinstance(spec, MLAAttentionSpec) and _is_glm5_next_spec(spec) and get_kv_cache_compression_ratio(spec) == 1


def _is_glm5_next_indexer_spec(spec: KVCacheSpec) -> bool:
    return isinstance(spec, MLAAttentionSpec) and _is_glm5_next_spec(spec) and get_kv_cache_compression_ratio(spec) > 1


def _is_glm5_next_state_spec(spec: KVCacheSpec) -> bool:
    return (
        isinstance(spec, SlidingWindowMLASpec)
        and _is_glm5_next_spec(spec)
        and getattr(spec, "cache_role", None) == "indexer_state"
    )


def _align_glm5_next_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]) -> None:
    """Align GLM-Next specs into two physical page-size classes in-place."""

    main_specs = [spec for spec in kv_cache_spec.values() if _is_glm5_next_main_spec(spec)]
    indexer_specs = [spec for spec in kv_cache_spec.values() if _is_glm5_next_indexer_spec(spec)]
    state_specs = [spec for spec in kv_cache_spec.values() if _is_glm5_next_state_spec(spec)]
    mamba_specs = [spec for spec in kv_cache_spec.values() if isinstance(spec, MambaSpec)]

    if not main_specs and not indexer_specs and not state_specs:
        return
    if not main_specs or not indexer_specs or not state_specs:
        raise ValueError("GLM-Next cache layout requires main MLA, compressed indexer, and compressor-state specs.")

    main_candidates = (*main_specs, *mamba_specs)
    main_page_size = max(
        max(spec.page_size_bytes for spec in main_candidates),
        max(_unpadded_page_size(spec) for spec in main_candidates),
    )
    small_candidates = (*indexer_specs, *state_specs)
    small_page_size = max(
        max(spec.page_size_bytes for spec in small_candidates),
        max(_unpadded_page_size(spec) for spec in small_candidates),
    )

    for spec in main_candidates:
        object.__setattr__(spec, "page_size_padded", main_page_size)
    for spec in small_candidates:
        object.__setattr__(spec, "page_size_padded", small_page_size)


def _create_glm5_next_attention_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Create the shared full-history group and separate state group."""

    main_names = _sorted_layer_names([name for name, spec in kv_cache_spec.items() if _is_glm5_next_main_spec(spec)])
    indexer_names = _sorted_layer_names(
        [name for name, spec in kv_cache_spec.items() if _is_glm5_next_indexer_spec(spec)]
    )
    state_names = _sorted_layer_names([name for name, spec in kv_cache_spec.items() if _is_glm5_next_state_spec(spec)])

    classified_names = {*main_names, *indexer_names, *state_names}
    if classified_names != set(kv_cache_spec):
        raise ValueError("GLM-Next KV cache specs contain an unsupported cache role.")
    if not (len(main_names) == len(indexer_names) == len(state_names) > 0):
        raise ValueError(
            "Every GLM-Next MLA layer requires one main MLA, compressed indexer, and compressor-state cache."
        )

    main_indices = _layer_indices(main_names)
    if main_indices is not None and (
        main_indices != _layer_indices(indexer_names) or main_indices != _layer_indices(state_names)
    ):
        raise ValueError("GLM-Next MLA, indexer, and state cache layer indices do not match.")

    full_block_sizes = {kv_cache_spec[name].block_size for name in (*main_names, *indexer_names)}
    if len(full_block_sizes) != 1:
        raise ValueError("GLM-Next main MLA and compressed indexer caches must use one logical block size.")

    for main_name, indexer_name, state_name in zip(main_names, indexer_names, state_names):
        main_spec = kv_cache_spec[main_name]
        indexer_spec = kv_cache_spec[indexer_name]
        state_spec = kv_cache_spec[state_name]
        assert isinstance(main_spec, MLAAttentionSpec)
        assert isinstance(indexer_spec, MLAAttentionSpec)
        assert isinstance(state_spec, SlidingWindowMLASpec)

        compress_ratio = get_kv_cache_compression_ratio(indexer_spec)
        if main_spec.block_size % compress_ratio:
            raise ValueError(
                "GLM-Next logical block size must be divisible by the indexer "
                f"compression ratio: block_size={main_spec.block_size}, "
                f"compress_ratio={compress_ratio}."
            )
        if state_spec.block_size != compress_ratio or state_spec.sliding_window != compress_ratio:
            raise ValueError(
                f"GLM-Next indexer state block/window size must equal the paired compression ratio {compress_ratio}."
            )

    # Main and indexer caches deliberately share scheduler block IDs. Keep a
    # main spec first because the scheduler unwraps the first nested spec and
    # must select FullAttentionManager for this combined group.
    full_names = [
        name for main_name, indexer_name in zip(main_names, indexer_names) for name in (main_name, indexer_name)
    ]
    full_specs = {name: kv_cache_spec[name] for name in full_names}
    full_uniform_spec = UniformTypeKVCacheSpecs.from_specs(full_specs)
    if full_uniform_spec is None:
        raise ValueError(
            "GLM-Next main MLA and compressed indexer caches must have uniform full-attention block-table semantics."
        )

    state_specs = {name: kv_cache_spec[name] for name in state_names}
    state_uniform_spec = UniformTypeKVCacheSpecs.from_specs(state_specs)
    if state_uniform_spec is None:
        raise ValueError("GLM-Next compressor-state caches must have uniform sliding-window block-table semantics.")

    return [
        KVCacheGroupSpec(full_names, full_uniform_spec),
        KVCacheGroupSpec(list(state_names), state_uniform_spec),
    ]


def _get_glm5_next_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> _Glm5NextCacheLayout | None:
    """Recognize validated GLM-Next groups and derive physical slot counts."""

    if not kv_cache_groups:
        return None

    full_groups: list[KVCacheGroupSpec] = []
    state_groups: list[KVCacheGroupSpec] = []
    mamba_groups: list[KVCacheGroupSpec] = []
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, MambaSpec):
            mamba_groups.append(group)
            continue
        if not isinstance(group_spec, UniformTypeKVCacheSpecs):
            continue

        values = list(group_spec.kv_cache_specs.values())
        if (
            values
            and all(isinstance(spec, MLAAttentionSpec) and _is_glm5_next_spec(spec) for spec in values)
            and any(_is_glm5_next_main_spec(spec) for spec in values)
            and any(_is_glm5_next_indexer_spec(spec) for spec in values)
        ):
            full_groups.append(group)
        elif values and all(_is_glm5_next_state_spec(spec) for spec in values):
            state_groups.append(group)

    has_glm5_next_group = bool(full_groups or state_groups)
    if not has_glm5_next_group:
        return None
    if len(full_groups) != 1 or len(state_groups) != 1:
        raise ValueError(
            "GLM-Next requires exactly one combined main/indexer group and one compressor-state KV cache group."
        )
    if len(full_groups) + len(state_groups) + len(mamba_groups) != len(kv_cache_groups):
        raise ValueError("GLM-Next KV cache groups contain an unsupported cache spec.")

    full_group = full_groups[0]
    state_group = state_groups[0]
    assert isinstance(full_group.kv_cache_spec, UniformTypeKVCacheSpecs)
    assert isinstance(state_group.kv_cache_spec, UniformTypeKVCacheSpecs)
    full_specs = full_group.kv_cache_spec.kv_cache_specs
    state_specs = state_group.kv_cache_spec.kv_cache_specs
    mla_names = _sorted_layer_names(
        [name for name in full_group.layer_names if _is_glm5_next_main_spec(full_specs[name])]
    )
    indexer_names = _sorted_layer_names(
        [name for name in full_group.layer_names if _is_glm5_next_indexer_spec(full_specs[name])]
    )
    state_names = _sorted_layer_names(state_group.layer_names)
    if not (len(mla_names) == len(indexer_names) == len(state_names)):
        raise ValueError("Every GLM-Next MLA layer must own one compressed indexer and one compressor-state cache.")
    mla_indices = _layer_indices(mla_names)
    if mla_indices is not None and (
        mla_indices != _layer_indices(indexer_names) or mla_indices != _layer_indices(state_names)
    ):
        raise ValueError("GLM-Next MLA, indexer, and state cache layer indices do not match.")

    # Pipeline-parallel projection keeps empty groups with their global spec.
    # Derive the two canonical page classes from the retained specs, while the
    # slot counts below use only this worker's projected layer names.
    main_page_sizes = {spec.page_size_bytes for spec in full_specs.values() if _is_glm5_next_main_spec(spec)} | {
        group.kv_cache_spec.page_size_bytes for group in mamba_groups
    }
    small_page_sizes = {spec.page_size_bytes for spec in full_specs.values() if _is_glm5_next_indexer_spec(spec)} | {
        spec.page_size_bytes for spec in state_specs.values()
    }
    if len(main_page_sizes) != 1 or len(small_page_sizes) != 1:
        raise ValueError("GLM-Next cache specs were not aligned to two physical page sizes.")

    main_slot_count = max(
        (
            len(mla_names),
            *(len(group.layer_names) for group in mamba_groups),
        )
    )
    return _Glm5NextCacheLayout(
        full_group=full_group,
        state_group=state_group,
        mamba_groups=tuple(mamba_groups),
        mla_names=mla_names,
        indexer_names=indexer_names,
        state_names=state_names,
        main_page_size=next(iter(main_page_sizes)),
        small_page_size=next(iter(small_page_sizes)),
        main_slot_count=main_slot_count,
        small_slot_count=len(indexer_names),
    )


def _group_glm5_next_mamba_layer_names(
    kv_cache_spec: dict[str, KVCacheSpec],
    mamba_specs: dict[str, MambaSpec],
) -> list[list[str]]:
    """Recover recurrent runs without depending on spec insertion order."""

    layer_is_mamba: dict[int, bool] = {}
    mamba_name_by_index: dict[int, str] = {}
    for name in kv_cache_spec:
        try:
            layer_idx = extract_layer_index(name)
        except ValueError as exc:
            raise ValueError(
                f"GLM-Next Mamba grouping requires layer names with numeric indices, got {name!r}."
            ) from exc

        is_mamba = name in mamba_specs
        previous_kind = layer_is_mamba.setdefault(layer_idx, is_mamba)
        if previous_kind != is_mamba:
            raise ValueError(
                f"A GLM-Next model layer cannot contain both Mamba and MLA cache specs: layer index {layer_idx}."
            )
        if is_mamba:
            if layer_idx in mamba_name_by_index:
                raise ValueError(
                    f"A GLM-Next model layer must own exactly one Mamba cache spec: layer index {layer_idx}."
                )
            mamba_name_by_index[layer_idx] = name

    max_run_length = 0
    run_length = 0
    previous_layer_idx: int | None = None
    for layer_idx in sorted(layer_is_mamba):
        is_consecutive = previous_layer_idx is not None and layer_idx == previous_layer_idx + 1
        if layer_is_mamba[layer_idx]:
            run_length = run_length + 1 if is_consecutive else 1
            max_run_length = max(max_run_length, run_length)
        else:
            run_length = 0
        previous_layer_idx = layer_idx

    if max_run_length == 0:
        raise ValueError("GLM-Next Mamba specs were provided but no Mamba layers were found.")

    sorted_mamba_names = [mamba_name_by_index[index] for index in sorted(mamba_name_by_index)]
    return [sorted_mamba_names[offset::max_run_length] for offset in range(max_run_length)]


def _create_mamba_groups(
    mamba_specs: dict[str, MambaSpec],
    grouped_layer_names: list[list[str]],
) -> list[KVCacheGroupSpec]:
    sorted_groups = [list(_sorted_layer_names(layer_names)) for layer_names in grouped_layer_names]
    return create_kv_cache_group_specs(mamba_specs, sorted_groups)


def get_glm5_next_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Build GLM-Next scheduler groups and align their physical page sizes."""

    if not any(_is_glm5_next_spec(spec) for spec in kv_cache_spec.values()):
        raise ValueError("Expected GLM-Next cache specs.")

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if getattr(scheduler_config, "disable_hybrid_kv_cache_manager", False):
        raise ValueError("GLM-Next's paired MLA/indexer and sliding state layout requires the hybrid KV cache manager.")

    _align_glm5_next_cache_specs(kv_cache_spec)
    mamba_specs = {name: spec for name, spec in kv_cache_spec.items() if isinstance(spec, MambaSpec)}
    attention_specs = {name: spec for name, spec in kv_cache_spec.items() if not isinstance(spec, MambaSpec)}
    groups = _create_glm5_next_attention_groups(attention_specs)
    if not mamba_specs:
        # The standalone MTP runner has the same attention/state pairing but
        # no recurrent groups.
        return groups

    grouped_names = _group_glm5_next_mamba_layer_names(kv_cache_spec, mamba_specs)
    groups.extend(_create_mamba_groups(mamba_specs, grouped_names))
    return groups


def get_glm5_next_pool_bytes_per_block(groups: list[KVCacheGroupSpec]) -> int:
    """Return physical bytes represented by one global block ID."""

    layout = _get_glm5_next_cache_layout(groups)
    if layout is None:
        raise ValueError("Expected GLM-Next cache groups.")
    return layout.main_slot_count * layout.main_page_size + layout.small_slot_count * layout.small_page_size


def get_glm5_next_kv_cache_config(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Describe one standard unpacked tensor per physical cache slot."""

    layout = _get_glm5_next_cache_layout(groups)
    if layout is None:
        raise ValueError("Expected GLM-Next cache groups.")

    bytes_per_block = get_glm5_next_pool_bytes_per_block(groups)
    num_blocks = may_override_num_blocks(vllm_config, max(available_memory // bytes_per_block, 0))
    tensors: list[KVCacheTensor] = []

    def make_tensor(size: int, layer_names: list[str], page_size: int) -> KVCacheTensor:
        return KVCacheTensor(
            size=size,
            layers=layer_names,
            offset=0,
            layer_stride=0,
            block_stride=page_size,
        )

    # Layers in independent scheduler groups can reuse the same physical slot
    # because their block IDs are allocated independently. A standard unpacked
    # descriptor lets the existing model-runner allocator create one backing
    # tensor per slot without a model-specific allocation path.
    for slot in range(layout.main_slot_count):
        shared_by: list[str] = []
        if slot < len(layout.mla_names):
            shared_by.append(layout.mla_names[slot])
        for group in layout.mamba_groups:
            if slot < len(group.layer_names):
                shared_by.append(group.layer_names[slot])
        tensors.append(
            make_tensor(
                layout.main_page_size * num_blocks,
                shared_by,
                layout.main_page_size,
            )
        )

    for indexer_name, state_name in zip(layout.indexer_names, layout.state_names):
        tensors.append(
            make_tensor(
                layout.small_page_size * num_blocks,
                [indexer_name, state_name],
                layout.small_page_size,
            )
        )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def get_glm5_next_max_memory_usage(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
) -> int:
    """Return capacity for GLM-Next's shared global block-id pool."""

    layout = _get_glm5_next_cache_layout(groups)
    if layout is None:
        raise ValueError("Expected GLM-Next cache groups.")
    # Scheduler groups allocate disjoint IDs from the shared global BlockPool.
    # One request therefore needs enough IDs for every group even though one
    # physical tensor slot can be reused by layers from different groups.
    blocks = sum(
        (group.kv_cache_spec.max_memory_usage_bytes(vllm_config) + group.kv_cache_spec.page_size_bytes - 1)
        // group.kv_cache_spec.page_size_bytes
        for group in groups
    )
    return blocks * get_glm5_next_pool_bytes_per_block(groups)
