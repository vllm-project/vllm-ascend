# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Assemble native and pooled layouts over the Core HiSparse planner."""

from collections import defaultdict
from collections.abc import Collection

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import cdiv
from vllm.v1.core import kv_cache_planning
from vllm.v1.hisparse.layout import get_hisparse_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
    iter_layer_specs,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec


def _is_native_indexer(spec: KVCacheSpec) -> bool:
    return isinstance(spec, AscendSFAIndexerCacheSpec) and spec.cache_role is SparseCacheRole.INDEXER


def _is_sparse_mla(spec: KVCacheSpec) -> bool:
    return isinstance(spec, MLAAttentionSpec) and spec.cache_role is SparseCacheRole.SPARSE


def has_hybrid_sfa_hisparse_specs(specs: Collection[KVCacheSpec]) -> bool:
    """Recognize native sparse attention combined with recurrent state specs."""
    return (
        any(_is_native_indexer(spec) for spec in specs)
        and any(isinstance(spec, MambaSpec) for spec in specs)
        and any(_is_sparse_mla(spec) for spec in specs)
    )


def has_hybrid_sfa_hisparse_groups(groups: list[KVCacheGroupSpec]) -> bool:
    """Recognize recurrent state with a native HiSparse indexer in the device pool."""
    return any(isinstance(group.kv_cache_spec, MambaSpec) for group in groups) and any(
        group.role is KVCacheGroupRole.HISPARSE_INDEXER
        and all(_is_native_indexer(spec) for spec in iter_layer_specs(group.kv_cache_spec))
        for group in groups
    )


def get_hybrid_sfa_hisparse_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec] | None:
    """Give native indexers one manager block per kernel page beside recurrent groups."""
    attention_config = getattr(vllm_config, "attention_config", None)
    if getattr(attention_config, "hisparse_config", None) is None:
        return None

    specs = kv_cache_spec.values()
    if not has_hybrid_sfa_hisparse_specs(specs):
        return None

    if vllm_config.speculative_config is not None or any(
        spec.is_draft or (isinstance(spec, MambaSpec) and spec.num_speculative_blocks) for spec in specs
    ):
        raise ValueError("Ascend hybrid SFA HiSparse does not support speculative decoding or draft caches.")

    sparse_specs: dict[str, KVCacheSpec] = {}
    recurrent_specs: dict[str, KVCacheSpec] = {}
    names_by_spec: dict[MambaSpec, list[str]] = defaultdict(list)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, MambaSpec):
            recurrent_specs[name] = spec
            names_by_spec[spec].append(name)
        elif _is_native_indexer(spec) or _is_sparse_mla(spec):
            sparse_specs[name] = spec
        elif not isinstance(spec, (HiSparseResidentSpec, HiSparseHotSpec)):
            raise ValueError(f"Unsupported cache spec in Ascend hybrid SFA HiSparse: {name} ({type(spec).__name__}).")

    # Core uses regular groups to enlarge indexer manager blocks. Lay out
    # attention separately so the Ascend indexer keeps its native slot mapping.
    sparse_groups = get_hisparse_kv_cache_groups(vllm_config, sparse_specs)
    assert sparse_groups is not None
    device_groups = [group for group in sparse_groups if not group.host_resident]
    logical_row_bytes = max(
        group.kv_cache_spec.page_size_bytes
        * (1 if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) else len(group.layer_names))
        for group in device_groups
    )

    def make_groups(row_bytes: int) -> list[KVCacheGroupSpec]:
        recurrent_layers: list[list[str]] = []
        for spec, names in names_by_spec.items():
            layers_per_group = max(1, row_bytes // spec.page_size_bytes)
            group_count = cdiv(len(names), layers_per_group)
            recurrent_layers.extend(names[index::group_count] for index in range(group_count))
        return sparse_groups + kv_cache_planning.create_kv_cache_group_specs(recurrent_specs, recurrent_layers)

    # Once the backend layout is known, the aligned stride may fit more
    # state layers per block.
    if vllm_config.cache_config.kv_cache_layout is None:
        return make_groups(logical_row_bytes)

    candidates = {logical_row_bytes, kv_cache_planning._pool_bytes_per_block(vllm_config, device_groups)}
    candidates.update(
        spec.page_size_bytes * count
        for spec, names in names_by_spec.items()
        for count in range(1, len(names) + 1)
        if spec.page_size_bytes * count > logical_row_bytes
    )
    builder = kv_cache_planning.DefaultKVCacheConfigBuilder()
    return min(
        (make_groups(row_bytes) for row_bytes in sorted(candidates)),
        key=lambda groups: kv_cache_planning._max_memory_usage_bytes_from_groups(builder, vllm_config, groups),
    )


def group_interleaved_mamba_layers(
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
                f"Interleaved Mamba grouping requires layer names with numeric indices, got {name!r}."
            ) from exc

        is_mamba = name in mamba_specs
        previous_kind = layer_is_mamba.setdefault(layer_idx, is_mamba)
        if previous_kind != is_mamba:
            raise ValueError(
                f"An interleaved layer cannot contain both Mamba and MLA cache specs: layer index {layer_idx}."
            )
        if is_mamba:
            if layer_idx in mamba_name_by_index:
                raise ValueError(
                    f"An interleaved layer must own exactly one Mamba cache spec: layer index {layer_idx}."
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
        raise ValueError("Mamba specs were provided but no Mamba layers were found.")

    sorted_mamba_names = [mamba_name_by_index[index] for index in sorted(mamba_name_by_index)]
    return [sorted_mamba_names[offset::max_run_length] for offset in range(max_run_length)]


def regroup_pooled_hisparse_kda(
    kv_cache_spec: dict[str, KVCacheSpec], groups: list[KVCacheGroupSpec]
) -> list[KVCacheGroupSpec]:
    """Split pooled HiSparse KDA groups to bound the physical cache pitch."""
    mamba_specs = {name: spec for name, spec in kv_cache_spec.items() if isinstance(spec, MambaSpec)}
    if not mamba_specs:
        return groups
    # Cap state layers per physical block so C8 retains enough global IDs
    # after charging for both the device slab and the compact KPool region.
    max_layers_per_group = 6
    native_groups = group_interleaved_mamba_layers(kv_cache_spec, mamba_specs)
    split_groups = [
        names[start : start + max_layers_per_group]
        for names in native_groups
        for start in range(0, len(names), max_layers_per_group)
    ]
    kda_groups = kv_cache_planning.create_kv_cache_group_specs(mamba_specs, split_groups)
    # Other groups retain their block tables and logical page sizes. Core
    # materializes the finished group list with the new aligned slab stride.
    other_groups = [group for group in groups if all(name not in mamba_specs for name in group.layer_names)]
    logger.info_once(
        "HiSparse pooled KDA groups: %d -> %d, layers per group: %s",
        len(groups) - len(other_groups),
        len(kda_groups),
        tuple(len(group.layer_names) for group in kda_groups),
    )
    return [*other_groups, *kda_groups]
