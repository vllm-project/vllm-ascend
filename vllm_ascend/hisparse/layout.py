# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Plan mixed sparse attention and recurrent state with native Ascend indexer pages."""

from collections import defaultdict
from collections.abc import Collection

from vllm.config import VllmConfig
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
    return (
        any(_is_native_indexer(spec) for spec in specs)
        and any(isinstance(spec, MambaSpec) for spec in specs)
        and any(_is_sparse_mla(spec) for spec in specs)
    )


def has_hybrid_sfa_hisparse_groups(groups: list[KVCacheGroupSpec]) -> bool:
    return any(isinstance(group.kv_cache_spec, MambaSpec) for group in groups) and any(
        group.role is KVCacheGroupRole.HISPARSE_INDEXER
        and all(_is_native_indexer(spec) for spec in iter_layer_specs(group.kv_cache_spec))
        for group in groups
    )


def get_hybrid_sfa_hisparse_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec] | None:
    """Keep native indexer geometry and size recurrent groups against the device pool."""
    attention_config = getattr(vllm_config, "attention_config", None)
    if getattr(attention_config, "hisparse_config", None) is None or not has_hybrid_sfa_hisparse_specs(
        kv_cache_spec.values()
    ):
        return None
    # TODO: Support DSpark speculative decoding with HiSparse.
    if any(spec.is_draft for spec in kv_cache_spec.values()):
        raise ValueError("Ascend HiSparse with native SFA indexer and recurrent states does not support draft caches.")
    if (speculative_config := vllm_config.speculative_config) is not None and speculative_config.method not in (
        "ngram",
        "ngram_gpu",
    ):
        raise ValueError(
            "Ascend HiSparse with native SFA indexer and recurrent states only supports "
            f"ngram and ngram_gpu speculative decoding, got {speculative_config.method!r}."
        )

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

    # The upstream layout uses regular groups to size the indexer manager block.
    # Keep them out of this call: Ascend's component-major indexer binds one
    # native kernel page per manager block and uses its own slot mapping.
    sparse_groups = get_hisparse_kv_cache_groups(vllm_config, sparse_specs)
    assert sparse_groups is not None
    device_groups = [group for group in sparse_groups if not group.host_resident]
    base_bytes = max(
        group.kv_cache_spec.page_size_bytes
        * (1 if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) else len(group.layer_names))
        for group in device_groups
    )

    def make_groups(row_bytes: int) -> list[KVCacheGroupSpec]:
        recurrent_layers: list[list[str]] = []
        for spec, names in names_by_spec.items():
            group_size = max(1, row_bytes // spec.page_size_bytes)
            count = cdiv(len(names), group_size)
            recurrent_layers.extend(names[index::count] for index in range(count))
        return sparse_groups + kv_cache_planning.create_kv_cache_group_specs(recurrent_specs, recurrent_layers)

    if vllm_config.cache_config.kv_cache_layout is None:
        return make_groups(base_bytes)

    # As in upstream #67262, consider whole recurrent pages and score them with
    # Core's device memory estimate. Only the recurrent grouping varies here.
    candidate_bytes = {base_bytes, kv_cache_planning._pool_bytes_per_block(vllm_config, sparse_groups)}
    candidate_bytes.update(
        spec.page_size_bytes * count
        for spec, names in names_by_spec.items()
        for count in range(1, len(names) + 1)
        if spec.page_size_bytes * count > base_bytes
    )
    builder = kv_cache_planning.DefaultKVCacheConfigBuilder()
    return min(
        (make_groups(row_bytes) for row_bytes in sorted(candidate_bytes)),
        key=lambda groups: kv_cache_planning._max_memory_usage_bytes_from_groups(builder, vllm_config, groups),
    )
