"""Snapshot configuration facts used by Backend wire naming, without encoding keys."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True, slots=True)
class KeyNamingConfig:
    model_name: str
    tp_size: int
    pp_size: int
    use_hybrid: bool
    group_block_sizes: tuple[int, ...]
    group_layer_names: tuple[tuple[str, ...], ...]
    group_page_sizes: tuple[tuple[int, ...], ...]
    layer_ranges: tuple[tuple[int, int], ...] = ()
    total_layers: int = 0
    model_hash: str = ""
    cache_hash: str = ""
    cache_dtype: str = ""
    cache_block_size: int = 0
    speculative_hash: str | None = None


def resolve_key_naming_config(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    *,
    model_name: str,
    use_hybrid: bool,
    group_block_sizes: tuple[int, ...],
    requires_layout_identity: bool,
) -> KeyNamingConfig:
    parallel = vllm_config.parallel_config
    names, page_sizes = [], []
    if requires_layout_identity and (use_hybrid or parallel.pipeline_parallel_size > 1):
        for group in kv_cache_config.kv_cache_groups:
            names.append(tuple(sorted(group.layer_names)))
            spec = group.kv_cache_spec
            specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
            page_sizes.append(tuple(sorted({int(item.block_size) for item in specs})))
    common = dict(
        model_name=model_name,
        tp_size=parallel.tensor_parallel_size,
        pp_size=parallel.pipeline_parallel_size,
        use_hybrid=use_hybrid,
        group_block_sizes=group_block_sizes,
        group_layer_names=tuple(names),
        group_page_sizes=tuple(page_sizes),
    )
    if not requires_layout_identity or parallel.pipeline_parallel_size == 1:
        return KeyNamingConfig(**common)
    model = vllm_config.model_config
    layer_ranges = []
    for stage in range(parallel.pipeline_parallel_size):
        stage_config = copy(parallel)
        stage_config.rank = stage * parallel.tensor_parallel_size
        layer_ranges.append(model.get_layers_start_end_indices(stage_config))
    # Scheduler and Worker can have different CacheConfig.block_size values;
    # the resolved group page sizes are shared, so normalize a copy for identity.
    cache = copy(vllm_config.cache_config)
    if page_sizes:
        cache.block_size = min(size for sizes in page_sizes for size in sizes)
    speculative = vllm_config.speculative_config
    return KeyNamingConfig(
        **common,
        layer_ranges=tuple(layer_ranges),
        total_layers=model.get_total_num_hidden_layers(),
        model_hash=model.compute_hash(),
        cache_hash=cache.compute_hash(),
        cache_dtype=cache.cache_dtype,
        cache_block_size=cache.block_size,
        speculative_hash=speculative.compute_hash() if speculative is not None else None,
    )
