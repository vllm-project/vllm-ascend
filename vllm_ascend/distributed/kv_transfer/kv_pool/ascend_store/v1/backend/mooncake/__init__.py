"""Mooncake adapter; native registration and session mechanics stay in this package."""

from __future__ import annotations

import hashlib
import json
from copy import copy
from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from ..naming import WireKeyBuilder
from .backend import MooncakeBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from ...infrastructure.configuration import KVPoolConfig


def create_key_builder(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, config: KVPoolConfig) -> WireKeyBuilder:
    """Compile stable key parts once; runtime requests never reread configuration."""
    use_hybrid = config.use_hybrid or len(config.group_specs) > 1
    names, page_sizes = [], []
    if use_hybrid or config.pp_size > 1:
        for group in kv_cache_config.kv_cache_groups:
            names.append(tuple(sorted(group.layer_names)))
            spec = group.kv_cache_spec
            specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
            page_sizes.append(tuple(sorted({item.block_size for item in specs})))
    namespace = ""
    if config.pp_size > 1:
        empty = [index for index, names in enumerate(names) if not names]
        if empty:
            raise ValueError(
                "Mooncake layerwise PP requires each stage to own layers in every KV cache group; "
                f"empty groups: {empty}. Use a partition covering all groups, or disable layerwise pooling. "
                "Draft-only groups are not supported."
            )
        model = vllm_config.model_config
        layer_ranges = []
        for stage in range(config.pp_size):
            stage_config = copy(vllm_config.parallel_config)
            stage_config.rank = stage * config.tp_size
            layer_ranges.append(model.get_layers_start_end_indices(stage_config))
        previous_end = 0
        for start, end in layer_ranges:
            if start != previous_end or end <= start:
                raise ValueError("Mooncake PP requires contiguous, nonempty model layer partitions")
            previous_end = end
        if len(layer_ranges) != config.pp_size or previous_end != model.get_total_num_hidden_layers():
            raise ValueError("Mooncake PP partitions must cover every model layer")
        # Scheduler/Worker block sizes may differ; identity uses group page sizes.
        cache = copy(vllm_config.cache_config)
        if page_sizes:
            cache.block_size = min(size for sizes in page_sizes for size in sizes)
        speculative = vllm_config.speculative_config
        identity = (
            config.tp_size,
            layer_ranges,
            model.compute_hash(),
            cache.compute_hash(),
            cache.cache_dtype,
            cache.block_size,
            speculative.compute_hash() if speculative is not None else None,
        )
        namespace = "mooncake_pp_v3:" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if use_hybrid:
        groups = [(() if namespace else names, sizes) for names, sizes in zip(names, page_sizes, strict=True)]
        layout_identity = (namespace, config.tp_size, groups) if namespace else (config.tp_size, groups)
        layout = hashlib.sha256(json.dumps(layout_identity, sort_keys=True).encode()).hexdigest()
        prefixes = tuple(
            tuple(
                _hybrid_key_prefix(
                    config.model_name, layout, group, spec.block_size, stage if config.pp_size > 1 else None
                )
                for stage in range(config.pp_size)
            )
            for group, spec in enumerate(config.group_specs)
        )
    else:
        prefixes = (tuple(_block_key_prefix(config.model_name, namespace, stage) for stage in range(config.pp_size)),)
    return WireKeyBuilder(prefixes, tuple(f"@{head}" for head in range(config.tp_size)))


def _block_key_prefix(model_name: str, namespace: str, pp_rank: int) -> str:
    if namespace:
        return f"{model_name}@{namespace}@pp_rank:{pp_rank}@"
    if pp_rank:
        raise ValueError("Mooncake PP keys require a layout namespace")
    return f"{model_name}@"


def _hybrid_key_prefix(model: str, layout: str, group: int, block_size: int, pp_rank: int | None) -> str:
    stage = "" if pp_rank is None else f"@pp_rank:{pp_rank}"
    return f"{model}@mooncake_hybrid_v1:{layout}{stage}@group:{group}@block:{block_size}@"


__all__ = ["MooncakeBackend", "create_key_builder"]
