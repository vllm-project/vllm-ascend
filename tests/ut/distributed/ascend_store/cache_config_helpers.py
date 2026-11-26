# SPDX-License-Identifier: Apache-2.0
"""Build connector test inputs with Core-resolved cache metadata."""

from copy import copy
from types import SimpleNamespace

import torch
from vllm.v1.core.kv_cache_utils import resolve_cache_hit_alignment_tokens, resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs


def make_resolved_cache_config(config, cache=None):
    cache = copy(cache)
    if cache is None:
        num_layers = config.model_config.get_num_layers(config.parallel_config)
        cache = SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(
                    [f"model.layers.{index}.attn" for index in range(num_layers)],
                    FullAttentionSpec(
                        block_size=config.cache_config.block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
            num_blocks=4,
            hash_block_size=None,
            cache_hit_alignment_tokens=None,
            prefix_cache_retention_interval=config.cache_config.prefix_cache_retention_interval,
        )
    if isinstance(getattr(cache, "hash_block_size", None), int) and isinstance(
        getattr(cache, "cache_hit_alignment_tokens", None), int
    ):
        return cache
    # EngineCore resolves the scheduler representation, then publishes the same
    # metadata to worker plans (whose UniformType containers remain intact).
    scheduler_cache = copy(cache)
    scheduler_cache.kv_cache_groups = [copy(group) for group in cache.kv_cache_groups]
    for group in scheduler_cache.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            group.kv_cache_spec = next(iter(group.kv_cache_spec.kv_cache_specs.values()))
    register_ascend_kv_cache_specs()
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(scheduler_cache, config)
    if not isinstance(getattr(cache, "hash_block_size", None), int):
        cache.hash_block_size = hash_block_size
    if not isinstance(getattr(cache, "cache_hit_alignment_tokens", None), int):
        cache.cache_hit_alignment_tokens = resolve_cache_hit_alignment_tokens(
            scheduler_cache, config, scheduler_block_size, cache.hash_block_size
        )
    if "prefix_cache_retention_interval" not in vars(cache):
        cache.prefix_cache_retention_interval = config.cache_config.prefix_cache_retention_interval
    return cache
