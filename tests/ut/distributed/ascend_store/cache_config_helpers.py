# SPDX-License-Identifier: Apache-2.0
"""Build connector test inputs with Core-resolved cache metadata."""

from copy import copy
from types import SimpleNamespace

import torch
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec


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
    if not isinstance(getattr(cache, "hash_block_size", None), int):
        _, cache.hash_block_size = resolve_kv_cache_block_sizes(cache, config)
    if not isinstance(getattr(cache, "cache_hit_alignment_tokens", None), int):
        cache.cache_hit_alignment_tokens, _ = resolve_kv_cache_block_sizes(cache, config)
    if "prefix_cache_retention_interval" not in vars(cache):
        cache.prefix_cache_retention_interval = config.cache_config.prefix_cache_retention_interval
    return cache
