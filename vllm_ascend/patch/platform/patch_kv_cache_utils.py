# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend block-size resolution for hybrid KV cache groups.

Model-specific KV cache planning (grouping, materialization, memory accounting)
lives in ``vllm_ascend.core.kv_cache_config_builder``, which the platform
selects through ``NPUPlatform.get_kv_cache_config_builder_cls``. Only the
block-size resolution below is still replaced in place, because engine startup
imports it directly.
"""

import math
from dataclasses import replace

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable
from vllm_ascend.models.deepseek_v41.cache_config import is_deepseek_v41_cache

_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support DCP.
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with DCP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp to maintain alignment.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups
    cacheable_groups = [group for group in groups if is_prefix_cacheable(group.kv_cache_spec)]
    filtered_private_groups = bool(cacheable_groups) and len(cacheable_groups) != len(groups)
    if filtered_private_groups and not is_deepseek_v41_cache(groups):
        # A fixed tail block is not a token-page scheduling or hashing unit.
        # Pool alignment is enforced by the GLM planner and prefix coordinator.
        kv_cache_config = replace(kv_cache_config, kv_cache_groups=cacheable_groups)
        groups = cacheable_groups

    cacheable_groups = [g for g in groups if is_prefix_cacheable(g.kv_cache_spec)]
    if len(cacheable_groups) != len(groups):
        scheduler_block_size = math.lcm(*(g.kv_cache_spec.block_size for g in groups)) * dcp
        if not cache_config.enable_prefix_caching or not cacheable_groups:
            return scheduler_block_size, scheduler_block_size
        filtered = replace(kv_cache_config, kv_cache_groups=cacheable_groups)
        if dcp == 1:
            _, hash_block_size = _orig_resolve_kv_cache_block_sizes(filtered, vllm_config)
        else:
            hash_block_size = math.gcd(*(g.kv_cache_spec.block_size for g in cacheable_groups))
        return scheduler_block_size, hash_block_size

    if len(groups) <= 1:
        # EngineCore's global size is the minimum across all original groups.
        # After filtering a private tail, it can therefore be smaller than the
        # sole cacheable group's token-page size.
        block_size = groups[0].kv_cache_spec.block_size if filtered_private_groups else cache_config.block_size
        bs = block_size * dcp
        return bs, bs

    group_block_sizes = [group.kv_cache_spec.block_size for group in groups]
    if dcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by DCP for proper alignment.
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp
        if not cache_config.enable_prefix_caching:
            return scheduler_block_size, scheduler_block_size
        hash_block_size = math.gcd(*group_block_sizes)
        return scheduler_block_size, hash_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
