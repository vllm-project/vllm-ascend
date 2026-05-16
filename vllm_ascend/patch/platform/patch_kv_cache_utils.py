# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    Two Ascend-specific behaviors:

    1. **Context parallelism with multiple KV cache groups**:
       vLLM PR #40860 added a restriction that hybrid KV cache groups with
       multiple block sizes do not support context parallelism (dcp/pcp > 1).
       This restriction is correct for CUDA but not for Ascend, which
       implements context parallelism for MLA and SWA-MLA layers independently.
       For multiple KV cache groups with CP, compute scheduler_block_size as
       lcm(group_block_sizes) * dcp * pcp to maintain alignment.

    2. **KV consumer partial-group caching**:
       When running as a KV consumer in P/D disaggregated inference with
       hybrid Mamba models, Mamba states are not transferred. The Mamba
       back-off (falling back to scheduler_block_size when Mamba block_size
       diverges) must be skipped so that hash_block_size remains at GCD
       granularity, allowing FullAttention-only prefix caching.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp * pcp
        return bs, bs

    if dcp != 1 or pcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by the CP factors for proper alignment.
        group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp * pcp
        return scheduler_block_size, scheduler_block_size

    # --- KV consumer partial-group caching ---
    # When the instance is a KV consumer with prefix caching enabled,
    # skip the Mamba back-off so that hash_block_size stays fine-grained.
    connector_enabled = vllm_config.kv_transfer_config is not None
    enable_kv_consumer_partial_group_caching = (
        connector_enabled and vllm_config.kv_transfer_config.is_kv_consumer and cache_config.enable_prefix_caching
    )

    if enable_kv_consumer_partial_group_caching:
        group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
        scheduler_block_size = math.lcm(*group_block_sizes)

        requested = cache_config.hash_block_size
        hash_block_size = requested if requested is not None else math.gcd(*group_block_sizes)
        if any(bs % hash_block_size != 0 for bs in group_block_sizes):
            raise ValueError(
                f"Invalid hash_block_size={hash_block_size}; all KV cache group "
                f"block sizes must be divisible by hash_block_size. "
                f"Got group block sizes={group_block_sizes}."
            )
        return scheduler_block_size, hash_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
