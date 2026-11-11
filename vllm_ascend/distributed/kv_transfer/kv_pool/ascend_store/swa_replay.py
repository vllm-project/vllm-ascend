# SPDX-License-Identifier: Apache-2.0
"""Admission constraints shared by native Store and the CANN range protocol."""

from typing import Any


def store_replay_enabled(vllm_config: Any, kv_cache_config: Any, *, use_layerwise: bool) -> bool:
    if not getattr(vllm_config.cache_config, "swa_bounded_replay", False) or kv_cache_config is None:
        return False
    windows = [getattr(group.kv_cache_spec, "prefix_replay_tokens", 0) for group in kv_cache_config.kv_cache_groups]
    if not any(isinstance(window, int) and window > 0 for window in windows):
        return False
    parallel = vllm_config.parallel_config
    if use_layerwise or parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
        raise ValueError("AscendStore SWA replay currently requires request-level transfer with PCP=1 and DCP=1")
    if vllm_config.kv_transfer_config.kv_connector_extra_config.get("hybrid_cache_c128_chunk", False):
        raise ValueError("AscendStore SWA replay does not support hybrid_cache_c128_chunk")
    return True
