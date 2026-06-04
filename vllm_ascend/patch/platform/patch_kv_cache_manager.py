# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

# Why this patch exists
# ─────────────────────
# In a PD-disaggregated setup, the Prefill (P) node transfers FullAttention KV
# blocks to the Decode (D) node via a KV connector.  Non-FullAttention state
# (e.g. Mamba recurrent state) is NOT transferred because the connector only
# handles attention KV.
#
# Upstream KVCacheManager.get_computed_blocks calls
# HybridKVCacheCoordinator.find_longest_cache_hit, which performs a
# min-reduction across ALL KV groups.  On the D node, non-FullAttention groups
# report 0 local cached blocks, so the min-reduction collapses the
# FullAttention hit length to 0 — discarding the transferred KV entirely.
#
# What this patch does
# ────────────────────
# For requests arriving with kv_transfer_params["do_remote_prefill"] = True,
# set hit_partial=True when calling find_longest_cache_hit.  That flag (added
# in patch_kv_cache_coordinator.py) makes the coordinator skip non-FullAttention
# groups so their 0-hit result never reduces the FullAttention hit length.
#
# Related upstream PR: https://github.com/vllm-project/vllm/pull/42524
# Remove this patch once PR #42524 is included in the supported vLLM version.

from vllm.v1.core.kv_cache_manager import KVCacheManager


def _get_computed_blocks(self, request):
    if not self.enable_caching or request.skip_reading_prefix_cache:
        return self.empty_kv_cache_blocks, 0

    max_cache_hit_length = request.num_tokens - 1
    hit_partial = bool(request.kv_transfer_params and request.kv_transfer_params.get("do_remote_prefill"))
    if hit_partial:
        computed_blocks, num_new_computed_tokens = self.coordinator.find_longest_cache_hit_partial_group(
            request.block_hashes, max_cache_hit_length
        )
    else:
        computed_blocks, num_new_computed_tokens = self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )

    if self.log_stats:
        assert self.prefix_cache_stats is not None
        self.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_new_computed_tokens,
            preempted=request.num_preemptions > 0,
        )
    return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens


KVCacheManager.get_computed_blocks = _get_computed_blocks
