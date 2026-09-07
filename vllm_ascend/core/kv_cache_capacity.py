# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs


def replicated_draft_pool_bytes_per_block(groups: list[KVCacheGroupSpec]) -> int | None:
    """Size the target/Mamba shared tensors plus independent replicated drafts.

    The replicated-draft planner stores each target layer once and aliases
    the corresponding Mamba layers into it. Draft tensors have larger pages
    but do not add those pages to every target/Mamba layer tuple.
    """
    if len(groups) < 2 or not all(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in groups):
        return None
    first = groups[0].kv_cache_spec.kv_cache_specs
    drafts = [first[name] for name in groups[0].layer_names if getattr(first[name], "dcp_replication_size", 1) > 1]
    targets = [first[name] for name in groups[0].layer_names if getattr(first[name], "dcp_replication_size", 1) == 1]
    if not drafts or not targets or not all(isinstance(spec, FullAttentionSpec) for spec in (*targets, *drafts)):
        return None
    for group in groups[1:]:
        specs = group.kv_cache_spec.kv_cache_specs
        if len(group.layer_names) > len(targets):
            return None
        for name, target in zip(group.layer_names, targets):
            spec = specs[name]
            if not isinstance(spec, MambaSpec) or spec.page_size_bytes != target.page_size_bytes:
                return None
    return sum(spec.page_size_bytes for spec in (*targets, *drafts))


def replicated_draft_max_memory_usage_bytes(config: VllmConfig, groups: list[KVCacheGroupSpec]) -> int | None:
    """Reserve enough shared pool slots for all groups of one max-length request."""
    bytes_per_block = replicated_draft_pool_bytes_per_block(groups)
    if bytes_per_block is None:
        return None
    blocks = sum(
        cdiv(group.kv_cache_spec.max_memory_usage_bytes(config), group.kv_cache_spec.page_size_bytes)
        for group in groups
    )
    return blocks * bytes_per_block
