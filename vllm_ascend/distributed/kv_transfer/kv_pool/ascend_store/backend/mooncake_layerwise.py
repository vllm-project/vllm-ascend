# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mooncake layerwise wire keys and layout-identity helpers."""

from __future__ import annotations

import hashlib
import json
from copy import copy
from typing import Any

from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.ascend_config import KVPPConfig


def make_block_key(
    model_name: str, block_hash_or_tail: str, head_or_tp_rank: int, *, namespace: str = "", pp_rank: int = 0
) -> str:
    """One object per stage-local layer stack; retain the PP=1 protocol."""
    if namespace:
        return f"{model_name}@{namespace}@pp_rank:{pp_rank}@{block_hash_or_tail}@{head_or_tp_rank}"
    if pp_rank:
        raise ValueError("Mooncake PP keys require a layout namespace")
    return f"{model_name}@{block_hash_or_tail}@{head_or_tp_rank}"


def layerwise_topology_namespace(vllm_config: Any, kv_cache_config: Any = None) -> str:
    """Isolate stage-local objects by the actual partition and model/cache config.

    Use vLLM's partition API, including VLLM_PP_LAYER_PARTITION, rather than
    assuming equal stage sizes. No worker-local layer membership enters this
    hash: the scheduler and every stage must derive the same namespace.
    """
    parallel = vllm_config.parallel_config
    kvpp_size = KVPPConfig.from_vllm_config(vllm_config).size if getattr(vllm_config, "additional_config", None) else 1
    kvpp_namespace = (
        f"kvpp-tp{parallel.tensor_parallel_size}-pcp{parallel.prefill_context_parallel_size}" if kvpp_size > 1 else ""
    )
    if parallel.pipeline_parallel_size == 1:
        # Owner-only objects have different sizes and layer offsets from the
        # ordinary replicated layout, even with the same model and TP size.
        return kvpp_namespace
    model = vllm_config.model_config
    partitions = []
    previous_end = 0
    for stage in range(parallel.pipeline_parallel_size):
        stage_config = copy(parallel)
        stage_config.rank = stage * parallel.tensor_parallel_size
        start, end = model.get_layers_start_end_indices(stage_config)
        if start != previous_end or end <= start:
            raise ValueError("Mooncake PP requires contiguous, nonempty model layer partitions")
        partitions.append((start, end))
        previous_end = end
    if previous_end != model.get_total_num_hidden_layers():
        raise ValueError("Mooncake PP partitions must cover every model layer")
    # EngineCore rewrites its CacheConfig.block_size to the minimum group
    # block size after spawning workers. Workers still hold the CLI value
    # (e.g. 32 while a DSV4 state group uses 2). Hashing either the raw value
    # or CacheConfig.compute_hash() would split lookup and PUT namespaces.
    # Normalize a copy from the resolved groups shared by both roles; never
    # change the worker's config, which model/cache code still needs.
    cache = copy(vllm_config.cache_config)
    if kv_cache_config is not None and kv_cache_config.kv_cache_groups:
        cache.block_size = min(
            size for group in kv_cache_config.kv_cache_groups for size in group_block_size_signature(group)
        )
    speculative = vllm_config.speculative_config
    identity: tuple[Any, ...] = (
        parallel.tensor_parallel_size,
        partitions,
        model.compute_hash(),
        cache.compute_hash(),
        cache.cache_dtype,
        cache.block_size,
        speculative.compute_hash() if speculative is not None else None,
    )
    if kvpp_namespace:
        identity += (kvpp_namespace,)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return f"mooncake_pp_v3:{digest}"


def validate_pp_groups(kv_cache_config: Any, parallel_config: Any) -> None:
    """Fail closed when an all-stage readability check cannot be satisfied.

    Empty projected groups need a group-owner manifest shared with the
    scheduler. Until that protocol exists, do not silently report permanent
    misses or ignore a group's state (including draft-only groups).
    """
    if parallel_config.pipeline_parallel_size == 1 or kv_cache_config is None:
        return
    empty = [index for index, group in enumerate(kv_cache_config.kv_cache_groups) if not group.layer_names]
    if empty:
        raise ValueError(
            f"Mooncake layerwise PP requires each stage to own layers in every KV cache group; empty groups: {empty}. "
            "Use a partition covering all groups, or disable layerwise pooling. Draft-only groups are not supported."
        )


def group_block_size_signature(group) -> tuple[int, ...]:
    """Per-group page sizes, independent of how the spec is packaged.

    The scheduler and a worker can hold the *same* group in two different
    shapes: one merged spec versus a ``UniformTypeKVCacheSpecs`` mapping every
    layer name to its own spec. Both describe the same pages, so reduce either
    shape to the set of block sizes it contains rather than reading a single
    representative (which depends on dict order).
    """
    spec = group.kv_cache_spec
    specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
    return tuple(sorted({int(sub.block_size) for sub in specs}))


def hybrid_layout_id(kv_cache_config, tp_size: int = 1, *, namespace: str = "") -> str:
    """Namespace every pool key, and must agree across processes.

    Only representation-independent fields participate. Hashing the spec
    objects does *not* work: the scheduler holds one merged spec per group
    while a worker holds the per-layer ``UniformTypeKVCacheSpecs`` for the same
    group, so ``asdict`` yields different JSON on each side. That produced two
    disjoint key spaces — writes landed under one prefix and the hit check
    queried another, so the pool never reported a hit, silently and with no
    error anywhere.

    Without PP, layer membership, group order and page sizes are identical in
    both representations. With PP, membership is stage-local: use the shared
    model/config/partition namespace plus the globally ordered page signatures.
    """
    groups = []
    for group in kv_cache_config.kv_cache_groups:
        # PP projects global groups onto each stage. The model/partition hash
        # supplies the global identity; local membership must not split it.
        groups.append(([] if namespace else sorted(group.layer_names), group_block_size_signature(group)))
    # No default= fallback on purpose: anything non-serializable here would have
    # to come from a repr that can differ per process, which is exactly the bug
    # this hash must never reacquire. Fail loudly instead.
    identity = (namespace, tp_size, groups) if namespace else (tp_size, groups)
    encoded = json.dumps(identity, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def hybrid_block_key(
    model: str, layout: str, group: int, block_size: int, block_hash: str, head: int, *, pp_rank: int | None = None
) -> str:
    stage = "" if pp_rank is None else f"@pp_rank:{pp_rank}"
    return f"{model}@mooncake_hybrid_v1:{layout}{stage}@group:{group}@block:{block_size}@{block_hash}@{head}"
