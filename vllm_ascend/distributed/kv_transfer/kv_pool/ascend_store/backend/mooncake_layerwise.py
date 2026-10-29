# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Group-aware Mooncake sessions using the shared hybrid reachability masks."""

from __future__ import annotations

import hashlib
import json
from copy import copy
from typing import TYPE_CHECKING, Any

from vllm.logger import logger
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ReqMeta,
    block_hash_to_str,
    get_block_hashes,
)

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker


def make_block_key(
    model_name: str, block_hash_or_tail: str, head_or_tp_rank: int, *, namespace: str = "", pp_rank: int = 0
) -> str:
    """One object per stage-local layer stack; retain the PP=1 protocol."""
    if namespace:
        return f"{model_name}@{namespace}@pp_rank:{pp_rank}@{block_hash_or_tail}@{head_or_tp_rank}"
    if pp_rank:
        raise ValueError("Mooncake PP keys require a layout namespace")
    return f"{model_name}@{block_hash_or_tail}@{head_or_tp_rank}"


def make_hit_check_keys(
    model_name: str,
    group_id: int,
    block_hash_hex: str,
    num_ranks: int,
    num_groups: int,
    pp_size: int = 1,
    *,
    namespace: str = "",
) -> list[str]:
    del group_id, num_groups
    return [
        make_block_key(model_name, block_hash_hex, rank, namespace=namespace, pp_rank=stage)
        for stage in range(pp_size)
        for rank in range(num_ranks)
    ]


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


def selected(mask, index: int) -> bool:
    return mask is None or (0 <= index < len(mask) and bool(mask[index]))


def prepare_group_sessions(worker: LayerwiseKVPoolWorker, requests: list[ReqMeta]) -> dict[int, list[ReqMeta]]:
    worker._layer_load_aborted.clear()
    with worker._put_started_keys_lock:
        previously_started = worker._put_started_keys.copy()
    try:
        return _prepare_group_sessions(worker, requests)
    except Exception:
        # Preparation has not submitted payload transfers yet. Roll back other
        # groups too if a later group's session allocation fails.
        worker._layer_load_aborted.set()
        with worker._put_started_keys_lock:
            keys = list(worker._put_started_keys - previously_started)
        worker._queue_layerwise_revoke_keys(keys)
        worker._finish_current_layerwise_load_sessions()
        raise


def _prepare_group_sessions(worker: LayerwiseKVPoolWorker, requests: list[ReqMeta]) -> dict[int, list[ReqMeta]]:
    """Views isolate per-group key slots while preserving real request ownership.

    Hybrid pooling publishes only complete blocks. Partial compressor/SWA state
    is not a portable snapshot: the shared coordinator supplies aligned extents
    and masks for the reachable state at each boundary.
    """
    result: dict[int, list[ReqMeta]] = {group: [] for group in range(worker.num_kv_cache_groups)}
    get_slots: list[tuple[ReqMeta, str, int, int | None]] = []
    tracker = worker._layerwise_session_tracker
    worker._current_layerwise_request_ids = {request.req_id for request in requests}
    worker._current_layerwise_last_chunk_req_ids = {request.req_id for request in requests if request.is_last_chunk}
    for request in requests:
        cached_tokens = request.save_start_token
        if request.load_spec is not None and request.load_spec.can_load:
            cached_tokens = request.load_spec.kvpool_cached_tokens
            if (
                not worker.use_eagle
                and len(worker.cacheable_group_ids) == worker.num_kv_cache_groups
                and request.load_spec.kvpool_store_skip_tokens is not None
            ):
                cached_tokens = request.load_spec.kvpool_store_skip_tokens
        load_masks = worker._compute_reachable_load_masks(request, cached_tokens)
        for group in worker.cacheable_group_ids:
            # Skip private rings before deriving token hashes: their physical
            # capacity need not even be divisible by the hash block size.
            block_size = worker.grouped_block_size[group]
            view = copy(request)
            # Keep the complete group block table: LayerBatchBuilder indexes it
            # using task.group_id. Only key/session metadata is group-local.
            view.save_block_keys = []
            view.load_block_keys = []
            view.load_keys = []
            view.save_last_block_key = view.load_last_block_key = None
            view.partial_block_index = None
            result[group].append(view)
            ids = request.block_ids_by_group[group]
            hashes = get_block_hashes(request.block_hashes, block_size, worker.hash_block_size)
            load_mask = load_masks[group] if load_masks is not None else None
            store_mask = request.store_masks[group] if request.store_masks is not None else None

            def key(index, group=group, hashes=hashes):
                return worker._make_layerwise_full_key(group, block_hash_to_str(hashes[index]))

            start = request.load_spec.vllm_cached_tokens // block_size if request.load_spec is not None else 0
            entries = (
                [
                    (key(index), index)
                    for index in range(start, min(cached_tokens // block_size, len(hashes), len(ids)))
                    if selected(load_mask, index)
                ]
                if request.load_spec is not None and request.load_spec.can_load
                else []
            )
            entries = tracker.prepare_load_entries(request.req_id, entries, group_id=group)
            entries = [
                (name, index)
                for name, index in entries
                if start <= index < min(len(ids), cached_tokens // block_size) and selected(load_mask, index)
            ]
            view.load_key_block_offset = 0
            view.load_block_keys = [None] * (max((index for _, index in entries), default=-1) + 1)
            for name, index in entries:
                view.load_block_keys[index] = name
                get_slots.append((view, name, ids[index], index))

            start = request.save_start_token // block_size
            end = min(request.save_end_token // block_size, len(hashes), len(ids))
            if request.load_spec is not None and request.load_spec.can_load:
                pool_hit = request.load_spec.kvpool_store_skip_tokens
                if pool_hit is None:
                    pool_hit = request.load_spec.kvpool_cached_tokens
                start = max(start, pool_hit // block_size)
            view.save_end_token = end * block_size
            view.save_key_block_offset = start
            view.save_block_keys = [None] * max(0, end - start)
            if not request.can_save or not worker._is_layerwise_save_owner(group):
                continue
            key_indices = [(key(index), index) for index in range(start, end) if selected(store_mask, index)]
            names = [name for name, _ in key_indices]
            with worker._put_started_keys_lock:
                started = set(names) & worker._put_started_keys
            new = [name for name in names if name not in started]
            # A block that is already readable in the pool must never be re-put.
            # Mooncake refuses to write into a committed object
            # (MMC_UNMATCHED_KEY) and the aborted session leaves that object
            # unusable for every reader. `_put_started_keys` only covers writers
            # still in flight, and it is cleared once a session commits, so
            # every later request sharing the prefix — and every concurrent
            # prefill rank, whose keys carry no DP component — would otherwise
            # re-open a session over the same committed blocks forever.
            # The non-layerwise path filters through
            # Backend.requires_exists_before_put; the hybrid path must too.
            if new:
                missing = worker._filter_pool_existing_keys(new)
                if len(missing) != len(new):
                    logger.debug(
                        "Mooncake hybrid: req=%s group=%d skipping %d/%d blocks already pooled",
                        request.req_id,
                        group,
                        len(new) - len(missing),
                        len(new),
                    )
                new = missing
            if new:
                try:
                    codes = worker._start_layerwise_put_keys(new, sum(worker.group_block_len[group]))
                except Exception:
                    worker._queue_layerwise_revoke_keys(new)
                    raise
                started_now = {name for name, code in zip(new, codes, strict=True) if code == 0}
                rejected = [name for name, code in zip(new, codes, strict=True) if code != 0]
                if rejected:
                    # Non-zero means the session was never opened, so this rank
                    # owns nothing to revoke — and revoking a key another rank
                    # committed would discard its object. Report instead of
                    # acting: this used to be dropped in silence, which is how a
                    # permanently unhittable pool went unnoticed.
                    logger.warning(
                        "Mooncake hybrid put_start rejected %d/%d keys for group %d (codes=%s); "
                        "those blocks stay unsaved this step.",
                        len(rejected),
                        len(new),
                        group,
                        [code for code in codes if code != 0][:4],
                    )
                started.update(started_now)
                if started_now:
                    with worker._put_started_keys_lock:
                        worker._put_started_keys.update(started_now)
            for name, index in key_indices:
                if name in started:
                    view.save_block_keys[index - start] = name
            tracker.register_put_keys(
                request.req_id, ((name, index) for name, index in key_indices if name in started), group_id=group
            )
    worker._open_layerwise_get_sessions(get_slots)
    return result
