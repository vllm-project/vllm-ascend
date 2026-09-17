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

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ReqMeta,
    block_hash_to_str,
    get_block_hashes,
)

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


LAYERWISE_DATA_PLANE = "block_key"


def extract_layout_config(extra_config: dict[str, Any]) -> dict[str, Any] | None:
    """Block-key transfer does not opt into GVA-backed physical reuse."""
    del extra_config
    return None


def make_block_key(model_name: str, block_hash_or_tail: str, head_or_tp_rank: int) -> str:
    """Build the canonical one-object-per-block-and-saving-rank key."""
    return f"{model_name}@{block_hash_or_tail}@{head_or_tp_rank}"


def make_hit_check_keys(
    model_name: str,
    group_id: int,
    block_hash_hex: str,
    num_ranks: int,
    num_groups: int,
    pp_size: int = 1,
) -> list[str]:
    del group_id, num_groups, pp_size
    return [make_block_key(model_name, block_hash_hex, rank) for rank in range(num_ranks)]


def validate_topology(parallel_config: Any) -> None:
    """Reject parallel coordinates omitted from Mooncake's block key."""

    def parallel_size(name: str) -> int:
        value = getattr(parallel_config, name, 1)
        return value if isinstance(value, int) and not isinstance(value, bool) else 1

    dimensions = (
        ("pipeline_parallel_size", parallel_size("pipeline_parallel_size")),
        ("prefill_context_parallel_size", parallel_size("prefill_context_parallel_size")),
        ("decode_context_parallel_size", parallel_size("decode_context_parallel_size")),
    )
    unsupported = [f"{name}={size}" for name, size in dimensions if size > 1]
    if unsupported:
        raise ValueError(
            "Mooncake block-key layerwise currently supports TP-only topology; unsupported " + ", ".join(unsupported)
        )


def validate_runtime(*, use_hybrid: bool, has_recurrent_state: bool, tp_mismatch: bool) -> None:
    if use_hybrid and has_recurrent_state:
        raise ValueError("Mooncake hybrid layerwise does not yet support recurrent Mamba state")
    if tp_mismatch:
        raise ValueError("Mooncake layerwise does not yet support prefill/decode TP mismatch")


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


def hybrid_layout_id(kv_cache_config, tp_size: int = 1) -> str:
    """Namespace every pool key, and must agree across processes.

    Only representation-independent fields participate. Hashing the spec
    objects does *not* work: the scheduler holds one merged spec per group
    while a worker holds the per-layer ``UniformTypeKVCacheSpecs`` for the same
    group, so ``asdict`` yields different JSON on each side. That produced two
    disjoint key spaces — writes landed under one prefix and the hit check
    queried another, so the pool never reported a hit, silently and with no
    error anywhere.

    Layer membership, group order and per-group page sizes are identical in
    both representations, and the model name is already part of every key, so
    this still isolates incompatible layouts from each other.
    """
    groups = []
    for group in kv_cache_config.kv_cache_groups:
        groups.append((sorted(group.layer_names), group_block_size_signature(group)))
    # No default= fallback on purpose: anything non-serializable here would have
    # to come from a repr that can differ per process, which is exactly the bug
    # this hash must never reacquire. Fail loudly instead.
    encoded = json.dumps((tp_size, groups), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def hybrid_block_key(
    model: str, layout: str, group: int, family: str, block_size: int, block_hash: str, head: int
) -> str:
    return f"{model}@mooncake_hybrid_v1:{layout}@group:{group}@family:{family}@block:{block_size}@{block_hash}@{head}"


def fence_drains_recv() -> bool:
    """Whether the attention-window fence also quiesces the LOAD queue.

    Off by default: the load queue carries the prefetch that layerwise exists
    to overlap, and per-layer completion is already enforced by
    ``wait_for_layer_load``. Set ``VLLM_ASCEND_KVPOOL_FENCE_DRAIN_RECV=1`` to
    restore the old whole-queue fence when diagnosing interconnect contention
    between loads and collectives.
    """
    return bool(envs.VLLM_ASCEND_KVPOOL_FENCE_DRAIN_RECV)


def send_fence_backlog() -> int:
    """How many queued layer saves may stay outstanding at an attention boundary.

    The attention-window fence used to drain the SAVE queue to zero after every
    layer. The send thread is single-threaded, so a zero backlog means the put
    for layer L must be entirely on the wire before layer L+1's collectives may
    launch: the transfer is serialized against compute instead of overlapping
    it, which is the opposite of what layerwise exists to do. Measured on the
    prefill node this pinned ~26 ms per layer (~1.2 s per 44-layer step).

    Bounding the backlog instead lets the send thread work on layer L-N -- whose
    ``sync_save_events`` have long since completed -- while the compute stream is
    already several layers ahead. Ordering is unaffected because the thread still
    consumes the queue strictly in submission order, and only one ranged put is
    ever in flight. Complete publication is still guaranteed: ``save_kv_layer``
    waits for the final layer's save event at the end of every step, which
    implies every earlier layer has committed.

    0 restores the old drain-to-zero fence; a larger value allows more overlap.
    """
    return max(0, int(envs.VLLM_ASCEND_KVPOOL_FENCE_SEND_BACKLOG))


def selected(mask, index: int) -> bool:
    return mask is None or (0 <= index < len(mask) and bool(mask[index]))


def prepare_group_sessions(worker: KVPoolWorker, requests: list[ReqMeta]) -> dict[int, list[ReqMeta]]:
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


def _prepare_group_sessions(worker: KVPoolWorker, requests: list[ReqMeta]) -> dict[int, list[ReqMeta]]:
    """Views isolate per-group key slots while preserving real request ownership.

    Hybrid pooling publishes only complete blocks. Partial compressor/SWA state
    is not a portable snapshot: the shared coordinator supplies aligned extents
    and masks for the reachable state at each boundary.
    """
    result = {group: [] for group in range(worker.num_kv_cache_groups)}
    get_slots = []
    tracker = worker._layerwise_session_tracker
    worker._current_layerwise_request_ids = {request.req_id for request in requests}
    worker._current_layerwise_last_chunk_req_ids = {request.req_id for request in requests if request.is_last_chunk}
    for request in requests:
        cached_tokens = request.save_start_token
        if request.load_spec is not None and request.load_spec.can_load:
            cached_tokens = request.load_spec.kvpool_cached_tokens
            if not worker.use_eagle and request.load_spec.kvpool_store_skip_tokens is not None:
                cached_tokens = request.load_spec.kvpool_store_skip_tokens
        load_masks = worker._compute_reachable_load_masks(request, cached_tokens)
        for group, block_size in enumerate(worker.grouped_block_size):
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

            def key(index, group=group, block_size=block_size, hashes=hashes):
                return hybrid_block_key(
                    worker.model_name,
                    worker.block_key_hybrid_layout,
                    group,
                    worker.kv_cache_group_families[group],
                    block_size,
                    block_hash_to_str(hashes[index]),
                    worker.head_or_tp_rank,
                )

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
            if not request.can_save or not worker._is_layerwise_save_owner():
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
