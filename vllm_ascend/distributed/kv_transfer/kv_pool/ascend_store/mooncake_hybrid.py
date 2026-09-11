# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Group-aware Mooncake sessions using the shared hybrid reachability masks."""

from __future__ import annotations

import hashlib
import json
from copy import copy
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ReqMeta,
    block_hash_to_str,
    get_block_hashes,
)

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


def hybrid_layout_id(kv_cache_config, tp_size: int = 1) -> str:
    """Stable across processes; isolates group order, layer membership and specs."""
    groups = []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        fields = asdict(spec) if is_dataclass(spec) else vars(spec)
        groups.append((sorted(group.layer_names), type(spec).__name__, fields))
    encoded = json.dumps((tp_size, groups), sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def hybrid_block_key(
    model: str, layout: str, group: int, family: str, block_size: int, block_hash: str, head: int
) -> str:
    return f"{model}@mooncake_hybrid_v1:{layout}@group:{group}@family:{family}@block:{block_size}@{block_hash}@{head}"


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
        worker._finish_current_mooncake_load_sessions()
        raise


def _prepare_group_sessions(worker: KVPoolWorker, requests: list[ReqMeta]) -> dict[int, list[ReqMeta]]:
    """Views isolate per-group key slots while preserving real request ownership.

    Hybrid pooling publishes only complete blocks. Partial compressor/SWA state
    is not a portable snapshot: the shared coordinator supplies aligned extents
    and masks for the reachable state at each boundary.
    """
    result = {group: [] for group in range(worker.num_kv_cache_groups)}
    get_slots = []
    tracker = worker._mooncake_session_tracker
    worker._current_mooncake_request_ids = {request.req_id for request in requests}
    worker._current_mooncake_last_chunk_req_ids = {request.req_id for request in requests if request.is_last_chunk}
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
                    worker.mooncake_hybrid_layout,
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
            if new:
                try:
                    codes = worker._start_mooncake_put_keys(new, sum(worker.group_block_len[group]))
                except Exception:
                    worker._queue_layerwise_revoke_keys(new)
                    raise
                started.update(name for name, code in zip(new, codes, strict=True) if code == 0)
                with worker._put_started_keys_lock:
                    worker._put_started_keys.update(started)
            for name, index in key_indices:
                if name in started:
                    view.save_block_keys[index - start] = name
            tracker.register_put_keys(
                request.req_id, ((name, index) for name, index in key_indices if name in started), group_id=group
            )
    worker._open_mooncake_get_sessions(get_slots)
    return result
