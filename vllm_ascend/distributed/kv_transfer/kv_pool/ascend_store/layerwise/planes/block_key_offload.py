# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-local Mooncake sessions for prefill scratch-buffer reuse.

Full blocks retain the ordinary pool wire format. Unhashed pages use immutable,
request-incarnation/endpoint keys so the next chunk cannot overwrite an active
read. There is no per-layer key or session preparation on the forward path.
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import require_aligned_batch_results
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import get_partial_block_index
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.planes.block_key import selected
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ReqMeta,
    block_hash_to_str,
    get_block_hashes,
)

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker


_OBJECT_ALREADY_EXISTS = -705


def snapshot_key(worker: LayerwiseKVPoolWorker, request: ReqMeta, group: int, index: int, endpoint: int) -> str:
    incarnation = request.layerwise_request_id
    if not incarnation:
        raise ValueError("Mooncake offload snapshots require a scheduler-assigned request incarnation")
    return worker._make_layerwise_full_key(group, f"offload_v1:{incarnation}:{endpoint}:{index}")


def _continuation_window_start(worker, group, endpoint, page_tokens, request=None):
    config = getattr(worker, "kv_cache_config", None)
    window = getattr(config, "block_key_continuation_windows", {}).get(group)
    prefill_end = getattr(request, "num_prompt_tokens", None)
    if window is None and prefill_end is not None and request.target_token_len <= prefill_end:
        window = getattr(config, "block_key_prefill_windows", {}).get(group)
    return None if window is None else max(0, endpoint - window + 1) // page_tokens


def _load_entries(worker, request, group, hashes, mask):
    spec = request.load_spec
    if spec is None or not spec.can_load:
        return []
    running = spec.layerwise_previous_step
    if running and group not in worker._reused_layers_by_group:
        return []
    tokens = spec.kvpool_cached_tokens
    size = worker.grouped_block_size[group]
    end = (tokens + size - 1) // size
    ids = request.block_ids_by_group[group]
    if end > len(ids):
        raise ValueError("Mooncake offload restore exceeds the allocated block table")
    snapshots = worker._snapshot_keys.get(request.req_id, {}).get(group, {}) if running else {}
    if running and worker.group_uses_align_state[group] and end:
        index = end - 1
        expected = snapshot_key(worker, request, group, index, tokens)
        if ids[index] == 0 or snapshots.get(index) != expected:
            raise ValueError("Mooncake recurrent restore requires its exact saved snapshot endpoint")
        return [(expected, index)]

    config = getattr(worker, "kv_cache_config", None)
    live_window = running and group in getattr(config, "block_key_prefill_windows", {})
    first = _continuation_window_start(worker, group, tokens, size, request) if live_window else None
    checkpoint = getattr(worker, "_offload_active_pages", {}).get(
        (request.req_id, request.layerwise_request_id, group)
    ) if live_window else None
    saved_pages = checkpoint[1] if checkpoint is not None else snapshots
    entries = []
    for index in range(first or 0, end):
        if live_window:
            if mask is not None and index < len(mask) and not mask[index] and index not in saved_pages:
                continue
        elif index not in snapshots and not selected(mask, index):
            continue
        if ids[index] == 0:
            if live_window:
                continue
            raise ValueError("Mooncake offload restore selected a null block")
        if checkpoint is not None:
            endpoint, pages = checkpoint
            if endpoint != tokens or index not in pages:
                raise ValueError("Mooncake offload has no committed version for a live continuation page")
            entries.append((pages[index], index))
            continue
        key = snapshots.get(index)
        if key is None:
            if index >= min(tokens // size, len(hashes)):
                raise ValueError("Mooncake offload restore has no exact snapshot identity")
            key = worker._make_layerwise_full_key(group, block_hash_to_str(hashes[index]))
        entries.append((key, index))
    return entries


def _save_entries(worker, request, view, group, hashes, snapshots):
    if not request.can_save:
        return []
    size = worker.grouped_block_size[group]
    recurrent = worker.group_uses_align_state[group]
    reuses_buffer = group in worker._reused_layers_by_group
    reused_state = recurrent and reuses_buffer
    if (
        reused_state
        and worker.vllm_config.speculative_config is not None
        and request.num_prompt_tokens is not None
        and request.target_token_len > request.num_prompt_tokens
    ):
        raise NotImplementedError("Shared recurrent state supports speculative prefill, not decode commits")
    tokens = request.target_token_len
    partial = get_partial_block_index(tokens, size, len(hashes), reuses_buffer and not recurrent)
    if reused_state and tokens > 0:
        partial = (tokens - 1) // size
    end = tokens // size
    if recurrent or not reuses_buffer:
        end = min(end, len(hashes))
    if not recurrent and partial is not None:
        end = min(end, partial)
    start = request.save_start_token // size
    load_spec = request.load_spec
    if load_spec is not None and not load_spec.layerwise_previous_step:
        start = load_spec.clamp_save_start(start, size)
    start = min(start, end)
    view.save_key_block_offset = start
    view.save_block_keys = [None] * (end - start)
    view.partial_block_index = partial

    ids = request.block_ids_by_group[group]
    if end > len(ids) or (partial is not None and (partial >= len(ids) or ids[partial] == 0)):
        raise ValueError("Mooncake offload save has no allocated checkpoint or block")
    mask = request.store_masks[group] if request.store_masks is not None else None
    snapshot_start = None
    transfer_size = getattr(worker, "cache_transfer_granularity", size)
    if not recurrent and size < transfer_size:
        # A transfer unit can contain several complete pages of this group.
        public_tokens = min(tokens, len(hashes) * size) // transfer_size * transfer_size
        snapshot_start = public_tokens // size
    continuation_start = None
    prompt_end = getattr(request, "num_prompt_tokens", None)
    if reuses_buffer and not recurrent and prompt_end is not None and tokens < prompt_end:
        continuation_start = _continuation_window_start(worker, group, tokens, size, request)

    entries = []
    for index in range(start, end):
        if ids[index] == 0:
            continue
        private = index >= len(hashes) or (snapshot_start is not None and index >= snapshot_start)
        if not private and mask is not None and index < len(mask) and not mask[index]:
            # Public prefix retention can omit pages needed by the next chunk.
            if continuation_start is None or index < continuation_start:
                continue
            private = True
        if not private:
            key = worker._make_layerwise_full_key(group, block_hash_to_str(hashes[index]))
            snapshots.pop(index, None)
        else:
            key = snapshots[index] = snapshot_key(worker, request, group, index, tokens)
        entries.append((key, index - start))
    if partial is not None:
        key = snapshots[partial] = snapshot_key(worker, request, group, partial, tokens)
        entries.append((key, None))
    return entries if worker._is_layerwise_save_owner(group) else []


def _open_missing(worker, target_sizes, started, *, allow_fallback):
    conflicts = set()
    missing = worker._filter_pool_existing_keys(list(target_sizes))
    for batch in worker._layerwise_key_batches(missing):
        codes = require_aligned_batch_results(
            "batch_put_start", batch, worker.m_store.batch_put_start(batch, [target_sizes[key] for key in batch])
        )
        opened = {key for key, code in zip(batch, codes, strict=True) if code == 0}
        started.update(opened)
        with worker._put_started_keys_lock:
            worker._put_started_keys.update(opened)
        rejected = {key: code for key, code in zip(batch, codes, strict=True) if code != 0}
        for key in worker._filter_pool_existing_keys(list(rejected)):
            if not allow_fallback or rejected[key] != _OBJECT_ALREADY_EXISTS or "offload_v1:" in key:
                raise RuntimeError("Mooncake offload could not open a required PUT session")
            conflicts.add(key)
    return conflicts


def _open_puts(worker, slots):
    # One metadata RPC per batch across requests and groups, with each group's
    # own object size. Deduplication also prevents duplicate concurrent writers.
    sizes: dict[str, int] = {}
    for _, key, _, size in slots:
        if key in sizes and sizes[key] != size:
            raise ValueError("Mooncake offload key has inconsistent object sizes")
        sizes[key] = size
    started = set()

    fallbacks = []
    resolved_slots = []
    fallback_sizes = {}
    try:
        conflicts = _open_missing(worker, sizes, started, allow_fallback=True)
        for view, key, slot, size in slots:
            if key in conflicts:
                incarnation = view.layerwise_request_id
                if not incarnation:
                    raise ValueError("Offload PUT conflict requires a request incarnation")
                # Do not wait here: the public writer may need this rank to
                # enter an EP collective before it can finish the write.
                key = f"{key}@offload_v1:fallback:{incarnation}"
                fallback_sizes[key] = size
                fallbacks.append((view, slot, key))
            resolved_slots.append((view, key, slot, size))
        if fallback_sizes:
            _open_missing(worker, fallback_sizes, started, allow_fallback=False)
    except Exception:
        worker._queue_layerwise_revoke_keys(list(started))
        raise
    for view, key, slot, _ in resolved_slots:
        if key in started:
            # Use one request's source pages for a deduplicated public write.
            started.remove(key)
            if slot is None:
                view.save_last_block_key = key
            else:
                view.save_block_keys[slot] = key
    return fallbacks


def prepare_offload_sessions(worker: LayerwiseKVPoolWorker, requests: list[ReqMeta]) -> dict[int, list[ReqMeta]]:
    """Prepare every group once; layers only select their address offsets."""
    worker._layer_load_aborted.clear()
    worker._current_layerwise_request_ids = {request.req_id for request in requests}
    worker._current_layerwise_last_chunk_req_ids = {request.req_id for request in requests if request.is_last_chunk}
    groups: dict[int, list[ReqMeta]] = {group: [] for group in worker.cacheable_group_ids}
    get_slots = []
    put_slots: list[tuple[ReqMeta, str, int | None, int]] = []
    pending = {}
    save_records = {}
    config = getattr(worker, "kv_cache_config", None)
    window_groups = getattr(config, "block_key_prefill_windows", {})
    next_snapshots: dict[str, dict[int, dict[int, str]]] = {}
    for request in requests:
        load_spec = request.load_spec
        running = load_spec is not None and load_spec.layerwise_previous_step
        previous = worker._snapshot_keys.get(request.req_id, {}) if running else {}
        snapshots = next_snapshots.setdefault(request.req_id, {})
        cached = load_spec.kvpool_cached_tokens if load_spec is not None and load_spec.can_load else 0
        # A running chunk can end inside a block. Use the aligned reachability
        # window for its full pages and let private snapshots cover the tail.
        aligned = cached // worker.cache_transfer_granularity * worker.cache_transfer_granularity
        masks = worker._compute_reachable_load_masks(request, aligned)
        for group in worker.cacheable_group_ids:
            view = copy(request)
            view.load_block_keys = []
            view.load_keys = []
            view.save_block_keys = []
            view.partial_block_index = None
            view.save_last_block_key = view.load_last_block_key = None
            view.load_key_block_offset = 0
            hashes = get_block_hashes(request.block_hashes, worker.grouped_block_size[group], worker.hash_block_size)
            mask = masks[group] if masks is not None else None
            if cached != aligned and running and group not in window_groups:
                mask = None
            entries = _load_entries(worker, request, group, hashes, mask)
            view.load_block_keys = [None] * (max((index for _, index in entries), default=-1) + 1)
            for key, index in entries:
                view.load_block_keys[index] = key
                get_slots.append((view, key, request.block_ids_by_group[group][index], index))
            group_snapshots = snapshots.setdefault(group, previous.get(group, {}).copy())
            saves = _save_entries(worker, request, view, group, hashes, group_snapshots)
            checkpoint_pages = None
            if (
                group in window_groups
                and not worker.group_uses_align_state[group]
                and request.can_save
                and request.layerwise_request_id
                and not request.is_last_chunk
                and group in worker._reused_layers_by_group
            ):
                pages = {index: key for key, index in entries}
                for key, slot in saves:
                    index = view.partial_block_index if slot is None else view.save_key_block_offset + slot
                    pages[index] = key
                ids = request.block_ids_by_group[group]
                first = _continuation_window_start(
                    worker, group, request.target_token_len, worker.grouped_block_size[group], request
                )
                checkpoint_pages = {
                    index: key for index, key in pages.items() if (first or 0) <= index < len(ids) and ids[index]
                }
                pending[(request.req_id, request.layerwise_request_id, group)] = (
                    request.target_token_len, checkpoint_pages
                )
            if saves:
                save_records[id(view)] = (group_snapshots, checkpoint_pages)
            put_slots.extend((view, key, slot, sum(worker.group_block_len[group])) for key, slot in saves)
            groups[group].append(view)
    try:
        worker._open_layerwise_get_sessions(get_slots)
        worker._check_hybrid_load_errors()
        for view, slot, key in _open_puts(worker, put_slots):
            index = view.partial_block_index if slot is None else view.save_key_block_offset + slot
            group_snapshots, checkpoint_pages = save_records[id(view)]
            group_snapshots[index] = key
            if checkpoint_pages is not None and index in checkpoint_pages:
                checkpoint_pages[index] = key
        worker._offload_pending_pages = pending
        for views in groups.values():
            for view in views:
                if not any(view.save_block_keys):
                    view.save_block_keys = []
    except Exception:
        worker._layer_load_aborted.set()
        worker._data_plane.finish_sessions()
        raise
    worker._snapshot_keys.update(next_snapshots)
    return groups
