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


def snapshot_key(worker: LayerwiseKVPoolWorker, request: ReqMeta, group: int, index: int, endpoint: int) -> str:
    incarnation = request.layerwise_request_id
    if not incarnation:
        raise ValueError("Mooncake offload snapshots require a scheduler-assigned request incarnation")
    return worker._make_layerwise_full_key(group, f"offload_v1:{incarnation}:{endpoint}:{index}")


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

    entries = []
    for index in range(end):
        if not selected(mask, index):
            continue
        if ids[index] == 0:
            raise ValueError("Mooncake offload restore selected a null block")
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
    entries = []
    for index in range(start, end):
        # Reachability masks describe published pages, not new private pages.
        if ids[index] == 0 or (index < len(hashes) and mask is not None and index < len(mask) and not mask[index]):
            continue
        if index < len(hashes):
            key = worker._make_layerwise_full_key(group, block_hash_to_str(hashes[index]))
            snapshots.pop(index, None)
        else:
            key = snapshots[index] = snapshot_key(worker, request, group, index, tokens)
        entries.append((key, index - start))
    if partial is not None:
        key = snapshots[partial] = snapshot_key(worker, request, group, partial, tokens)
        entries.append((key, None))
    return entries if worker._is_layerwise_save_owner(group) else []


def _open_puts(worker, slots):
    # One metadata RPC per batch across requests and groups, with each group's
    # own object size. Deduplication also prevents duplicate concurrent writers.
    sizes: dict[str, int] = {}
    for _, key, _, size in slots:
        if key in sizes and sizes[key] != size:
            raise ValueError("Mooncake offload key has inconsistent object sizes")
        sizes[key] = size
    missing = worker._filter_pool_existing_keys(list(sizes))
    started = set()
    try:
        for batch in worker._layerwise_key_batches(missing):
            codes = require_aligned_batch_results(
                "batch_put_start", batch, worker.m_store.batch_put_start(batch, [sizes[key] for key in batch])
            )
            opened = {key for key, code in zip(batch, codes, strict=True) if code == 0}
            started.update(opened)
            with worker._put_started_keys_lock:
                worker._put_started_keys.update(opened)
            rejected = [key for key, code in zip(batch, codes, strict=True) if code != 0]
            # A competing writer may have committed since the existence probe.
            # An unfinished competing PUT is not a safe offload destination.
            if rejected and worker._filter_pool_existing_keys(rejected):
                raise RuntimeError("Mooncake offload could not open a required PUT session")
    except Exception:
        worker._queue_layerwise_revoke_keys(list(started))
        raise
    for view, key, slot, _ in slots:
        if key in started:
            # Use one request's source pages for a deduplicated public write.
            started.remove(key)
            if slot is None:
                view.save_last_block_key = key
            else:
                view.save_block_keys[slot] = key


def prepare_offload_sessions(worker: LayerwiseKVPoolWorker, requests: list[ReqMeta]) -> dict[int, list[ReqMeta]]:
    """Prepare every group once; layers only select their address offsets."""
    worker._layer_load_aborted.clear()
    worker._current_layerwise_request_ids = {request.req_id for request in requests}
    worker._current_layerwise_last_chunk_req_ids = {request.req_id for request in requests if request.is_last_chunk}
    groups: dict[int, list[ReqMeta]] = {group: [] for group in worker.cacheable_group_ids}
    get_slots = []
    put_slots: list[tuple[ReqMeta, str, int | None, int]] = []
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
        if cached != aligned and running:
            masks = None
        for group in worker.cacheable_group_ids:
            view = copy(request)
            view.load_block_keys = []
            view.load_keys = []
            view.save_block_keys = []
            view.partial_block_index = None
            view.save_last_block_key = view.load_last_block_key = None
            view.load_key_block_offset = 0
            hashes = get_block_hashes(request.block_hashes, worker.grouped_block_size[group], worker.hash_block_size)
            entries = _load_entries(worker, request, group, hashes, masks[group] if masks is not None else None)
            view.load_block_keys = [None] * (max((index for _, index in entries), default=-1) + 1)
            for key, index in entries:
                view.load_block_keys[index] = key
                get_slots.append((view, key, request.block_ids_by_group[group][index], index))
            put_slots.extend(
                (view, key, slot, sum(worker.group_block_len[group]))
                for key, slot in _save_entries(
                    worker, request, view, group, hashes, snapshots.setdefault(group, previous.get(group, {}).copy())
                )
            )
            groups[group].append(view)
    try:
        worker._open_layerwise_get_sessions(get_slots)
        worker._check_hybrid_load_errors()
        _open_puts(worker, put_slots)
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
