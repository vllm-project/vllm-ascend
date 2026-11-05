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
    recurrent = worker.group_uses_align_state[group]
    if spec.layerwise_previous_step and recurrent and not worker._group_reuses_buffer(group):
        return []
    tokens = spec.kvpool_cached_tokens
    if not worker.use_eagle and spec.kvpool_store_skip_tokens is not None:
        tokens = spec.kvpool_store_skip_tokens
    size = worker.grouped_block_size[group]
    end = (tokens + size - 1) // size
    ids = request.block_ids_by_group[group]
    if end > len(ids):
        raise ValueError("Mooncake offload restore exceeds the allocated block table")
    if recurrent and spec.layerwise_previous_step:
        if tokens <= 0:
            return []
        index = end - 1
        if ids[index] == 0:
            raise ValueError("Mooncake offload restore selected a null recurrent checkpoint")
        if spec.layerwise_snapshot_end_token == tokens:
            return [(snapshot_key(worker, request, group, index, tokens), index)]
        if tokens % size or index >= len(hashes):
            # A recurrent state represents one exact endpoint, including the
            # final, unaligned prompt chunk. It cannot be floored to an older
            # hash boundary or rolled back from a later speculative state.
            raise ValueError("Mooncake recurrent restore requires its exact saved snapshot endpoint")
        return [(worker._make_layerwise_full_key(group, block_hash_to_str(hashes[index])), index)]
    snapshot_start = worker._layerwise_load_snapshot_start(request, group, tokens)
    entries = []
    for index in range(end):
        if mask is not None and index < len(mask) and not mask[index]:
            continue
        if recurrent and ids[index] == 0:
            raise ValueError("Mooncake offload restore selected a null recurrent checkpoint")
        if snapshot_start is not None and index >= snapshot_start:
            key = snapshot_key(worker, request, group, index, spec.layerwise_snapshot_end_token)
        elif index < min(tokens // size, len(hashes)):
            key = worker._make_layerwise_full_key(group, block_hash_to_str(hashes[index]))
        else:
            # Never invent a snapshot endpoint for an external prefix hit.
            raise ValueError("Mooncake offload restore has no exact snapshot identity")
        entries.append((key, index))
    return entries


def _save_entries(worker, request, view, group, hashes):
    size = worker.grouped_block_size[group]
    recurrent = worker.group_uses_align_state[group]
    # Resident recurrent state is selected by the model after speculative
    # acceptance. Only scratch-backed recurrent state needs a private snapshot
    # at every chunk endpoint, including an unaligned or not-yet-hashed one.
    reused_state = recurrent and worker._group_reuses_buffer(group)
    partial = get_partial_block_index(request.target_token_len, size, len(hashes), not recurrent or reused_state)
    if reused_state and request.target_token_len > 0:
        transfer_size = worker.cache_transfer_granularity
        # Match RequestTracker's snapshot identity in the transfer LCM unit.
        # A smaller recurrent group can already be aligned/hashed while the
        # overall chunk still needs an immutable request endpoint snapshot.
        if request.target_token_len % transfer_size or (
            request.target_token_len // transfer_size
            > len(request.block_hashes) * worker.hash_block_size // transfer_size
        ):
            partial = (request.target_token_len - 1) // size
    end = worker._layerwise_save_full_end(request, group, size, partial)
    start = min(request.save_start_token // size, end)
    view.save_key_block_offset = start
    view.save_block_keys = [None] * (end - start)
    view.partial_block_index = partial
    if not request.can_save or not worker._is_layerwise_save_owner(group):
        return []
    ids = request.block_ids_by_group[group]
    if end > len(ids) or (partial is not None and partial >= len(ids)):
        raise ValueError("Mooncake offload save exceeds the allocated block table")
    mask = request.store_masks[group] if request.store_masks is not None else None
    entries = []
    for index in range(start, end):
        if mask is not None and index < len(mask) and not mask[index]:
            continue
        key = (
            worker._make_layerwise_full_key(group, block_hash_to_str(hashes[index]))
            if index < len(hashes)
            else snapshot_key(worker, request, group, index, request.target_token_len)
        )
        entries.append((key, index - start))
    if partial is not None:
        entries.append((snapshot_key(worker, request, group, partial, request.target_token_len), None))
    return entries


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
    for request in requests:
        cached = request.load_spec.kvpool_cached_tokens if request.load_spec is not None else 0
        masks = worker._compute_reachable_load_masks(request, cached)
        for group in worker.cacheable_group_ids:
            view = copy(request)
            view.load_block_keys = []
            view.load_keys = []
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
                for key, slot in _save_entries(worker, request, view, group, hashes)
            )
            groups[group].append(view)
    try:
        worker._open_layerwise_get_sessions(get_slots)
        worker._check_layerwise_transfer_errors()
        _open_puts(worker, put_slots)
    except Exception:
        worker._layer_load_aborted.set()
        worker._finish_current_layerwise_load_sessions()
        raise
    return groups
