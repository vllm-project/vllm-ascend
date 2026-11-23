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
    window = getattr(config, "dsv4_continuation_windows", {}).get(group)
    prefill_end = getattr(request, "num_prompt_tokens", None)
    if window is None and prefill_end is not None and request.target_token_len <= prefill_end:
        window = getattr(config, "dsv4_nominal_windows", {}).get(group)
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
    live_window = running and group in getattr(config, "dsv4_window_group_ids", ())
    first = _continuation_window_start(worker, group, tokens, size, request) if live_window else None
    committed = getattr(worker, "_offload_snapshot_endpoints", {}).get((request.req_id, group)) if running else None
    incarnation = request.layerwise_request_id if committed is not None else None
    strict = (
        committed is not None
        and committed[0] == incarnation
        and not worker.group_uses_align_state[group]
        and getattr(config, "dsv4_component_identity", None) is not None
    )
    if committed is not None and committed[0] != incarnation:
        snapshots = {}
    entries = []
    for index in range(first or 0, end):
        if not live_window and not selected(mask, index):
            continue
        if ids[index] == 0:
            if live_window:
                continue
            raise ValueError("Mooncake offload restore selected a null block")
        key = snapshots.get(index)
        if strict and (committed[1] != tokens or key is None):
            raise ValueError("Mooncake offload has no committed version for a live continuation page")
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


def _save_dsv4_entries(worker, request, view, group, hashes):
    size = worker.grouped_block_size[group]
    recurrent = worker.group_uses_align_state[group]
    # Resident recurrent state is selected by the model after speculative
    # acceptance. Only scratch-backed recurrent state needs a private snapshot
    # at every chunk endpoint, including an unaligned or not-yet-hashed one.
    reused_state = recurrent and group in worker._reused_layers_by_group
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
    end = request.target_token_len // size
    if partial is not None:
        end = min(end, partial)
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
    # RequestTracker describes the private suffix in transfer-LCM units.
    # A finer group can have hashed complete pages inside that suffix. Saving
    # those under public keys makes the next step's private-key GET miss.
    snapshot_start = None
    if not recurrent:
        transfer_size = getattr(worker, "cache_transfer_granularity", size)
        full_transfer_blocks = request.target_token_len // transfer_size
        hashed_transfer_blocks = len(hashes) * size // transfer_size
        if request.target_token_len % transfer_size or full_transfer_blocks > hashed_transfer_blocks:
            snapshot_start = min(full_transfer_blocks, hashed_transfer_blocks) * transfer_size // size
    entries = []
    window_start = _continuation_window_start(worker, group, request.target_token_len, size, request)
    if window_start is None and group in getattr(
        getattr(worker, "kv_cache_config", None), "dsv4_window_group_ids", ()
    ) and group in worker._reused_layers_by_group:
        # Speculative decoding disables window trimming. Keep every live
        # page needed for continuation, including pages outside the public mask.
        window_start = 0
    for index in range(start, end):
        if ids[index] == 0:
            continue
        public_page = mask is None or (index < len(mask) and mask[index])
        private_window_page = window_start is not None and index >= window_start
        if window_start is None:
            if mask is not None and index < len(mask) and not mask[index]:
                continue
        elif not public_page and not private_window_page:
            continue
        use_private_snapshot = (snapshot_start is not None and index >= snapshot_start) or (
            private_window_page and not public_page
        )
        key = (
            snapshot_key(worker, request, group, index, request.target_token_len)
            if use_private_snapshot or index >= len(hashes)
            else worker._make_layerwise_full_key(group, block_hash_to_str(hashes[index]))
        )
        entries.append((key, index - start))
    if reused_state and request.target_token_len > 0:
        partial = (request.target_token_len - 1) // size
        view.partial_block_index = partial
    if partial is not None and partial < len(ids) and ids[partial] != 0:
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
    fallback_slots = []
    aliases = {}
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
            unresolved = worker._filter_pool_existing_keys(rejected) if rejected else []
            codes_by_key = dict(zip(batch, codes, strict=True))
            conflicts = {
                key for key in unresolved
                if codes_by_key[key] == _OBJECT_ALREADY_EXISTS and "offload_v1:" not in key
            }
            fatal = [key for key in unresolved if key not in conflicts]
            if fatal:
                details = [
                    dict(result=codes_by_key[key], bytes=sizes[key], private_snapshot="offload_v1:" in key)
                    for key in fatal[:8]
                ]
                raise RuntimeError(
                    f"Mooncake offload could not open a required PUT session: "
                    f"failed_keys={len(fatal)} samples={details}"
                )
            # Do not overwrite a competing writer or wait for a peer that
            # may need this rank's EP collectives to finish its own PUT.
            for view, key, slot, size in slots:
                if key not in conflicts:
                    continue
                incarnation = view.layerwise_request_id
                if not incarnation:
                    raise ValueError("Offload PUT conflict requires a request incarnation")
                private_key = f"{key}@offload_v1:fallback:{incarnation}"
                fallback_slots.append((view, private_key, slot, size))
                aliases[(view.req_id, incarnation, key)] = private_key
        if fallback_slots:
            with worker._put_started_keys_lock:
                before_fallback = worker._put_started_keys.copy()
            try:
                _open_puts(worker, fallback_slots)
            finally:
                with worker._put_started_keys_lock:
                    started.update(worker._put_started_keys - before_fallback)
    except Exception:
        worker._queue_layerwise_revoke_keys(list(started))
        raise
    for view, key, slot, _ in slots:
        if key in started:
            if slot is None:
                view.save_last_block_key = key
            else:
                view.save_block_keys[slot] = key
    return aliases


def prepare_offload_sessions(worker: LayerwiseKVPoolWorker, requests: list[ReqMeta]) -> dict[int, list[ReqMeta]]:
    """Prepare every group once; layers only select their address offsets."""
    worker._layer_load_aborted.clear()
    worker._current_layerwise_request_ids = {request.req_id for request in requests}
    worker._current_layerwise_last_chunk_req_ids = {request.req_id for request in requests if request.is_last_chunk}
    groups: dict[int, list[ReqMeta]] = {group: [] for group in worker.cacheable_group_ids}
    get_slots = []
    put_slots: list[tuple[ReqMeta, str, int | None, int]] = []
    pending = {}
    ordinary_saves = []
    dsv4 = getattr(getattr(worker, "kv_cache_config", None), "dsv4_component_identity", None) is not None
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
            group_snapshots = snapshots.setdefault(group, previous.get(group, {}).copy())
            if dsv4 and not worker.group_uses_align_state[group]:
                saves = _save_dsv4_entries(worker, request, view, group, hashes)
                if request.can_save and request.layerwise_request_id and not request.is_last_chunk and (
                    group in worker._reused_layers_by_group
                ):
                    pages = {index: key for key, index in entries}
                    for key, slot in saves:
                        index = view.partial_block_index if slot is None else view.save_key_block_offset + slot
                        pages[index] = key
                    ids = request.block_ids_by_group[group]
                    first = _continuation_window_start(
                        worker, group, request.target_token_len, worker.grouped_block_size[group], request
                    )
                    pending[(request.req_id, request.layerwise_request_id, group)] = (
                        request.target_token_len,
                        {index: key for index, key in pages.items() if (first or 0) <= index < len(ids) and ids[index]},
                    )
            else:
                saves = _save_entries(worker, request, view, group, hashes, group_snapshots)
                ordinary_saves.append((request, group, view, saves, group_snapshots))
            put_slots.extend((view, key, slot, sum(worker.group_block_len[group])) for key, slot in saves)
            groups[group].append(view)
    try:
        worker._open_layerwise_get_sessions(get_slots)
        worker._check_hybrid_load_errors()
        aliases = _open_puts(worker, put_slots)
        for request, group, view, saves, snapshots in ordinary_saves:
            identity = (request.req_id, request.layerwise_request_id, group)
            for key, slot in saves:
                fallback = aliases.get((*identity[:2], key))
                if fallback is None:
                    continue
                index = view.partial_block_index if slot is None else view.save_key_block_offset + slot
                # A private tail snapshot can supersede a public save for the same page.
                if index not in snapshots:
                    _, pages = pending.setdefault(identity, (request.target_token_len, {}))
                    pages[index] = fallback
            if identity in pending:
                pending[identity][1].update(snapshots)
        worker._offload_pending_pages = {
            identity: (endpoint, {index: aliases.get((*identity[:2], key), key) for index, key in pages.items()})
            for identity, (endpoint, pages) in pending.items()
        }
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
