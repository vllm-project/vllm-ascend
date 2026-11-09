# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Block-key plane: the pool stores one addressable object per block.

Two session protocols live here, both ending in the same worker store I/O:

* Single group (``block_key_hybrid`` unset): one group, one hash-addressed
  object per block plus a request-scoped trailing partial-block key. The
  requests themselves carry the session slots.
* Hybrid (multi-group): per-group request views isolate each group's key
  slots while preserving real request ownership; only complete blocks are
  published, and reachability masks gate both load and save entries.
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.planes.base import (
    LayerwiseDataPlane,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ReqMeta,
    block_hash_to_str,
    get_block_hashes,
)

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import (
        LayerwiseKVPoolWorker,
    )


def selected(mask, index: int) -> bool:
    return mask is None or (0 <= index < len(mask) and bool(mask[index]))


class BlockKeyDataPlane(LayerwiseDataPlane):
    """Session bookkeeping for backends addressing one object per block."""

    def __init__(self, worker: "LayerwiseKVPoolWorker"):
        self._worker = worker

    # --- step entry points -----------------------------------------------------

    def prepare_sessions(self, requests: list[ReqMeta]) -> dict[int, list[ReqMeta]]:
        worker = self._worker
        if worker.block_key_hybrid:
            return self._prepare_hybrid_sessions(requests)
        self._prepare_single_group_sessions(requests)
        return {}

    def prepare_load_gvas(self, requests: list[ReqMeta]) -> None:
        """The block-key plane addresses pool objects, not worker GVAs."""

    def alloc_save_gvas(self, requests: list[ReqMeta]) -> None:
        """The block-key plane addresses pool objects, not worker GVAs."""

    def finish_current_load_sessions(self) -> None:
        worker = self._worker
        if worker._layer_load_aborted.is_set():
            req_ids = worker._current_layerwise_request_ids.copy()
            worker._release_layerwise_requests_for_retry(req_ids)
        else:
            req_ids = worker._current_layerwise_last_chunk_req_ids.copy()
            worker._release_layerwise_requests_terminal(req_ids)
        worker._current_layerwise_request_ids.difference_update(req_ids)
        worker._current_layerwise_last_chunk_req_ids.difference_update(req_ids)

    # --- single-group protocol ---------------------------------------------------

    def _prepare_single_group_sessions(self, requests: list[ReqMeta]) -> None:
        worker = self._worker
        worker._layer_load_aborted.clear()
        worker._current_layerwise_request_ids = {request.req_id for request in requests}
        worker._current_layerwise_last_chunk_req_ids = {request.req_id for request in requests if request.is_last_chunk}
        get_key_slots: list[tuple[ReqMeta, str, int, int | None]] = []
        for request in requests:
            get_key_slots.extend(
                (request, key, block_id, slot) for key, block_id, slot in self._prepare_single_get_session(request)
            )
        worker._open_layerwise_get_sessions(get_key_slots)
        for request in requests:
            self._prepare_single_put_session(request)

    def _prepare_single_put_session(self, request: ReqMeta) -> None:
        worker = self._worker
        request.save_block_keys = []
        request.save_last_block_key = None
        if not worker._is_layerwise_save_owner() or not request.can_save:
            return

        start_block = request.save_start_token // worker.block_size
        end_block = request.save_end_token // worker.block_size
        if request.load_spec is not None and request.load_spec.can_load:
            pool_hit_tokens = (
                request.load_spec.kvpool_store_skip_tokens
                if request.load_spec.kvpool_store_skip_tokens is not None
                else request.load_spec.kvpool_cached_tokens
            )
            start_block = max(start_block, pool_hit_tokens // worker.block_size)
        group_block_hashes = get_block_hashes(request.block_hashes, worker.block_size, worker.hash_block_size)
        request.save_key_block_offset = start_block
        request.save_block_keys = [None] * max(0, end_block - start_block)
        key_slots: list[tuple[str, int | None, int]] = []
        for block_index in range(start_block, min(end_block, len(group_block_hashes))):
            key = worker._make_mooncake_layerwise_key(block_hash_to_str(group_block_hashes[block_index]))
            request.save_block_keys[block_index - start_block] = key
            key_slots.append((key, block_index - start_block, block_index))

        if request.partial_block_index is not None:
            request.save_last_block_key = worker._make_mooncake_layerwise_key(f"{request.req_id}_lastblock")
            key_slots.append((request.save_last_block_key, None, request.partial_block_index))

        requested_keys = list(dict.fromkeys(key for key, _, _ in key_slots))
        if not requested_keys:
            return
        with worker._put_started_keys_lock:
            previously_started = set(requested_keys) & worker._put_started_keys
            new_keys = [key for key in requested_keys if key not in worker._put_started_keys]

        started = set(previously_started)
        if new_keys:
            try:
                results = worker._start_layerwise_put_keys(new_keys, worker._layerwise_object_size_bytes())
            except Exception as exc:
                logger.error("Layerwise batch_put_start failed keys=%s error=%s", new_keys, exc)
                with worker._put_started_keys_lock:
                    worker._put_started_keys.update(new_keys)
                worker._queue_layerwise_revoke_keys(new_keys)
            else:
                newly_started = {key for key, result in zip(new_keys, results, strict=True) if result == 0}
                with worker._put_started_keys_lock:
                    worker._put_started_keys.update(newly_started)
                started.update(newly_started)

        for key, slot, _ in key_slots:
            if key in started:
                if slot is None:
                    request.save_last_block_key = key
            elif slot is None:
                request.save_last_block_key = None
            else:
                request.save_block_keys[slot] = None
        worker._layerwise_session_tracker.register_put_keys(
            request.req_id,
            ((key, block_index) for key, _, block_index in key_slots if key in started),
        )

    def _prepare_single_get_session(self, request: ReqMeta) -> list[tuple[str, int, int | None]]:
        worker = self._worker
        request.load_block_keys = []
        request.load_last_block_key = None
        request.load_keys = []
        current_entries: list[tuple[str, int]] = []
        load_spec = request.load_spec
        if load_spec is not None and load_spec.can_load:
            cached_tokens = load_spec.kvpool_cached_tokens
            if not worker.use_eagle and load_spec.kvpool_store_skip_tokens is not None:
                cached_tokens = load_spec.kvpool_store_skip_tokens
            start_block = load_spec.vllm_cached_tokens // worker.block_size
            cached_full_blocks = cached_tokens // worker.block_size
            group_block_hashes = get_block_hashes(request.block_hashes, worker.block_size, worker.hash_block_size)
            end_block = min(cached_full_blocks, len(group_block_hashes))
            for block_index in range(start_block, end_block):
                current_entries.append(
                    (
                        worker._make_mooncake_layerwise_key(block_hash_to_str(group_block_hashes[block_index])),
                        block_index,
                    )
                )
            needs_last_block = cached_tokens % worker.block_size != 0 or (
                cached_tokens > 0 and end_block < cached_full_blocks
            )
            partial_block_index = cached_full_blocks if cached_tokens % worker.block_size else cached_full_blocks - 1
            if needs_last_block and 0 <= partial_block_index < len(request.block_ids):
                current_entries.append(
                    (
                        worker._make_mooncake_layerwise_key(f"{request.req_id}_lastblock"),
                        partial_block_index,
                    )
                )

        load_entries = worker._layerwise_session_tracker.prepare_load_entries(request.req_id, current_entries)
        valid_entries = [
            (key, block_index) for key, block_index in load_entries if 0 <= block_index < len(request.block_ids)
        ]
        if not valid_entries:
            return []
        request.load_key_block_offset = 0
        request.load_block_keys = [None] * (max(block_index for _, block_index in valid_entries) + 1)
        key_slots: list[tuple[str, int, int | None]] = []
        for key, block_index in valid_entries:
            request.load_block_keys[block_index] = key
            key_slots.append((key, request.block_ids[block_index], block_index))
        return key_slots

    # --- hybrid (multi-group) protocol ---------------------------------------------

    def _prepare_hybrid_sessions(self, requests: list[ReqMeta]) -> dict[int, list[ReqMeta]]:
        worker = self._worker
        worker._layer_load_aborted.clear()
        with worker._put_started_keys_lock:
            previously_started = worker._put_started_keys.copy()
        try:
            return self._prepare_group_sessions(requests)
        except Exception:
            # Preparation has not submitted payload transfers yet. Roll back other
            # groups too if a later group's session allocation fails.
            worker._layer_load_aborted.set()
            with worker._put_started_keys_lock:
                keys = list(worker._put_started_keys - previously_started)
            worker._queue_layerwise_revoke_keys(keys)
            worker._finish_current_layerwise_load_sessions()
            raise

    def _prepare_group_sessions(self, requests: list[ReqMeta]) -> dict[int, list[ReqMeta]]:
        """Views isolate per-group key slots while preserving real request ownership.

        Hybrid pooling publishes only complete blocks. Partial compressor/SWA state
        is not a portable snapshot: the shared coordinator supplies aligned extents
        and masks for the reachable state at each boundary.
        """
        worker = self._worker
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
