# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GVA plane: the pool stores data behind worker-allocated virtual addresses."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import (
    get_partial_block_index,
)
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

# Read lease TTL (ms) for the layerwise load path. batch_add_lease acquires a
# read lease before batch_copy(G2L); the lease must cover the asynchronous
# multi-layer load time.
LAYERWISE_READ_LEASE_TTL_MS = 5 * 60 * 1000
MEMCACHE_SSD_HIT_GVA = (1 << 64) - 1

# A partial snapshot can be visible to readers before the rank responsible for
# saving it has published its final layer.
MEMCACHE_UNMATCHED_STATE = -3101
PARTIAL_LEASE_RETRY_COUNT = 10
PARTIAL_LEASE_RETRY_INTERVAL_S = 0.001


@dataclass
class _LoadGroup:
    gvas: np.ndarray
    block_indices: list[int]
    keys: list[str]
    has_partial: bool
    snapshot_fallbacks: dict[int, str]


_LoadPlan = tuple[ReqMeta, list[_LoadGroup], dict[str, None]]


class GvaDataPlane(LayerwiseDataPlane):
    """GVA-addressed pooling: allocate write targets and lease readable GVAs.

    The save path allocates per-group GVAs plus the request-scoped trailing
    partial block; the load path resolves resident GVAs, rewarm-queries
    SSD-only keys, and acquires read leases before ``batch_copy(G2L)``.
    Shared machinery (store handle, key construction, the save-disable
    circuit breaker, failure recording) stays on the worker and is reached
    through ``self._worker``.
    """

    def __init__(self, worker: LayerwiseKVPoolWorker):
        self._worker = worker

    def prepare_sessions(self, requests: list[ReqMeta]) -> dict[int, list[ReqMeta]]:
        """GVA opens no pool sessions; resolve the transfer targets instead.

        Protect the previous partial before allocating the next snapshot:
        load preparation queries the pool-registered GVA of the previous
        step's snapshot, so it must complete before save allocation
        repoints the partial key at a fresh one.
        """
        self._prepare_load_gvas(requests)
        self._alloc_gvas_for_save(requests)
        return {}

    def finish_sessions(self) -> None:
        """GVA loads read through worker-held addresses, not pool sessions."""

    # --- save path -----------------------------------------------------------

    def _alloc_gvas_for_save(self, requests: list[ReqMeta]) -> None:
        worker = self._worker
        if not worker.use_layerwise_transfer:
            return
        if worker.kv_role == "kv_consumer" and not worker.consumer_is_to_put:
            return
        for request in requests:
            if worker._save_disabled.is_set():
                break
            if not request.can_save:
                continue
            if worker.layerwise_offload and (
                request.load_spec is None or not request.load_spec.layerwise_previous_step
            ):
                worker._snapshot_keys.pop(request.req_id, None)
            if self._alloc_gvas_for_save_for_request([request]) is False:
                logger.error("Memcache save preparation failed for %s", request.req_id)
                worker._save_disabled.set()
        if worker._save_disabled.is_set():
            # batch_alloc may return a shared allocation. Stop this step's
            # saves without deleting keys that another writer may have published.
            for request in requests:
                if worker.layerwise_offload and request.can_save:
                    worker._record_failed_request(request)
                    worker._snapshot_keys.pop(request.req_id, None)
                request.can_save = False
                request.save_keys = []

    def _alloc_gvas_for_save_for_request(self, requests: list[ReqMeta]) -> bool | None:
        """Allocate per-group GVA on the worker side right before batch_copy.

        For multi-group models, iterates all KV cache groups and allocates
        per-group GVAs. Key format: model@group_id@hash@head_or_tp_rank
        (multi-group) or model@hash@head_or_tp_rank (single-group, backward
        compat with PR #11585).

        batch_alloc is idempotent, so every write target is allocated fresh
        within the step and no cross-step GVA cache is kept. Blocks already
        committed in the pool are filtered out with a medium-agnostic
        exists check to avoid re-transferring them.
        """
        worker = self._worker
        if not worker.use_layerwise_transfer:
            return None
        if not worker.can_put:
            return None
        for request in requests:
            if request.can_save is None or not request.can_save:
                continue
            block_hashes = request.block_hashes

            all_group_gvas: list[np.ndarray] = []
            all_group_block_ids: list[np.ndarray] = []
            all_group_save_keys: list[str] = []
            request.partial_save_gva_per_group = [0] * worker.num_kv_cache_groups
            for group_id in range(worker.num_kv_cache_groups):
                block_ids_by_group = (
                    request.block_ids_by_group_np[group_id]
                    if (request.block_ids_by_group_np is not None and group_id < len(request.block_ids_by_group_np))
                    else request.block_ids_np
                )
                if block_ids_by_group is None:
                    raise RuntimeError(f"Block IDs are not initialized for request {request.req_id}")
                save_owner = worker._is_layerwise_save_owner(group_id)
                if group_id not in worker.cacheable_group_ids or (not save_owner and not worker.layerwise_offload):
                    all_group_gvas.append(np.zeros(len(block_ids_by_group), dtype=np.int64))
                    all_group_block_ids.append(np.asarray(block_ids_by_group, dtype=np.int64))
                    continue
                effective_block_size = worker.grouped_block_size[group_id]
                if (
                    worker.layerwise_offload
                    and request.target_token_len > 0
                    and worker.group_uses_align_state[group_id]
                    and group_id in worker._reused_layers_by_group
                ):
                    endpoint_block = (request.target_token_len - 1) // effective_block_size
                    if endpoint_block >= len(block_ids_by_group) or block_ids_by_group[endpoint_block] == 0:
                        raise ValueError("Recurrent cache endpoint has no allocated state page")
                alloc_size = worker._global_group_alloc_size(group_id)
                group_block_hashes = get_block_hashes(block_hashes, effective_block_size, worker.hash_block_size)

                save_start_block = request.save_start_token // effective_block_size
                partial_block_index = get_partial_block_index(
                    request.target_token_len, effective_block_size, len(group_block_hashes), worker.layerwise_offload
                )
                save_end_block = worker._layerwise_save_full_end(request, effective_block_size, partial_block_index)
                if request.load_spec is not None and not request.load_spec.layerwise_previous_step:
                    save_start_block = request.load_spec.clamp_save_start(save_start_block, effective_block_size)
                group_store_mask = (
                    request.store_masks[group_id]
                    if request.store_masks is not None and group_id < len(request.store_masks)
                    else None
                )
                end_limit = min(save_end_block, len(block_ids_by_group))
                if not worker.layerwise_offload or worker.group_uses_align_state[group_id]:
                    end_limit = min(end_limit, len(group_block_hashes))
                candidate_blocks = [
                    block_idx
                    for block_idx in range(save_start_block, end_limit)
                    if group_store_mask is None or block_idx >= len(group_store_mask) or group_store_mask[block_idx]
                ]
                candidate_keys = [
                    worker._make_layerwise_full_key(
                        group_id,
                        block_hash_to_str(group_block_hashes[block_idx]),
                    )
                    if block_idx < len(group_block_hashes)
                    else worker._make_layerwise_partial_key(request, group_id, block_idx, request.target_token_len)
                    for block_idx in candidate_blocks
                ]
                partial_key = None
                if (
                    worker.layerwise_offload
                    and partial_block_index is not None
                    and partial_block_index < len(block_ids_by_group)
                ):
                    partial_key = worker._make_layerwise_partial_key(
                        request, group_id, partial_block_index, request.target_token_len
                    )
                if worker.layerwise_offload:
                    pages = worker._snapshot_keys.setdefault(request.req_id, {}).setdefault(group_id, {})
                    for block_idx, key in zip(candidate_blocks, candidate_keys):
                        if block_idx < len(group_block_hashes):
                            pages.pop(block_idx, None)
                        else:
                            pages[block_idx] = key
                    if partial_key is not None:
                        pages[partial_block_index] = partial_key
                if not save_owner:
                    all_group_gvas.append(np.zeros(len(block_ids_by_group), dtype=np.int64))
                    all_group_block_ids.append(np.asarray(block_ids_by_group, dtype=np.int64))
                    continue
                # Skip blocks already committed in the pool. The exists check
                # is medium-agnostic on purpose: a committed block (DRAM or
                # SSD) does not need re-saving, and the load path rewarms
                # SSD blocks on read.
                missing_keys = worker._pool_keys_missing(candidate_keys)
                transfer_blocks = [
                    block_idx for block_idx, key in zip(candidate_blocks, candidate_keys) if key in missing_keys
                ]

                # batch_alloc is idempotent: allocating an existing key
                # returns its current, client-registered GVA. Every write
                # target is therefore allocated fresh within this step; no
                # cross-step GVA cache is kept.
                block_gvas: list[int] = []
                if transfer_blocks:
                    transfer_keys = [key for key in candidate_keys if key in missing_keys]
                    transfer_gvas = worker.m_store.batch_alloc(
                        transfer_keys, [alloc_size] * len(transfer_keys), LAYERWISE_READ_LEASE_TTL_MS
                    )
                    if transfer_gvas is None or len(transfer_gvas) != len(transfer_keys):
                        return False
                    if any(gva <= 0 for gva in transfer_gvas):
                        logger.error(
                            "alloc_gvas FAIL: req=%s group=%d alloc_size=%d alloc_keys=%d gvas_sample=%s fail_count=%d",
                            request.req_id,
                            group_id,
                            alloc_size,
                            len(transfer_keys),
                            transfer_gvas[:5],
                            sum(1 for g in transfer_gvas if g <= 0),
                        )
                    # 0 means allocation failed; UINT64_MAX means a readable
                    # SSD hit. Filter both before converting GVAs to np.int64.
                    block_gvas = [gva if 0 < gva < MEMCACHE_SSD_HIT_GVA else 0 for gva in transfer_gvas]
                    for key, gva in zip(transfer_keys, block_gvas):
                        if gva > 0:
                            all_group_save_keys.append(key)
                    if worker.layerwise_offload and any(gva <= 0 for gva in transfer_gvas):
                        return False

                if partial_key is not None:
                    # Partial keys are request-scoped: allocate fresh each step.
                    allocated = worker.m_store.batch_alloc(
                        [partial_key],
                        [alloc_size],
                        LAYERWISE_READ_LEASE_TTL_MS,
                    )
                    if not allocated or len(allocated) != 1:
                        return False
                    partial_gva = allocated[0]
                    if 0 < partial_gva < MEMCACHE_SSD_HIT_GVA:
                        all_group_save_keys.append(partial_key)
                    elif partial_gva <= 0:
                        logger.error(
                            "alloc_gvas: partial allocation failed req=%s group=%d block=%d gva=%d",
                            request.req_id,
                            group_id,
                            partial_block_index,
                            partial_gva,
                        )
                    if worker.layerwise_offload and partial_gva <= 0:
                        return False
                    request.partial_save_gva_per_group[group_id] = (
                        partial_gva if 0 < partial_gva < MEMCACHE_SSD_HIT_GVA else 0
                    )

                logger.debug(
                    "alloc_gvas: req=%s group=%d eff_bs=%d save_blocks=[%d,%d) "
                    "transfer_keys=%d committed=%d masked=%s alloc_size=%d",
                    request.req_id,
                    group_id,
                    effective_block_size,
                    save_start_block,
                    save_end_block,
                    len(transfer_blocks),
                    len(candidate_blocks) - len(transfer_blocks),
                    group_store_mask is not None,
                    alloc_size,
                )

                # Pad block_gvas to match block_ids length (fill 0 for non-transferred blocks)
                full_gvas = [0] * len(block_ids_by_group)
                for blk_idx, gva in zip(transfer_blocks, block_gvas):
                    if blk_idx < len(full_gvas):
                        full_gvas[blk_idx] = gva

                all_group_gvas.append(np.asarray(full_gvas, dtype=np.int64))
                all_group_block_ids.append(np.asarray(block_ids_by_group, dtype=np.int64))

            if all_group_gvas:
                request.save_keys = all_group_save_keys
                request.block_gvas_by_group_np = all_group_gvas
                request.block_ids_by_group_np = all_group_block_ids
                request.block_gvas_np = all_group_gvas[0]
                request.gva_block_offset = 0
        return None

    # --- load path -----------------------------------------------------------

    def _prepare_load_gvas(self, requests: list[ReqMeta]) -> None:
        if not self._worker.use_layerwise_transfer:
            return

        plans: list[_LoadPlan] = []
        for request in requests:
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            plan = self._plan_load_gvas_for_request(request)
            if plan is None:
                self._fail_load_preparation(request)
                continue
            plans.append((request, *plan))

        for (request, groups, _), addresses in zip(plans, self._lease_step_load_gvas(plans), strict=True):
            if addresses is None:
                self._fail_load_preparation(request)
                continue
            partial_gvas = []
            for group, values in zip(groups, addresses, strict=True):
                if group.has_partial:
                    group.gvas[group.block_indices[:-1]] = values[:-1]
                    partial_gvas.append(values[-1])
                else:
                    group.gvas[group.block_indices] = values
                    partial_gvas.append(0)
            if groups:
                request.load_keys = [key for group in groups for key in group.keys]
                request.load_block_gvas_by_group_np = [group.gvas for group in groups]
                request.load_block_gvas_np = groups[0].gvas
                request.partial_load_gva_per_group = partial_gvas
                request.load_gva_block_offset = 0

    def _fail_load_preparation(self, request: ReqMeta) -> None:
        logger.error("Memcache load preparation failed for %s", request.req_id)
        self._worker._record_failed_request(request)
        assert request.load_spec is not None
        request.load_spec.can_load = False

    def _lease_step_load_gvas(self, plans: list[_LoadPlan]) -> Iterable[list[list[int]] | None]:
        public_keys: dict[str, None] = {}
        for _, _, keys in plans:
            public_keys.update(keys)

        public_gvas: dict[str, int] = {}
        missing_keys: set[str] = set()
        for keys in self._worker._layerwise_key_batches(list(public_keys)):
            addresses = self._lease_load_gvas(keys, missing_keys=missing_keys)
            if addresses is not None:
                public_gvas.update((key, gva) for key, gva in zip(keys, addresses, strict=True) if gva)

        for _, groups, request_public_keys in plans:
            private_keys: dict[str, None] = {}
            unavailable = False
            for group in groups:
                for index, key in enumerate(group.keys):
                    if key not in request_public_keys:
                        private_keys[key] = None
                    elif key not in public_gvas:
                        fallback = group.snapshot_fallbacks.get(index) if key in missing_keys else None
                        if fallback is None:
                            unavailable = True
                            break
                        group.keys[index] = fallback
                        private_keys[fallback] = None
                if unavailable:
                    break
            if unavailable:
                yield None
                continue

            private_gvas: dict[str, int] = {}
            for keys in self._worker._layerwise_key_batches(list(private_keys)):
                addresses = self._lease_load_gvas(keys, retry_private=True)
                if addresses is not None:
                    private_gvas.update((key, gva) for key, gva in zip(keys, addresses, strict=True) if gva)
            if len(private_gvas) != len(private_keys):
                yield None
                continue
            yield [
                [public_gvas[key] if key in public_gvas else private_gvas[key] for key in group.keys]
                for group in groups
            ]

    def _lease_load_gvas(
        self, keys: list[str], *, retry_private: bool = False, missing_keys: set[str] | None = None
    ) -> list[int] | None:
        worker = self._worker
        infos = worker.m_store.batch_get_key_info(keys, for_load=True)
        if infos is None or len(infos) != len(keys):
            return None
        readable_indices = []
        for index, (key, info) in enumerate(zip(keys, infos, strict=True)):
            size = info.size() if info is not None else 0
            if size and size > 0 and any(address > 0 for address in info.gva_list()):
                readable_indices.append(index)
            else:
                if info is None and missing_keys is not None:
                    missing_keys.add(key)
                logger.warning("load_gvas: key=%s has no readable GVA", key)

        gvas = [0] * len(keys)
        readable_keys = [keys[index] for index in readable_indices]
        new_keys = [key for key in readable_keys if key not in worker._step_load_keys]
        results = worker.m_store.batch_add_lease(new_keys, LAYERWISE_READ_LEASE_TTL_MS) if new_keys else []
        if results is None or len(results) != len(new_keys):
            return None
        result_by_key = dict(zip(new_keys, results, strict=True))
        worker._step_load_keys.update(key for key, status in result_by_key.items() if status == 0)

        leased_indices = []
        for index, key in zip(readable_indices, readable_keys, strict=True):
            if key in worker._step_load_keys:
                leased_indices.append(index)
                continue
            status = result_by_key[key]
            if retry_private and status == MEMCACHE_UNMATCHED_STATE:
                for _ in range(PARTIAL_LEASE_RETRY_COUNT):
                    time.sleep(PARTIAL_LEASE_RETRY_INTERVAL_S)
                    retry = worker.m_store.batch_add_lease([key], LAYERWISE_READ_LEASE_TTL_MS)
                    if retry is None or len(retry) != 1:
                        return None
                    status = retry[0]
                    if status != MEMCACHE_UNMATCHED_STATE:
                        break
                if status == 0:
                    worker._step_load_keys.add(key)
                    leased_indices.append(index)
                    continue
            logger.warning("load_gvas: key=%s lease failed result=%d", key, status)

        # Refresh only after each lease pins the current allocation.
        if leased_indices:
            infos = worker.m_store.batch_get_key_info([keys[index] for index in leased_indices])
            if infos is None or len(infos) != len(leased_indices):
                return None
            for index, info in zip(leased_indices, infos, strict=True):
                if info is not None:
                    gvas[index] = next((address for address in info.gva_list() if address > 0), 0)
        return gvas

    def _plan_load_gvas_for_request(self, request: ReqMeta) -> tuple[list[_LoadGroup], dict[str, None]] | None:
        worker = self._worker
        assert request.load_spec is not None
        cached_tokens = request.load_spec.kvpool_cached_tokens
        if not getattr(worker, "use_eagle", False):
            cached_tokens = request.load_spec.pool_hit_tokens
        if (
            getattr(worker, "use_eagle", False)
            and request.load_spec.kvpool_cached_tokens == request.target_token_len - 1
        ):
            # Full-hit path: the trailing block is recomputed and will be
            # re-stored by the normal save path, so never skip it here.
            logger.debug(
                "Reqid: %s full-hit tail recompute path, tail block will be re-stored",
                request.req_id,
            )
        block_hashes = request.block_hashes
        request.load_masks = worker._compute_reachable_load_masks(request, cached_tokens)

        groups: list[_LoadGroup] = []
        public_keys: dict[str, None] = {}
        restore_previous_step = worker.layerwise_offload and request.load_spec.layerwise_previous_step
        for group_id in range(worker.num_kv_cache_groups):
            if group_id not in worker.cacheable_group_ids:
                groups.append(_LoadGroup(np.zeros(0, dtype=np.int64), [], [], False, {}))
                continue
            group_block_size = worker.grouped_block_size[group_id]
            effective_block_size = group_block_size

            group_block_hashes = get_block_hashes(block_hashes, effective_block_size, worker.hash_block_size)
            load_start_block = (
                0 if worker.layerwise_offload else request.load_spec.vllm_cached_tokens // effective_block_size
            )
            cached_full_blocks = cached_tokens // effective_block_size
            full_blocks = min(cached_full_blocks, len(group_block_hashes))
            get_saved_key = None
            partial_block_index = None
            if restore_previous_step:
                snapshot = worker._get_layerwise_load_snapshot(request, group_id, cached_tokens, group_block_hashes)
                if snapshot is None:
                    groups.append(_LoadGroup(np.zeros(0, dtype=np.int64), [], [], False, {}))
                    continue
                full_blocks, get_saved_key = snapshot
                partial_block_index = worker._resolve_layerwise_load_partial(cached_tokens, effective_block_size)

            block_ids_by_group = (
                request.block_ids_by_group_np[group_id]
                if (request.block_ids_by_group_np is not None and group_id < len(request.block_ids_by_group_np))
                else request.block_ids_np
            )
            if (
                worker.layerwise_offload
                and cached_tokens > 0
                and worker.group_uses_align_state[group_id]
                and group_id in worker._reused_layers_by_group
            ):
                endpoint_block = (cached_tokens - 1) // effective_block_size
                if (
                    block_ids_by_group is None
                    or endpoint_block >= len(block_ids_by_group)
                    or block_ids_by_group[endpoint_block] == 0
                ):
                    raise ValueError("Recurrent cache endpoint has no allocated state page")
            if block_ids_by_group is None:
                if get_saved_key is not None:
                    return None
                groups.append(_LoadGroup(np.zeros(0, dtype=np.int64), [], [], False, {}))
                continue
            full_len = len(block_ids_by_group)

            if partial_block_index is not None and (
                partial_block_index < load_start_block or partial_block_index >= full_len
            ):
                return None

            if load_start_block >= full_blocks and partial_block_index is None:
                groups.append(_LoadGroup(np.zeros(full_len, dtype=np.int64), [], [], False, {}))
                continue

            group_load_mask = (
                request.load_masks[group_id]
                if request.load_masks is not None and group_id < len(request.load_masks)
                else None
            )
            block_indices = [
                block_idx
                for block_idx in range(load_start_block, full_blocks)
                if block_idx != partial_block_index
                and (group_load_mask is None or block_idx >= len(group_load_mask) or group_load_mask[block_idx])
            ]
            if any(block_idx >= full_len for block_idx in block_indices):
                logger.error(
                    "Memcache load range exceeds allocated blocks: req=%s group=%d "
                    "load_blocks=[%d,%d) allocated_blocks=%d",
                    request.req_id,
                    group_id,
                    load_start_block,
                    full_blocks,
                    full_len,
                )
                return None

            keys = []
            snapshot_fallbacks: dict[int, str] = {}
            private_snapshot = (
                worker._snapshot_keys.get(request.req_id, {}).get(group_id, {}) if restore_previous_step else {}
            )
            if partial_block_index is not None:
                block_indices.append(partial_block_index)
            for block_idx in block_indices:
                if get_saved_key is not None:
                    saved_key = get_saved_key(block_idx)
                    if saved_key is None:
                        return None
                    key, is_public = saved_key
                else:
                    block_hash = block_hash_to_str(group_block_hashes[block_idx])
                    key = worker._make_layerwise_full_key(group_id, block_hash)
                    is_public = True
                if is_public:
                    public_keys[key] = None
                    if block_idx in private_snapshot:
                        snapshot_fallbacks[len(keys)] = private_snapshot[block_idx]
                keys.append(key)
            groups.append(
                _LoadGroup(
                    np.zeros(full_len, dtype=np.int64),
                    block_indices,
                    keys,
                    partial_block_index is not None,
                    snapshot_fallbacks,
                )
            )
        return groups, public_keys
