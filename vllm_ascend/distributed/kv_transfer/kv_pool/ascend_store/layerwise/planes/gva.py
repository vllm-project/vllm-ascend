# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GVA plane: the pool stores data behind worker-allocated virtual addresses."""

from __future__ import annotations

import time
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
            if self._alloc_gvas_for_save_for_request([request]) is False:
                logger.error("Memcache save preparation failed for %s", request.req_id)
                worker._save_disabled.set()
        if worker._save_disabled.is_set():
            # batch_alloc may return a shared allocation. Stop this step's
            # saves without deleting keys that another writer may have published.
            for request in requests:
                if worker.layerwise_offload and request.can_save:
                    worker._record_failed_request(request)
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
                if group_id not in worker.cacheable_group_ids or not worker._is_layerwise_save_owner(group_id):
                    all_group_gvas.append(np.zeros(len(block_ids_by_group), dtype=np.int64))
                    all_group_block_ids.append(np.asarray(block_ids_by_group, dtype=np.int64))
                    continue
                effective_block_size = worker.grouped_block_size[group_id]
                alloc_size = worker._global_group_alloc_size(group_id)
                group_block_hashes = get_block_hashes(block_hashes, effective_block_size, worker.hash_block_size)

                save_start_block = request.save_start_token // effective_block_size
                partial_block_index = get_partial_block_index(
                    request.target_token_len, effective_block_size, len(group_block_hashes), worker.layerwise_offload
                )
                save_end_block = worker._layerwise_save_full_end(
                    request, group_id, effective_block_size, partial_block_index
                )
                if request.load_spec is not None and not request.load_spec.layerwise_previous_step:
                    save_start_block = request.load_spec.clamp_save_start(save_start_block, effective_block_size)
                group_store_mask = (
                    request.store_masks[group_id]
                    if request.store_masks is not None and group_id < len(request.store_masks)
                    else None
                )
                end_limit = min(save_end_block, len(block_ids_by_group))
                if not (
                    worker.layerwise_offload
                    and worker.num_kv_cache_groups == 1
                    and not worker.group_uses_align_state[group_id]
                ):
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

                partial_block_index = get_partial_block_index(
                    request.target_token_len,
                    effective_block_size,
                    len(group_block_hashes),
                    worker.layerwise_offload,
                )
                if partial_block_index is not None and partial_block_index < len(block_ids_by_group):
                    partial_key = worker._make_layerwise_partial_key(
                        request,
                        group_id,
                        partial_block_index,
                        request.target_token_len,
                    )
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
        worker = self._worker
        for request in requests:
            if self._prepare_load_gvas_for_request([request]) is False:
                logger.error("Memcache load preparation failed for %s", request.req_id)
                worker._record_failed_request(request)
                if request.load_spec is not None:
                    request.load_spec.can_load = False

    def _prepare_load_gvas_for_request(self, requests: list[ReqMeta]) -> bool | None:
        """Fetch per-rank GVA and acquire read lease for the load path.

        memcache requires batch_copy (read) to find the blob in the per-process
        gvaBlobTracker with a valid lease. The scheduler only checks existence
        (batch_is_exist) to decide the load range; before batch_copy(G2L) the
        worker must, for its own per-rank keys:
          1. Rewarm SSD-only keys through the backend, then query a resident GVA.
          2. Acquire a read lease and refresh the GVA while its allocation is pinned.
        """
        worker = self._worker
        if not worker.use_layerwise_transfer:
            return None
        for request in requests:
            if request.load_spec is None or not request.load_spec.can_load:
                continue
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

            all_group_load_gvas: list[np.ndarray] = []
            all_group_load_keys: list[str] = []
            request.partial_load_gva_per_group = [0] * worker.num_kv_cache_groups
            for group_id in range(worker.num_kv_cache_groups):
                if group_id not in worker.cacheable_group_ids:
                    all_group_load_gvas.append(np.zeros(0, dtype=np.int64))
                    continue
                group_block_size = worker.grouped_block_size[group_id]
                effective_block_size = group_block_size

                group_block_hashes = get_block_hashes(block_hashes, effective_block_size, worker.hash_block_size)
                load_start_block = (
                    0 if worker.layerwise_offload else request.load_spec.vllm_cached_tokens // effective_block_size
                )
                cached_full_blocks = cached_tokens // effective_block_size
                full_blocks = min(cached_full_blocks, len(group_block_hashes))
                snapshot_start = worker._layerwise_load_snapshot_start(request, group_id, cached_tokens)
                if snapshot_start is not None:
                    full_blocks = cached_full_blocks

                block_ids_by_group = (
                    request.block_ids_by_group_np[group_id]
                    if (request.block_ids_by_group_np is not None and group_id < len(request.block_ids_by_group_np))
                    else request.block_ids_np
                )
                if block_ids_by_group is None:
                    all_group_load_gvas.append(np.zeros(0, dtype=np.int64))
                    continue
                full_len = len(block_ids_by_group)

                partial_block_index, partial_end_token = worker._resolve_layerwise_load_partial(
                    request,
                    group_id,
                    cached_tokens,
                    effective_block_size,
                    len(group_block_hashes),
                )
                if partial_block_index is not None and (
                    partial_block_index < load_start_block or partial_block_index >= full_len
                ):
                    partial_block_index = None

                if load_start_block >= full_blocks and partial_block_index is None:
                    all_group_load_gvas.append(np.zeros(full_len, dtype=np.int64))
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
                keys = []
                for block_idx in block_indices:
                    if snapshot_start is None or block_idx < snapshot_start:
                        key = worker._make_layerwise_full_key(
                            group_id, block_hash_to_str(group_block_hashes[block_idx])
                        )
                    else:
                        snapshot_end_token = request.load_spec.layerwise_snapshot_end_token
                        assert snapshot_end_token is not None
                        key = worker._make_layerwise_partial_key(request, group_id, block_idx, snapshot_end_token)
                    keys.append(key)
                has_partial_key = False
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
                    return False
                if partial_block_index is not None:
                    keys.append(
                        worker._make_layerwise_partial_key(
                            request,
                            group_id,
                            partial_block_index,
                            partial_end_token,
                        )
                    )
                    block_indices.append(partial_block_index)
                    has_partial_key = True
                if not keys:
                    all_group_load_gvas.append(np.zeros(full_len, dtype=np.int64))
                    continue

                key_infos = worker.m_store.batch_get_key_info(keys, for_load=True)
                if key_infos is None or len(key_infos) != len(keys):
                    return False
                gvas = []
                valid_gva_indices = []
                invalid_block_ids: list[int] = []
                for ki, key, block_idx in zip(key_infos, keys, block_indices):
                    # A None entry means the key is absent from the store;
                    # treat it like an unreadable block (gva=0) below.
                    sizes = ki.size() if ki is not None else 0
                    gva = next((address for address in ki.gva_list() if address > 0), 0) if sizes and sizes > 0 else 0
                    gvas.append(gva)
                    if gva > 0:
                        valid_gva_indices.append(len(gvas) - 1)
                    else:
                        if block_idx < len(block_ids_by_group):
                            invalid_block_ids.append(int(block_ids_by_group[block_idx]))
                        logger.warning(
                            "load_gvas: req=%s group=%d got invalid gva=%d (size=%d), block_id=%s load failed",
                            request.req_id,
                            group_id,
                            gva,
                            sizes if sizes else 0,
                            int(block_ids_by_group[block_idx]) if block_idx < len(block_ids_by_group) else "N/A",
                        )

                # Only call batch_add_lease for keys with valid size.
                valid_keys = [keys[index] for index in valid_gva_indices]
                if valid_keys:
                    # Multiple requests in one step can share prefix keys. One
                    # lease pins each key for the whole step, so acquire it only
                    # once to match the set-based release at the step fence.
                    new_keys = list(dict.fromkeys(key for key in valid_keys if key not in worker._step_load_keys))
                    new_results = (
                        worker.m_store.batch_add_lease(new_keys, LAYERWISE_READ_LEASE_TTL_MS) if new_keys else []
                    )
                    if new_results is None or len(new_results) != len(new_keys):
                        return False
                    result_by_key = dict(zip(new_keys, new_results))
                    lease_results = [0 if key in worker._step_load_keys else result_by_key[key] for key in valid_keys]
                    for key, res in zip(new_keys, new_results):
                        if res == 0:
                            worker._step_load_keys.add(key)
                    leased_keys = []
                    for gva_index, lease_res in zip(valid_gva_indices, lease_results):
                        block_idx = block_indices[gva_index]
                        if lease_res == MEMCACHE_UNMATCHED_STATE and block_idx == partial_block_index:
                            partial_key = keys[gva_index]
                            for retry in range(1, PARTIAL_LEASE_RETRY_COUNT + 1):
                                time.sleep(PARTIAL_LEASE_RETRY_INTERVAL_S)
                                retry_results = worker.m_store.batch_add_lease(
                                    [partial_key],
                                    LAYERWISE_READ_LEASE_TTL_MS,
                                )
                                if retry_results is None or len(retry_results) != 1:
                                    return False
                                lease_res = retry_results[0]
                                if lease_res == 0:
                                    worker._step_load_keys.add(partial_key)
                                if lease_res != MEMCACHE_UNMATCHED_STATE:
                                    break
                        block_id = int(block_ids_by_group[block_idx]) if block_idx < len(block_ids_by_group) else None
                        if lease_res == 0:
                            leased_keys.append(keys[gva_index])
                        else:
                            gvas[gva_index] = 0
                            if block_id is not None:
                                invalid_block_ids.append(block_id)
                            logger.warning(
                                "load_gvas: req=%s group=%d lease failed result=%d, block_id=%s load failed",
                                request.req_id,
                                group_id,
                                lease_res,
                                block_id,
                            )
                else:
                    lease_results = []
                    leased_keys = []

                # Query under the acquired lease: rewarm can leave the SSD
                # descriptor first, and its GVA is zero. The lease pins the
                # current DRAM allocation until all layer copies finish.
                if leased_keys:
                    leased_infos = worker.m_store.batch_get_key_info(leased_keys)
                    if leased_infos is None or len(leased_infos) != len(leased_keys):
                        return False
                    key_indices: dict[str, list[int]] = {}
                    for index, key in enumerate(keys):
                        key_indices.setdefault(key, []).append(index)
                    pinned_gvas: dict[str, int] = {}
                    for key, info in zip(leased_keys, leased_infos, strict=True):
                        pinned_gvas[key] = (
                            next((address for address in info.gva_list() if address > 0), 0) if info is not None else 0
                        )
                    for key, pinned_gva in pinned_gvas.items():
                        for index in key_indices[key]:
                            gvas[index] = pinned_gva
                            if not pinned_gva:
                                invalid_block_ids.append(int(block_ids_by_group[block_indices[index]]))

                if invalid_block_ids:
                    return False
                all_group_load_keys.extend(leased_keys)

                logger.debug(
                    "load_gvas: req=%s group=%d eff_bs=%d load_blocks=[%d,%d) keys=%d valid_gvas=%d lease_fail=%d",
                    request.req_id,
                    group_id,
                    effective_block_size,
                    load_start_block,
                    full_blocks,
                    len(keys),
                    sum(1 for g in gvas if g > 0),
                    sum(1 for r in lease_results if r != 0),
                )

                # Pad to match block_ids_by_group length, filling only the
                # positions whose keys were actually queried (the trailing
                # partial key is excluded from the per-position fill).
                full_gvas = [0] * full_len
                full_key_count = len(block_indices) - (1 if has_partial_key else 0)
                for gva_index in range(full_key_count):
                    block_idx = block_indices[gva_index]
                    if block_idx < len(full_gvas):
                        full_gvas[block_idx] = gvas[gva_index]
                all_group_load_gvas.append(np.asarray(full_gvas, dtype=np.int64))
                if has_partial_key and gvas:
                    request.partial_load_gva_per_group[group_id] = gvas[-1]

            if all_group_load_gvas:
                request.load_keys = all_group_load_keys
                request.load_block_gvas_by_group_np = all_group_load_gvas
                request.load_block_gvas_np = all_group_load_gvas[0]
                request.load_gva_block_offset = 0
        return None
