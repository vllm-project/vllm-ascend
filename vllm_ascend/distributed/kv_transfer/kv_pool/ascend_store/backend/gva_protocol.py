#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Layerwise GVA transfer protocol (memcache backend).

This module centralizes the GVA key/allocation/lease protocol that used to
live inside ``KVPoolWorker`` and ``KVPoolScheduler``. It is a
behavior-preserving relocation: key formats, log messages, and the
prepare-load -> alloc-save ordering are kept byte-for-byte.

Ownership:
- :class:`GVAKeyFactory`: string formats of full/partial/hit-check keys.
- :class:`GVASession`: worker-side allocation, load preparation, and leases.
- :class:`GVAHitChecker`: scheduler-side prefix hit computation over the
  all-rank keys.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ReqMeta,
    block_hash_to_str,
    get_block_hashes,
    get_group_block_size,
    get_partial_block_index,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request

# Read lease TTL (ms) for the layerwise load path. batch_add_lease acquires a
# read lease before batch_copy(G2L); the lease must cover the asynchronous
# multi-layer load time.
LAYERWISE_READ_LEASE_TTL_MS = 5 * 60 * 1000

# A partial snapshot can be visible to readers before the rank responsible for
# saving it has published its final layer.
MEMCACHE_UNMATCHED_STATE = -3101
PARTIAL_LEASE_RETRY_COUNT = 10
PARTIAL_LEASE_RETRY_INTERVAL_S = 0.001


class GVAStore(Protocol):
    """Structural type of the backend surface the GVA protocol needs."""

    def ensure_initialized(self) -> None: ...

    def batch_is_exist(self, keys: list[str]) -> list[int]: ...

    def batch_get_key_info(self, keys: list[str]) -> list[Any]: ...

    def batch_alloc(self, keys: list[str], sizes: list[int]) -> list[int]: ...

    def batch_add_lease(self, keys: list[str], lease_ttl_ms: int = 0) -> list[int]: ...

    def batch_remove_lease(self, keys: list[str]) -> int: ...


class GVAKeyFactory:
    """String formats for the layerwise GVA keys.

    Single-group models use the PR #11585 format (model@hash@rank) for
    backward compatibility. Multi-group models include group_id
    (model@group_id@hash@rank) to distinguish groups.
    """

    @staticmethod
    def full_key(
        model_name: str,
        group_id: int,
        block_hash_hex: str,
        head_or_tp_rank: int,
        num_groups: int,
    ) -> str:
        if num_groups > 1:
            return f"{model_name}@{group_id}@{block_hash_hex}@{head_or_tp_rank}"
        else:
            return f"{model_name}@{block_hash_hex}@{head_or_tp_rank}"

    @staticmethod
    def partial_key(
        model_name: str,
        req_id: str,
        group_id: int,
        block_index: int,
        end_token: int,
        head_or_tp_rank: int,
    ) -> str:
        return f"{model_name}@partial@{req_id}@{group_id}@{block_index}@{end_token}@{head_or_tp_rank}"

    @staticmethod
    def hit_check_keys(
        model_name: str,
        group_id: int,
        block_hash_hex: str,
        num_ranks: int,
        num_groups: int,
    ) -> list[str]:
        """All-rank GVA keys for scheduler-side hit check.

        Returns one key per head_or_tp_rank (ranks in the same put_step
        group share one key for MLA).
        """
        if num_groups > 1:
            return [f"{model_name}@{group_id}@{block_hash_hex}@{h}" for h in range(num_ranks)]
        else:
            return [f"{model_name}@{block_hash_hex}@{h}" for h in range(num_ranks)]


class GVASession:
    """Worker-side GVA lifecycle: allocation for save, preparation for load.

    Constructed only when ``use_gva_layerwise`` is true. Layout-dependent
    parameters (group_block_len / page_size_bytes) arrive later via
    :meth:`bind_layout`, once ``register_kv_caches`` has measured them.

    Invalid-block reporting is injected as the ``on_invalid_blocks``
    callback so the protocol never touches worker state directly.
    """

    def __init__(
        self,
        store: GVAStore,
        model_name: str,
        head_or_tp_rank: int,
        tp_rank: int,
        put_step: int,
        num_kv_cache_groups: int,
        grouped_block_size: list[int],
        hash_block_size: int,
        layerwise_offload: bool,
        use_eagle: bool,
        kv_role: str,
        consumer_is_to_put: bool,
        on_invalid_blocks: Callable[[list[int]], None],
    ) -> None:
        # Self-ensure (LIFE): a session must never run against an
        # uninitialized store, even when the worker skips the eager
        # on_worker_ready path (lazy_init mode relies on this too).
        store.ensure_initialized()
        self._store = store
        self._model_name = model_name
        self._head_or_tp_rank = head_or_tp_rank
        self._tp_rank = tp_rank
        self._put_step = put_step
        self._num_kv_cache_groups = num_kv_cache_groups
        self._grouped_block_size = grouped_block_size
        self._hash_block_size = hash_block_size
        self._layerwise_offload = layerwise_offload
        self._use_eagle = use_eagle
        self._kv_role = kv_role
        self._consumer_is_to_put = consumer_is_to_put
        self._on_invalid_blocks = on_invalid_blocks
        self._group_block_len: dict[int, list[int]] = {}
        self._page_size_bytes = 0
        self._allocated_gvas: dict[str, int] = {}

    def bind_layout(self, group_block_len: dict[int, list[int]], page_size_bytes: int) -> None:
        """Bind the layout-dependent parameters measured by register_kv_caches."""
        self._group_block_len = group_block_len
        self._page_size_bytes = page_size_bytes

    def _make_gva_key(self, group_id: int, block_hash_hex: str) -> str:
        return GVAKeyFactory.full_key(
            self._model_name,
            group_id,
            block_hash_hex,
            self._head_or_tp_rank,
            self._num_kv_cache_groups,
        )

    def _make_partial_key(self, request: ReqMeta, group_id: int, block_index: int, end_token: int) -> str:
        return GVAKeyFactory.partial_key(
            self._model_name,
            request.req_id,
            group_id,
            block_index,
            end_token,
            self._head_or_tp_rank,
        )

    def _refresh_allocated_gvas(self, keys: list[str]) -> None:
        """Drop local GVA entries whose MemCache blobs were evicted."""
        cached_keys = list(dict.fromkeys(key for key in keys if key in self._allocated_gvas))
        if not cached_keys:
            return
        exists_states = self._store.batch_is_exist(cached_keys)
        if len(exists_states) != len(cached_keys):
            raise RuntimeError(
                "MemCache exists check returned unexpected number of states: "
                f"expected={len(cached_keys)}, actual={len(exists_states)}"
            )
        for key, exists in zip(cached_keys, exists_states):
            if exists == 0:
                self._allocated_gvas.pop(key, None)
            elif exists != 1:
                raise RuntimeError(f"MemCache exists check failed for {key}: state={exists}")

    def alloc_gvas_for_save(self, requests: list[ReqMeta]) -> None:
        """Allocate per-group GVA on the worker side right before batch_copy.

        For multi-group models, iterates all KV cache groups and allocates
        per-group GVAs. Key format: model@group_id@hash@head_or_tp_rank
        (multi-group) or model@hash@head_or_tp_rank (single-group, backward
        compat with PR #11585).
        """
        if self._kv_role == "kv_consumer" and not self._consumer_is_to_put:
            return
        if self._tp_rank % self._put_step != 0:
            return
        for request in requests:
            if request.can_save is None or not request.can_save:
                continue
            block_hashes = request.block_hashes

            all_group_gvas: list[np.ndarray] = []
            all_group_block_ids: list[np.ndarray] = []
            all_group_save_keys: list[str] = []
            request.partial_save_gva_per_group = [0] * self._num_kv_cache_groups
            for group_id in range(self._num_kv_cache_groups):
                group_block_size = self._grouped_block_size[group_id]
                effective_block_size = group_block_size
                group_block_len = self._group_block_len.get(group_id, self._group_block_len.get(0, []))
                alloc_size = sum(group_block_len) if group_block_len else self._page_size_bytes

                group_block_hashes = get_block_hashes(block_hashes, effective_block_size, self._hash_block_size)
                block_ids_by_group = (
                    request.block_ids_by_group_np[group_id]
                    if (request.block_ids_by_group_np is not None and group_id < len(request.block_ids_by_group_np))
                    else request.block_ids_np
                )
                if block_ids_by_group is None:
                    raise RuntimeError(f"Block IDs are not initialized for request {request.req_id}")

                save_start_block = request.save_start_token // effective_block_size
                save_end_block = request.save_end_token // effective_block_size
                if request.load_spec is not None and request.load_spec.can_load:
                    pool_hit_tokens = (
                        request.load_spec.kvpool_store_skip_tokens
                        if request.load_spec.kvpool_store_skip_tokens is not None
                        else request.load_spec.kvpool_cached_tokens
                    )
                    hit_full_blocks = pool_hit_tokens // effective_block_size
                    save_start_block = max(save_start_block, hit_full_blocks)
                candidate_keys = [
                    self._make_gva_key(
                        group_id,
                        block_hash_to_str(group_block_hashes[block_idx]),
                    )
                    for block_idx in range(
                        save_start_block,
                        min(save_end_block, len(group_block_hashes)),
                    )
                ]
                self._refresh_allocated_gvas(candidate_keys)
                # Skip blocks that are still present and readable in MemCache.
                while save_start_block < save_end_block and save_start_block < len(group_block_hashes):
                    key = self._make_gva_key(group_id, block_hash_to_str(group_block_hashes[save_start_block]))
                    if key in self._allocated_gvas:
                        save_start_block += 1
                    else:
                        break

                block_gvas: list[int] = []
                new_keys: list[str] = []
                new_positions: list[int] = []
                for blk_idx in range(save_start_block, min(save_end_block, len(group_block_hashes))):
                    key = self._make_gva_key(group_id, block_hash_to_str(group_block_hashes[blk_idx]))
                    cached = self._allocated_gvas.get(key)
                    if cached is not None:
                        block_gvas.append(cached)
                    else:
                        new_keys.append(key)
                        new_positions.append(len(block_gvas))
                        block_gvas.append(0)

                if new_keys:
                    new_gvas = self._store.batch_alloc(new_keys, [alloc_size] * len(new_keys))
                    if any(gva <= 0 for gva in new_gvas):
                        logger.error(
                            "alloc_gvas FAIL: req=%s group=%d alloc_size=%d new_keys=%d gvas_sample=%s zero_count=%d",
                            request.req_id,
                            group_id,
                            alloc_size,
                            len(new_keys),
                            new_gvas[:5],
                            sum(1 for g in new_gvas if g <= 0),
                        )
                    for pos, key, gva in zip(new_positions, new_keys, new_gvas):
                        if gva > 0:
                            block_gvas[pos] = gva
                            self._allocated_gvas[key] = gva
                            all_group_save_keys.append(key)

                partial_block_index = get_partial_block_index(
                    request.target_token_len,
                    effective_block_size,
                    len(group_block_hashes),
                    self._layerwise_offload,
                )
                if partial_block_index is not None and partial_block_index < len(block_ids_by_group):
                    partial_key = self._make_partial_key(
                        request,
                        group_id,
                        partial_block_index,
                        request.target_token_len,
                    )
                    partial_gva = self._allocated_gvas.get(partial_key)
                    if partial_gva is None:
                        allocated = self._store.batch_alloc(
                            [partial_key],
                            [alloc_size],
                        )
                        partial_gva = allocated[0] if allocated else 0
                        if partial_gva > 0:
                            self._allocated_gvas[partial_key] = partial_gva
                            all_group_save_keys.append(partial_key)
                        else:
                            logger.error(
                                "alloc_gvas: partial allocation failed req=%s group=%d block=%d gva=%d",
                                request.req_id,
                                group_id,
                                partial_block_index,
                                partial_gva,
                            )
                    # Partial keys are request-scoped; do not retain them forever.
                    self._allocated_gvas.pop(partial_key, None)
                    request.partial_save_gva_per_group[group_id] = partial_gva

                logger.debug(
                    "alloc_gvas: req=%s group=%d eff_bs=%d save_blocks=[%d,%d) "
                    "new_keys=%d cached_keys=%d alloc_size=%d",
                    request.req_id,
                    group_id,
                    effective_block_size,
                    save_start_block,
                    save_end_block,
                    len(new_keys),
                    len(block_gvas) - len(new_keys),
                    alloc_size,
                )

                # Pad block_gvas to match block_ids length (fill 0 for blocks before save_start)
                full_gvas = [0] * len(block_ids_by_group)
                for i, gva in enumerate(block_gvas):
                    if save_start_block + i < len(full_gvas):
                        full_gvas[save_start_block + i] = gva

                all_group_gvas.append(np.asarray(full_gvas, dtype=np.int64))
                all_group_block_ids.append(np.asarray(block_ids_by_group, dtype=np.int64))

            if all_group_gvas:
                request.save_keys = all_group_save_keys
                request.block_gvas_by_group_np = all_group_gvas
                request.block_ids_by_group_np = all_group_block_ids
                request.block_gvas_np = all_group_gvas[0]
                request.gva_block_offset = 0

    def prepare_load_gvas(self, requests: list[ReqMeta]) -> None:
        """Fetch per-rank GVA and acquire read lease for the load path.

        memcache requires batch_copy (read) to find the blob in the per-process
        gvaBlobTracker with a valid lease. The scheduler only checks existence
        (batch_is_exist) to decide the load range; before batch_copy(G2L) the
        worker must, for its own per-rank keys:
          1. batch_get_key_info to fetch the GVA (fills block_gvas_np)
          2. batch_add_lease to register the blob locally + acquire a read lease
        """
        for request in requests:
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            cached_tokens = request.load_spec.kvpool_cached_tokens
            if not self._use_eagle and request.load_spec.kvpool_store_skip_tokens is not None:
                cached_tokens = request.load_spec.kvpool_store_skip_tokens
            block_hashes = request.block_hashes

            all_group_load_gvas: list[np.ndarray] = []
            all_group_load_keys: list[str] = []
            request.partial_load_gva_per_group = [0] * self._num_kv_cache_groups
            for group_id in range(self._num_kv_cache_groups):
                group_block_size = self._grouped_block_size[group_id]
                effective_block_size = group_block_size

                group_block_hashes = get_block_hashes(block_hashes, effective_block_size, self._hash_block_size)
                load_start_block = (
                    0 if self._layerwise_offload else request.load_spec.vllm_cached_tokens // effective_block_size
                )
                cached_full_blocks = cached_tokens // effective_block_size
                full_blocks = min(cached_full_blocks, len(group_block_hashes))

                block_ids_by_group = (
                    request.block_ids_by_group_np[group_id]
                    if (request.block_ids_by_group_np is not None and group_id < len(request.block_ids_by_group_np))
                    else request.block_ids_np
                )
                if block_ids_by_group is None:
                    all_group_load_gvas.append(np.zeros(0, dtype=np.int64))
                    continue
                full_len = len(block_ids_by_group)

                partial_block_index = get_partial_block_index(
                    cached_tokens,
                    effective_block_size,
                    len(group_block_hashes),
                    self._layerwise_offload,
                )
                if partial_block_index is not None and (
                    partial_block_index < load_start_block or partial_block_index >= full_len
                ):
                    partial_block_index = None

                if load_start_block >= full_blocks and partial_block_index is None:
                    all_group_load_gvas.append(np.zeros(full_len, dtype=np.int64))
                    continue

                keys = [
                    self._make_gva_key(group_id, block_hash_to_str(group_block_hashes[i]))
                    for i in range(load_start_block, full_blocks)
                ]
                block_indices = list(range(load_start_block, full_blocks))
                if partial_block_index is not None:
                    keys.append(
                        self._make_partial_key(
                            request,
                            group_id,
                            partial_block_index,
                            cached_tokens,
                        )
                    )
                    block_indices.append(partial_block_index)
                if not keys:
                    all_group_load_gvas.append(np.zeros(full_len, dtype=np.int64))
                    continue

                key_infos = self._store.batch_get_key_info(keys)
                gvas = []
                valid_gva_indices = []
                invalid_block_ids: list[int] = []
                for ki, key, block_idx in zip(key_infos, keys, block_indices):
                    sizes = ki.size()
                    gva = ki.gva_list()[0] if sizes and sizes > 0 else 0
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

                # Only call batch_add_lease for keys with valid size
                valid_keys = [keys[index] for index in valid_gva_indices]
                if valid_keys:
                    lease_results = self._store.batch_add_lease(valid_keys, LAYERWISE_READ_LEASE_TTL_MS)
                    if len(lease_results) != len(valid_keys):
                        raise RuntimeError(
                            "MemCache lease returned unexpected number of results: "
                            f"expected={len(valid_keys)}, actual={len(lease_results)}"
                        )
                    leased_keys = []
                    for gva_index, lease_res in zip(valid_gva_indices, lease_results):
                        block_idx = block_indices[gva_index]
                        if lease_res == MEMCACHE_UNMATCHED_STATE and block_idx == partial_block_index:
                            partial_key = keys[gva_index]
                            for retry in range(1, PARTIAL_LEASE_RETRY_COUNT + 1):
                                time.sleep(PARTIAL_LEASE_RETRY_INTERVAL_S)
                                retry_results = self._store.batch_add_lease(
                                    [partial_key],
                                    LAYERWISE_READ_LEASE_TTL_MS,
                                )
                                if len(retry_results) != 1:
                                    raise RuntimeError(
                                        "MemCache partial lease retry returned "
                                        f"unexpected number of results: {len(retry_results)}"
                                    )
                                lease_res = retry_results[0]
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

                # Report invalid blocks to scheduler for recompute.
                # Single-group models can safely report individual block IDs.
                # Multi-group (hybrid) models must not report partial group
                # failures, as the scheduler cannot handle inconsistent KV
                # cache state across groups (see PR #9701 for rationale).
                if invalid_block_ids:
                    if self._num_kv_cache_groups == 1:
                        self._on_invalid_blocks(invalid_block_ids)
                    else:
                        leased_keys_to_release = list(
                            dict.fromkeys(
                                [
                                    *all_group_load_keys,
                                    *leased_keys,
                                ]
                            )
                        )
                        if leased_keys_to_release:
                            self._store.batch_remove_lease(leased_keys_to_release)
                        raise RuntimeError(
                            "Layerwise multi-group KV load failed and cannot "
                            "safely fall back to per-block recomputation: "
                            f"request={request.req_id}, "
                            f"failed_blocks={invalid_block_ids}"
                        )
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

                # Pad to match block_ids_by_group length, with 0s before load_start_block
                full_gvas = [0] * full_len
                normal_gva_count = full_blocks - load_start_block
                for i, gva in enumerate(gvas[:normal_gva_count]):
                    if load_start_block + i < len(full_gvas):
                        full_gvas[load_start_block + i] = gva
                all_group_load_gvas.append(np.asarray(full_gvas, dtype=np.int64))
                if partial_block_index is not None and len(gvas) > normal_gva_count:
                    request.partial_load_gva_per_group[group_id] = gvas[normal_gva_count]

            if all_group_load_gvas:
                request.load_keys = all_group_load_keys
                request.load_block_gvas_by_group_np = all_group_load_gvas
                request.load_block_gvas_np = all_group_load_gvas[0]
                request.load_gva_block_offset = 0


class GVAHitChecker:
    """Scheduler-side GVA prefix hit computation."""

    def __init__(
        self,
        store: GVAStore,
        model_name: str,
        head_or_tp_ranks: int,
        grouped_block_size: list[int],
        hash_block_size: int,
        num_groups: int,
        use_layerwise: bool,
    ) -> None:
        self._store = store
        self._model_name = model_name
        self._head_or_tp_ranks = head_or_tp_ranks
        self._grouped_block_size = grouped_block_size
        self._hash_block_size = hash_block_size
        self._num_groups = num_groups
        self._use_layerwise = use_layerwise

    def _make_hit_check_keys(self, group_id: int, block_hash_hex: str) -> list[str]:
        return GVAKeyFactory.hit_check_keys(
            self._model_name,
            group_id,
            block_hash_hex,
            self._head_or_tp_ranks,
            self._num_groups,
        )

    def hit_tokens(self, request: Request, token_len: int, num_computed_tokens: int) -> int:
        # In layerwise mode, always query from block 0 because the remote
        # pool stores per-layer data that may not match local prefix cache.
        num_hash_blocks = token_len // self._hash_block_size
        block_hashes_to_check = request.block_hashes[:num_hash_blocks]
        hits_per_group: list[int] = []

        for group_id in range(len(self._grouped_block_size)):
            effective_block_size = get_group_block_size(self._grouped_block_size, group_id)
            group_block_hashes = get_block_hashes(block_hashes_to_check, effective_block_size, self._hash_block_size)
            query_start_block = (
                0 if self._use_layerwise else min(num_computed_tokens // effective_block_size, len(group_block_hashes))
            )
            group_block_hashes = group_block_hashes[query_start_block:]
            # Generate all-rank keys for each block hash
            keys_by_block = [self._make_hit_check_keys(group_id, block_hash_to_str(bh)) for bh in group_block_hashes]
            all_keys = [key for block_keys in keys_by_block for key in block_keys]
            if not all_keys:
                continue

            key_infos = self._store.batch_get_key_info(all_keys)
            if len(key_infos) != len(all_keys):
                logger.error(
                    "KV pool batch_get_key_info returned unexpected number of results: expected=%d, actual=%d",
                    len(all_keys),
                    len(key_infos),
                )
                hits_per_group.append(0)
                continue

            # A block is hit only when ALL ranks' keys return valid GVA
            num_hit_blocks = 0
            offset = 0
            for block_keys in keys_by_block:
                block_infos = key_infos[offset : offset + len(block_keys)]
                offset += len(block_keys)
                if all(ki.size() and ki.size() > 0 for ki in block_infos):
                    num_hit_blocks += 1
                else:
                    break

            hits_per_group.append((query_start_block + num_hit_blocks) * effective_block_size)

        if not hits_per_group:
            logger.debug(
                "hit_check: req=%s token_len=%d no participating groups (all skipped)",
                request.request_id,
                token_len,
            )
            return 0
        hit_tokens = min(hits_per_group)
        logger.debug(
            "hit_check: req=%s token_len=%d hits_per_group=%s hit_tokens=%d",
            request.request_id,
            token_len,
            hits_per_group,
            hit_tokens,
        )
        return hit_tokens
