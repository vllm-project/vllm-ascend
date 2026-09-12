from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    LayerBatchReqMeta,
    LayerBlockRange,
    LayerRangeReqMeta,
    LayerTransferTask,
    ReqMeta,
    SharedBlockData,
    block_hash_to_str,
    get_block_hashes,
    get_partial_block_index,
)

LAYERWISE_READ_LEASE_TTL_MS = 5 * 60 * 1000
MEMCACHE_UNMATCHED_STATE = -3101
PARTIAL_LEASE_RETRY_COUNT = 10
PARTIAL_LEASE_RETRY_INTERVAL_S = 0.001


class LayerTransferArrayBuilder:
    """Build vectorized local/GVA transfer arrays for one KV cache group."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        num_layers: int,
        group_id: int = 0,
    ) -> None:
        self._block_len_np = np.asarray(token_database.group_block_len[group_id], dtype=np.int64)
        self._kv_caches_base_addr_np = np.asarray(
            token_database.group_kv_caches_base_addr[group_id],
            dtype=np.int64,
        )
        group_block_stride = token_database.group_block_stride.get(group_id, token_database.group_block_len[group_id])
        self._block_stride_np = np.asarray(group_block_stride, dtype=np.int64)
        offsets_by_group = getattr(token_database, "group_layer_cache_entry_offsets", None)
        layer_offsets = offsets_by_group.get(group_id) if isinstance(offsets_by_group, dict) else None
        if layer_offsets is None:
            if num_layers <= 0 or self._block_len_np.size % num_layers:
                raise ValueError(
                    f"Cannot infer a uniform layerwise layout: legs={self._block_len_np.size}, layers={num_layers}"
                )
            caches_per_layer = self._block_len_np.size // num_layers
            layer_offsets = np.arange(num_layers + 1, dtype=np.int64) * caches_per_layer
        else:
            layer_offsets = np.asarray(layer_offsets, dtype=np.int64)
            if (
                layer_offsets.size != num_layers + 1
                or layer_offsets[0] != 0
                or layer_offsets[-1] != self._block_len_np.size
                or np.any(layer_offsets[1:] < layer_offsets[:-1])
            ):
                raise ValueError(
                    f"Invalid layerwise offsets for group {group_id}: "
                    f"offsets={layer_offsets.tolist()}, legs={self._block_len_np.size}, layers={num_layers}"
                )
        self._layer_offsets_np = layer_offsets

        # The flat buffers are layer-major. Their byte prefix therefore gives
        # both each layer's compact GVA offset and each leg's inner-layer offset.
        byte_prefix = np.empty(self._block_len_np.size + 1, dtype=np.int64)
        byte_prefix[0] = 0
        np.cumsum(self._block_len_np, out=byte_prefix[1:])
        self._layer_gva_offsets_np = byte_prefix[self._layer_offsets_np[:-1]]
        leg_counts = np.diff(self._layer_offsets_np)
        self._leg_inner_offsets_np = byte_prefix[:-1] - np.repeat(
            self._layer_gva_offsets_np,
            leg_counts,
        )
        self.page_size_bytes = int(np.diff(byte_prefix[self._layer_offsets_np]).max())

    def _build_transfer_arrays(
        self,
        block_ids_arr: np.ndarray,
        base_gvas_arr: np.ndarray,
        layer_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        leg_start = int(self._layer_offsets_np[layer_id])
        leg_end = int(self._layer_offsets_np[layer_id + 1])
        layer_base_addrs = self._kv_caches_base_addr_np[leg_start:leg_end]
        layer_block_len = self._block_len_np[leg_start:leg_end]
        layer_block_stride = self._block_stride_np[leg_start:leg_end]
        layer_inner_offsets = self._leg_inner_offsets_np[leg_start:leg_end]
        rank_layer_offset = self._layer_gva_offsets_np[layer_id]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[KVPOOL] build_transfer layer=%d page_size=%d cache_legs=%d "
                "rank_layer_offset=%d layer_block_len=%s layer_inner_offsets=%s base_gvas=%s",
                layer_id,
                self.page_size_bytes,
                leg_end - leg_start,
                rank_layer_offset,
                layer_block_len.tolist(),
                layer_inner_offsets.tolist(),
                base_gvas_arr.tolist(),
            )

        addr_arr = layer_base_addrs[None, :] + block_ids_arr[:, None] * layer_block_stride[None, :]
        size_arr = np.broadcast_to(layer_block_len, addr_arr.shape)
        gvas_arr = base_gvas_arr[:, None] + rank_layer_offset + layer_inner_offsets[None, :]
        return addr_arr.ravel(), size_arr.ravel(), gvas_arr.ravel()


class LayerBatchBuilder(LayerTransferArrayBuilder):
    def __init__(
        self, token_database: ChunkedTokenDatabase, page_size_bytes: int, num_layers: int, group_id: int = 0
    ) -> None:
        super().__init__(token_database, num_layers, group_id)
        self.num_layers = num_layers
        self.group_id = group_id
        self._layer_cache_entry_offsets_np = self._layer_offsets_np
        self._block_ids_buf: np.ndarray | None = None
        self._block_gvas_buf: np.ndarray | None = None

    def _ensure_buf(self, capacity: int) -> tuple[np.ndarray, np.ndarray]:
        if self._block_ids_buf is None or len(self._block_ids_buf) < capacity:
            self._block_ids_buf = np.empty(capacity, dtype=np.int64)
            self._block_gvas_buf = np.empty(capacity, dtype=np.int64)
        assert self._block_ids_buf is not None and self._block_gvas_buf is not None
        return self._block_ids_buf[:capacity], self._block_gvas_buf[:capacity]

    @staticmethod
    def _dedupe_transfer_blocks(
        block_ids_arr: np.ndarray,
        block_gvas_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if block_ids_arr.size <= 1:
            return block_ids_arr, block_gvas_arr

        block_transfer_array = np.column_stack((block_ids_arr, block_gvas_arr))
        _, unique_indices = np.unique(
            block_transfer_array,
            axis=0,
            return_index=True,
        )
        if unique_indices.size == block_ids_arr.size:
            return block_ids_arr, block_gvas_arr

        return (
            block_ids_arr[unique_indices],
            block_gvas_arr[unique_indices],
        )

    def _require_request_arrays(
        self,
        block_range: LayerBlockRange,
        is_save: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        request = block_range.request
        group_id = self.group_id
        block_ids_np: np.ndarray | None
        block_gvas_np: np.ndarray | None
        if is_save:
            group_block_ids = request.block_ids_by_group_np
            group_block_gvas = request.block_gvas_by_group_np
            if (
                group_block_ids is not None
                and group_block_gvas is not None
                and group_id < len(group_block_ids)
                and group_id < len(group_block_gvas)
            ):
                block_ids_np = group_block_ids[group_id]
                block_gvas_np = group_block_gvas[group_id]
            else:
                block_ids_np = request.block_ids_np
                block_gvas_np = request.block_gvas_np
        else:
            group_block_ids = request.block_ids_by_group_np
            group_block_gvas = request.load_block_gvas_by_group_np
            if (
                group_block_ids is not None
                and group_block_gvas is not None
                and group_id < len(group_block_ids)
                and group_id < len(group_block_gvas)
            ):
                block_ids_np = group_block_ids[group_id]
                block_gvas_np = group_block_gvas[group_id]
            else:
                block_ids_np = request.block_ids_np
                block_gvas_np = request.load_block_gvas_np
        if block_ids_np is None or block_gvas_np is None:
            raise RuntimeError(
                f"ReqMeta {'save' if is_save else 'load'} block metadata"
                f" is not initialized for request {request.req_id}"
            )
        return block_ids_np, block_gvas_np

    def build_shared(self, task: LayerTransferTask, is_save: bool = True) -> SharedBlockData | None:
        """Pre-compute shared block data that is identical across all layers."""
        if not task.block_ranges:
            return None

        if task.use_key_major_ranges:
            return self._build_key_major_shared(task, is_save)

        total = 0
        for block_range in task.block_ranges:
            total += block_range.end_block - block_range.start_block
            if block_range.partial_block_index is not None:
                total += 1

        block_ids_arr, block_gvas_arr = self._ensure_buf(total)
        req_ids: list[str] = []
        is_last_chunks: list[bool | None] = []
        all_save_keys: list[str] = []
        all_load_keys: list[str] = []
        collected_load_keys_request_ids: set[int] = set()
        offset = 0

        for block_range in task.block_ranges:
            request = block_range.request
            req_ids.append(request.req_id)
            is_last_chunks.append(request.is_last_chunk)
            if request.save_keys:
                all_save_keys.extend(request.save_keys)
            if request.load_keys and id(request) not in collected_load_keys_request_ids:
                # Avoid collecting the same request's load_keys multiple times
                collected_load_keys_request_ids.add(id(request))
                all_load_keys.extend(request.load_keys)
            block_ids_np, block_gvas_np = self._require_request_arrays(block_range, is_save)
            gva_block_offset = request.gva_block_offset if is_save else request.load_gva_block_offset

            num_blocks = block_range.end_block - block_range.start_block
            if num_blocks > 0:
                gva_start = block_range.start_block - gva_block_offset
                gva_end = block_range.end_block - gva_block_offset
                if gva_start < 0 or gva_end > len(block_gvas_np):
                    raise RuntimeError(
                        "ReqMeta GVA metadata does not cover requested block "
                        f"range [{block_range.start_block}, {block_range.end_block}) "
                        f"with offset {gva_block_offset}"
                    )
                end = offset + num_blocks
                block_ids_arr[offset:end] = block_ids_np[block_range.start_block : block_range.end_block]
                block_gvas_arr[offset:end] = block_gvas_np[gva_start:gva_end]
                offset = end

            if block_range.partial_block_index is not None:
                partial_block_gva = None
                partial_gva_per_group = (
                    request.partial_save_gva_per_group if is_save else request.partial_load_gva_per_group
                )
                if task.group_id < len(partial_gva_per_group):
                    partial_block_gva = partial_gva_per_group[task.group_id]
                if partial_block_gva is None:
                    partial_block_gva = request.last_block_gva
                assert partial_block_gva is not None
                block_ids_arr[offset] = block_ids_np[block_range.partial_block_index]
                block_gvas_arr[offset] = partial_block_gva
                offset += 1

        block_ids_slice = block_ids_arr[:offset]
        block_gvas_slice = block_gvas_arr[:offset]
        valid_mask = block_gvas_slice > 0
        if not np.all(valid_mask):
            skip_count = int(np.sum(~valid_mask))
            logger.warning(
                "[KVPOOL] build_shared skipping %d blocks with invalid gva (gva<=0)",
                skip_count,
            )
            block_ids_slice = block_ids_slice[valid_mask]
            block_gvas_slice = block_gvas_slice[valid_mask]

        block_ids_arr, block_gvas_arr = self._dedupe_transfer_blocks(block_ids_slice, block_gvas_slice)

        logger.debug(
            "[KVPOOL] build_shared req_ids=%s block_gvas_arr=%s block_ids_arr=%s",
            req_ids,
            block_gvas_arr.tolist(),
            block_ids_arr.tolist(),
        )
        return SharedBlockData(
            block_ids_arr=block_ids_arr,
            block_gvas_arr=block_gvas_arr,
            req_ids=req_ids,
            is_last_chunks=is_last_chunks,
            save_keys=all_save_keys,
            load_keys=all_load_keys,
        )

    @staticmethod
    def _request_block_keys(
        request: ReqMeta,
        is_save: bool,
    ) -> tuple[list[str | None], int, str | None]:
        if is_save:
            return (
                request.save_block_keys,
                request.save_key_block_offset,
                request.save_last_block_key,
            )
        return (
            request.load_block_keys,
            request.load_key_block_offset,
            request.load_last_block_key,
        )

    def _request_block_ids(self, request: ReqMeta) -> list[int]:
        if request.block_ids_by_group_np is not None and self.group_id < len(request.block_ids_by_group_np):
            return request.block_ids_by_group_np[self.group_id].tolist()
        if request.block_ids_np is not None:
            return request.block_ids_np.tolist()
        if self.group_id < len(request.block_ids_by_group):
            return request.block_ids_by_group[self.group_id]
        return request.block_ids

    def _build_key_major_shared(
        self,
        task: LayerTransferTask,
        is_save: bool,
    ) -> SharedBlockData:
        """Build block-key rows shared by all Mooncake layer ranges."""
        block_ids: list[int] = []
        block_keys: list[str] = []
        req_ids: list[str] = []
        is_last_chunks: list[bool | None] = []
        all_load_keys: list[str] = []
        seen_save_keys: set[str] = set()

        for block_range in task.block_ranges:
            request = block_range.request
            req_ids.append(request.req_id)
            is_last_chunks.append(request.is_last_chunk)
            all_load_keys.extend(request.load_keys)
            request_block_ids = self._request_block_ids(request)
            if (
                block_range.start_block < 0
                or block_range.end_block < block_range.start_block
                or block_range.end_block > len(request_block_ids)
            ):
                raise RuntimeError(
                    "ReqMeta block metadata does not cover requested block range "
                    f"[{block_range.start_block}, {block_range.end_block})"
                )

            request_keys, key_block_offset, last_block_key = self._request_block_keys(request, is_save)
            key_start = block_range.start_block - key_block_offset
            key_end = block_range.end_block - key_block_offset
            if key_start < 0 or key_end > len(request_keys):
                raise RuntimeError(
                    f"ReqMeta {'save' if is_save else 'load'} block key metadata "
                    f"does not cover requested range [{block_range.start_block}, "
                    f"{block_range.end_block}) with offset {key_block_offset}"
                )

            for block_id, key in zip(
                request_block_ids[block_range.start_block : block_range.end_block],
                request_keys[key_start:key_end],
                strict=True,
            ):
                if key is None or (is_save and key in seen_save_keys):
                    continue
                block_ids.append(block_id)
                block_keys.append(key)
                if is_save:
                    seen_save_keys.add(key)

            partial_block_index = block_range.partial_block_index
            if partial_block_index is not None:
                if partial_block_index < 0 or partial_block_index >= len(request_block_ids):
                    raise RuntimeError(f"ReqMeta block metadata does not cover partial block {partial_block_index}")
                if last_block_key is not None and (not is_save or last_block_key not in seen_save_keys):
                    block_ids.append(request_block_ids[partial_block_index])
                    block_keys.append(last_block_key)
                    if is_save:
                        seen_save_keys.add(last_block_key)

        return SharedBlockData(
            block_ids_arr=np.asarray(block_ids, dtype=np.int64),
            block_gvas_arr=None,
            block_keys=block_keys,
            req_ids=req_ids,
            is_last_chunks=is_last_chunks,
            load_keys=all_load_keys,
        )

    def build_addrs(
        self,
        shared: SharedBlockData,
        layer_id: int,
    ) -> LayerBatchReqMeta | LayerRangeReqMeta:
        """Compute per-layer addresses from pre-computed shared block data."""
        if shared.block_keys is not None:
            base_offset = int(self._layer_cache_entry_offsets_np[layer_id])
            end_offset = int(self._layer_cache_entry_offsets_np[layer_id + 1])
            layer_base_addrs = self._kv_caches_base_addr_np[base_offset:end_offset]
            layer_block_len = self._block_len_np[base_offset:end_offset]
            layer_block_stride = self._block_stride_np[base_offset:end_offset]
            layer_inner_offsets = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(layer_block_len[:-1], dtype=np.int64))
            )
            layer_object_offset = int(self._block_len_np[:base_offset].sum())
            offsets = (layer_object_offset + layer_inner_offsets).tolist()
            sizes = layer_block_len.tolist()
            all_buffers = [
                (layer_base_addrs + block_id * layer_block_stride).tolist() for block_id in shared.block_ids_arr
            ]
            return LayerRangeReqMeta(
                req_ids=shared.req_ids,
                layer_id=layer_id,
                block_ids=shared.block_ids_arr.tolist(),
                keys=shared.block_keys,
                all_buffers=all_buffers,
                all_sizes=[sizes.copy() for _ in shared.block_ids_arr],
                all_offsets=[offsets.copy() for _ in shared.block_ids_arr],
                load_keys=shared.load_keys,
            )

        assert shared.block_gvas_arr is not None
        addr_array, size_array, gvas_array = self._build_transfer_arrays(
            shared.block_ids_arr, shared.block_gvas_arr, layer_id
        )

        return LayerBatchReqMeta(
            req_ids=shared.req_ids,
            layer_id=layer_id,
            is_last_chunks=shared.is_last_chunks,
            addr_array=addr_array,
            size_array=size_array,
            gvas_array=gvas_array,
            load_keys=shared.load_keys,
        )

    def build(
        self,
        task: LayerTransferTask,
        is_save: bool = True,
    ) -> LayerBatchReqMeta | LayerRangeReqMeta | None:
        """Full build: shared data + per-layer addresses (backward compat)."""
        shared = self.build_shared(task, is_save)
        if shared is None:
            return None
        layer_index = task.layer_id if task.use_key_major_ranges else task.layer_idx_in_group
        return self.build_addrs(shared, layer_index)


@dataclass
class GroupKeyPlan:
    request: ReqMeta
    group_id: int
    block_ids: np.ndarray
    block_indices: list[int]
    keys: list[str]
    partial_block_index: int | None
    partial_key: str | None


class LayerwiseTransferPreparer:
    """Own GVA metadata and leases without depending on the pool worker.

    Keep backend key protocols, partial-snapshot retry, and reachable-block
    filtering from main while sharing preparation across all layer tasks.
    """

    def __init__(
        self,
        m_store: Backend,
        model_name: str,
        head_or_tp_rank: int,
        hash_block_size: int,
        *,
        enabled: bool,
        can_allocate: bool,
        block_sizes: list[int],
        group_block_len: dict[int, list[int]],
        page_size_bytes: int,
        layerwise_offload: bool,
        protocol: Any,
        record_invalid_blocks: Callable[[list[int]], None],
        use_eagle: bool = False,
        allocated_gvas: dict[str, int] | None = None,
    ) -> None:
        self.m_store = m_store
        self.model_name = model_name
        self.head_or_tp_rank = head_or_tp_rank
        self.hash_block_size = hash_block_size
        self.use_layerwise_transfer = enabled
        self.enabled = enabled
        self.can_allocate = can_allocate
        self.grouped_block_size = block_sizes
        self.num_kv_cache_groups = len(block_sizes)
        self.group_block_len = group_block_len
        self.page_size_bytes = page_size_bytes
        self.layerwise_offload = layerwise_offload
        self.layerwise_protocol = protocol
        self.record_invalid_blocks = record_invalid_blocks
        self.use_eagle = use_eagle
        self._allocated_gvas = allocated_gvas if allocated_gvas is not None else {}
        self.load_lease_keys_by_request: dict[str, set[str]] = {}
        self.load_lease_refcounts: dict[str, int] = {}
        self._lease_lock = threading.RLock()

    def _make_layerwise_full_key(self, group_id: int, block_hash_hex: str) -> str:
        """Full-block key for the layerwise transfer, built by the
        backend's protocol module.

        Single-group models use the PR #11585 format (model@hash@rank) for
        backward compatibility. Multi-group models include group_id
        (model@group_id@hash@rank) to distinguish groups.
        """
        return self.layerwise_protocol.make_full_key(
            self.model_name,
            group_id,
            block_hash_hex,
            self.head_or_tp_rank,
            self.num_kv_cache_groups,
        )

    def _make_layerwise_partial_key(
        self,
        request: ReqMeta,
        group_id: int,
        block_index: int,
        end_token: int,
    ) -> str:
        return self.layerwise_protocol.make_partial_key(
            self.model_name,
            request.req_id,
            group_id,
            block_index,
            end_token,
            self.head_or_tp_rank,
        )

    def _refresh_allocated_gvas(self, keys: list[str]) -> None:
        """Drop local GVA entries whose MemCache blobs were evicted."""
        cached_keys = list(dict.fromkeys(key for key in keys if key in self._allocated_gvas))
        if not cached_keys:
            return
        exists_states = self.m_store.batch_is_exist(cached_keys)
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

    def _build_key_plans(self, requests: list[ReqMeta], *, is_save: bool) -> list[GroupKeyPlan]:
        """Resolve logical ranges once, before any backend operation."""
        plans = []
        for request in requests:
            if is_save:
                if not request.can_save:
                    continue
                cached_tokens = request.target_token_len
            else:
                if request.load_spec is None or not request.load_spec.can_load:
                    continue
                cached_tokens = request.load_spec.kvpool_cached_tokens
                if not self.use_eagle and request.load_spec.kvpool_store_skip_tokens is not None:
                    cached_tokens = request.load_spec.kvpool_store_skip_tokens
            for group_id, block_size in enumerate(self.grouped_block_size):
                group_ids = request.block_ids_by_group_np
                block_ids = (
                    group_ids[group_id] if group_ids is not None and group_id < len(group_ids) else request.block_ids_np
                )
                if block_ids is None:
                    raise RuntimeError(f"Block IDs are not initialized for request {request.req_id}, group {group_id}")
                hashes = get_block_hashes(request.block_hashes, block_size, self.hash_block_size)
                if is_save:
                    start = request.save_start_token // block_size
                    end = request.save_end_token // block_size
                    if request.load_spec is not None and request.load_spec.can_load:
                        pool_hit = request.load_spec.kvpool_store_skip_tokens
                        if pool_hit is None:
                            pool_hit = request.load_spec.kvpool_cached_tokens
                        start = max(start, pool_hit // block_size)
                    masks = request.store_masks
                else:
                    assert request.load_spec is not None
                    start = 0 if self.layerwise_offload else request.load_spec.vllm_cached_tokens // block_size
                    end = cached_tokens // block_size
                    masks = request.load_masks
                mask = masks[group_id] if masks is not None and group_id < len(masks) else None
                indices = [
                    i
                    for i in range(start, min(end, len(hashes), len(block_ids)))
                    if mask is None or i >= len(mask) or mask[i]
                ]
                keys = [self._make_layerwise_full_key(group_id, block_hash_to_str(hashes[i])) for i in indices]
                partial = get_partial_block_index(cached_tokens, block_size, len(hashes), self.layerwise_offload)
                if partial is not None and (partial < start or partial >= len(block_ids)):
                    partial = None
                partial_key = (
                    self._make_layerwise_partial_key(request, group_id, partial, cached_tokens)
                    if partial is not None
                    else None
                )
                plans.append(
                    GroupKeyPlan(
                        request, group_id, np.asarray(block_ids, dtype=np.int64), indices, keys, partial, partial_key
                    )
                )
        return plans

    def _alloc_gvas_for_save(self, requests: list[ReqMeta]) -> None:
        """Allocate all groups in one batch, preserving cached and partial keys."""
        if not self.enabled or not self.can_allocate:
            return
        plans = self._build_key_plans(requests, is_save=True)
        self._refresh_allocated_gvas([key for plan in plans for key in plan.keys])
        sizes_by_key: dict[str, int] = {}
        for plan in plans:
            # Already-published prefixes must not be written again. Preserve
            # main's eviction check before consulting the local GVA cache.
            prefix = 0
            for key in plan.keys:
                if key not in self._allocated_gvas:
                    break
                prefix += 1
            plan.keys = plan.keys[prefix:]
            plan.block_indices = plan.block_indices[prefix:]
            block_lengths = self.group_block_len.get(plan.group_id, self.group_block_len.get(0, []))
            alloc_size = sum(block_lengths) if block_lengths else self.page_size_bytes
            for key in [*plan.keys, *([plan.partial_key] if plan.partial_key else [])]:
                if key in self._allocated_gvas:
                    continue
                if key in sizes_by_key and sizes_by_key[key] != alloc_size:
                    raise RuntimeError(f"Conflicting allocation sizes for layerwise key {key}")
                sizes_by_key[key] = alloc_size
        keys = list(sizes_by_key)
        new_gvas = (
            self.m_store.batch_alloc(keys, list(sizes_by_key.values()), LAYERWISE_READ_LEASE_TTL_MS) if keys else []
        )
        if len(new_gvas) != len(keys) or any(gva <= 0 for gva in new_gvas):
            raise RuntimeError(f"batch_alloc returned invalid GVAs: expected={len(keys)}, actual={len(new_gvas)}")
        new_gvas_by_key = dict(zip(keys, new_gvas, strict=True))
        seen_requests: set[str] = set()
        copied_keys: set[str] = set()
        for plan in plans:
            request = plan.request
            if request.req_id not in seen_requests:
                request.block_gvas_by_group_np = [
                    np.zeros(len(ids), dtype=np.int64) for ids in request.block_ids_by_group_np
                ]
                request.partial_save_gva_per_group = [0] * self.num_kv_cache_groups
                request.save_keys = []
                seen_requests.add(request.req_id)
            gvas = request.block_gvas_by_group_np[plan.group_id]
            for index, key in zip(plan.block_indices, plan.keys, strict=True):
                if key not in copied_keys:
                    gva = new_gvas_by_key[key] if key in new_gvas_by_key else self._allocated_gvas[key]
                    gvas[index] = gva
                    self._allocated_gvas[key] = gva
                    copied_keys.add(key)
                    if key in new_gvas_by_key:
                        request.save_keys.append(key)
            if plan.partial_key is not None and plan.partial_key not in copied_keys:
                key = plan.partial_key
                request.partial_save_gva_per_group[plan.group_id] = (
                    new_gvas_by_key[key] if key in new_gvas_by_key else self._allocated_gvas[key]
                )
                request.save_keys.append(key)
                copied_keys.add(key)
                self._allocated_gvas.pop(key, None)
            request.block_gvas_np = request.block_gvas_by_group_np[0]
            request.gva_block_offset = 0

    def _prepare_load_gvas(self, requests: list[ReqMeta]) -> None:
        """Fetch one deduplicated batch and acquire shared request read leases."""
        if not self.enabled:
            return
        plans = self._build_key_plans(requests, is_save=False)
        keys = list(
            dict.fromkeys(
                key for plan in plans for key in [*plan.keys, *([plan.partial_key] if plan.partial_key else [])]
            )
        )
        if not plans:
            return
        key_infos = self.m_store.batch_get_key_info(keys) if keys else []
        if len(key_infos) != len(keys):
            raise RuntimeError(
                "Layerwise load preparation returned unexpected key infos: "
                f"expected={len(keys)}, actual={len(key_infos)}"
            )
        gvas: dict[str, int] = {}
        for key, info in zip(keys, key_infos, strict=True):
            size = info.size()
            addresses = info.gva_list()
            if size and size > 0 and not addresses:
                raise RuntimeError(f"Layerwise load preparation returned invalid GVA metadata for {key}")
            gvas[key] = addresses[0] if size and size > 0 else 0
        valid_keys = [key for key in keys if gvas[key] > 0]
        partial_keys = {plan.partial_key for plan in plans if plan.partial_key is not None}
        # Serialize acquisition/registration with final release. A finishing
        # request must not remove a key just acquired by the next batch.
        with self._lease_lock:
            new_keys = [key for key in valid_keys if key not in self.load_lease_refcounts]
            lease_results = (
                list(self.m_store.batch_add_lease(new_keys, LAYERWISE_READ_LEASE_TTL_MS)) if new_keys else []
            )
            acquired = {key for key, result in zip(new_keys, lease_results) if result == 0}
            try:
                if len(lease_results) != len(new_keys):
                    raise RuntimeError(
                        "MemCache lease returned unexpected number of results: "
                        f"expected={len(new_keys)}, actual={len(lease_results)}"
                    )
                for index, key in enumerate(new_keys):
                    if key in partial_keys:
                        for _ in range(PARTIAL_LEASE_RETRY_COUNT):
                            if lease_results[index] != MEMCACHE_UNMATCHED_STATE:
                                break
                            time.sleep(PARTIAL_LEASE_RETRY_INTERVAL_S)
                            retry = self.m_store.batch_add_lease([key], LAYERWISE_READ_LEASE_TTL_MS)
                            if len(retry) != 1:
                                raise RuntimeError(
                                    f"MemCache partial lease retry returned unexpected number of results: {len(retry)}"
                                )
                            lease_results[index] = retry[0]
                    if lease_results[index] == 0:
                        acquired.add(key)
                    else:
                        gvas[key] = 0
                keys_by_request: dict[str, set[str]] = {}
                seen_requests: set[str] = set()
                for plan in plans:
                    request = plan.request
                    if request.req_id not in seen_requests:
                        request.load_block_gvas_by_group_np = [
                            np.zeros(len(ids), dtype=np.int64) for ids in request.block_ids_by_group_np
                        ]
                        request.partial_load_gva_per_group = [0] * self.num_kv_cache_groups
                        request.load_keys = []
                        seen_requests.add(request.req_id)
                    request_keys = keys_by_request.setdefault(request.req_id, set())
                    positions = list(zip(plan.block_indices, plan.keys, strict=True))
                    if plan.partial_key is not None:
                        positions.append((plan.partial_block_index, plan.partial_key))
                    invalid_blocks = []
                    for block_index, key in positions:
                        gva = gvas[key]
                        if gva <= 0:
                            invalid_blocks.append(int(plan.block_ids[block_index]))
                        else:
                            request_keys.add(key)
                            request.load_keys.append(key)
                        if key == plan.partial_key:
                            request.partial_load_gva_per_group[plan.group_id] = gva
                        else:
                            request.load_block_gvas_by_group_np[plan.group_id][block_index] = gva
                    if invalid_blocks:
                        if self.num_kv_cache_groups > 1:
                            raise RuntimeError(
                                "Layerwise multi-group KV load failed and cannot safely "
                                "fall back to per-block recomputation: "
                                f"request={request.req_id}, failed_blocks={invalid_blocks}"
                            )
                        self.record_invalid_blocks(invalid_blocks)
                    request.load_block_gvas_np = request.load_block_gvas_by_group_np[0]
                    request.load_gva_block_offset = 0
                self.register_load_leases(keys_by_request)
            except Exception:
                if acquired:
                    result = self.m_store.batch_remove_lease(sorted(acquired))
                    if result != 0:
                        logger.error("Failed to roll back layerwise load leases: result=%s", result)
                raise

    def register_load_leases(self, keys_by_request: dict[str, set[str]]) -> None:
        with self._lease_lock:
            for req_id, keys in keys_by_request.items():
                registered = self.load_lease_keys_by_request.setdefault(req_id, set())
                for key in keys - registered:
                    self.load_lease_refcounts[key] = self.load_lease_refcounts.get(key, 0) + 1
                registered.update(keys)

    def release_finished_load_leases(self, finished_req_ids: set[str]) -> None:
        if not finished_req_ids or not self.enabled:
            return
        keys_to_release: list[str] = []
        with self._lease_lock:
            for req_id in finished_req_ids:
                for key in self.load_lease_keys_by_request.pop(req_id, ()):
                    refcount = self.load_lease_refcounts[key] - 1
                    if refcount == 0:
                        del self.load_lease_refcounts[key]
                        keys_to_release.append(key)
                    else:
                        self.load_lease_refcounts[key] = refcount
            if keys_to_release:
                result = self.m_store.batch_remove_lease(keys_to_release)
                if result != 0:
                    logger.error(
                        "Failed to release %d layerwise load leases for %d finished requests: res=%d",
                        len(keys_to_release),
                        len(finished_req_ids),
                        result,
                    )
