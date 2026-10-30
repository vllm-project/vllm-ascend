from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import (
    Backend,
    require_aligned_batch_results,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.session_tracker import (
    LayerwiseSessionTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVTransferThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import (
    LayerBatchReqMeta,
    LayerBlockRange,
    LayerLoadTask,
    LayerRangeReqMeta,
    LayerTransferTask,
    SharedBlockData,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase, ReqMeta

_KVPOOL_RANGE_DEBUG_PREFIX = "[KVPOOL_RANGE_DEBUG]"


def _build_range_debug_payload(
    direction: str,
    layer_id: int,
    sizes: list[list[int]],
    object_offsets: list[list[int]],
    results: list[int],
) -> dict[str, Any]:
    nested_sizes = [[int(size) for size in key_sizes] for key_sizes in sizes]
    return {
        "event": "range",
        "direction": direction,
        "layer_id": int(layer_id),
        "key_count": len(results),
        "requested_bytes": [sum(key_sizes) for key_sizes in nested_sizes],
        "sizes": nested_sizes,
        "object_offsets": [[int(offset) for offset in key_offsets] for key_offsets in object_offsets],
        "results": [int(result) for result in results],
    }


def _emit_range_debug_event(
    direction: str,
    layer_id: int,
    sizes: list[list[int]],
    object_offsets: list[list[int]],
    results: list[int],
) -> None:
    try:
        if not get_ascend_config().kv_pool_range_debug:
            return
        payload = _build_range_debug_payload(direction, layer_id, sizes, object_offsets, results)
        logger.info("%s %s", _KVPOOL_RANGE_DEBUG_PREFIX, json.dumps(payload, separators=(",", ":")))
    except Exception:
        # Debug instrumentation must never affect the transfer path.
        pass


def _emit_commit_debug_event(layer_id: int, key_count: int, results: list[int]) -> None:
    try:
        if not get_ascend_config().kv_pool_range_debug:
            return
        payload = {
            "event": "commit",
            "layer_id": int(layer_id),
            "key_count": int(key_count),
            "results": [int(result) for result in results],
        }
        logger.info("%s %s", _KVPOOL_RANGE_DEBUG_PREFIX, json.dumps(payload, separators=(",", ":")))
    except Exception:
        pass


def _circular_shift(lst: list, offset: int) -> list:
    if not lst or offset == 0:
        return lst
    return lst[offset:] + lst[:offset]


@dataclass(frozen=True)
class _LayerRevokeTask:
    keys: tuple[str, ...]


class LayerBatchBuilder:
    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        page_size_bytes: int,
        num_layers: int,
        group_id: int = 0,
        layer_byte_offset: int = 0,
    ) -> None:
        self.page_size_bytes = page_size_bytes
        self.num_layers = num_layers
        self.group_id = group_id
        self.layer_byte_offset = layer_byte_offset
        self._block_len_np = np.asarray(token_database.group_block_len[group_id], dtype=np.int64)
        self._kv_caches_base_addr_np = np.asarray(
            token_database.group_kv_caches_base_addr[group_id],
            dtype=np.int64,
        )
        group_block_stride = token_database.group_block_stride.get(group_id, token_database.group_block_len[group_id])
        self._block_stride_np = np.asarray(group_block_stride, dtype=np.int64)
        layer_cache_entry_offsets = token_database.group_layer_cache_entry_offsets.get(group_id)
        if layer_cache_entry_offsets is None:
            caches_per_layer = max(1, self._block_len_np.shape[0] // max(1, num_layers))
            layer_cache_entry_offsets = [
                min(layer * caches_per_layer, self._block_len_np.shape[0]) for layer in range(num_layers + 1)
            ]
        self._layer_cache_entry_offsets_np = np.asarray(layer_cache_entry_offsets, dtype=np.int64)
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

    def _build_transfer_arrays(
        self,
        block_ids_arr: np.ndarray,
        base_gvas_arr: np.ndarray,
        layer_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_offset = int(self._layer_cache_entry_offsets_np[layer_id])
        end_offset = int(self._layer_cache_entry_offsets_np[layer_id + 1])
        layer_base_addrs = self._kv_caches_base_addr_np[base_offset:end_offset]
        layer_block_len = self._block_len_np[base_offset:end_offset]
        layer_block_stride = self._block_stride_np[base_offset:end_offset]
        # Per-cache inner offsets within one layer's page: [0, len0, len0+len1, ...].
        layer_inner_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(layer_block_len[:-1], dtype=np.int64))
        )
        rank_layer_offset = self.layer_byte_offset + int(self._block_len_np[:base_offset].sum())
        if base_gvas_arr.size > 0 and np.any(base_gvas_arr <= 0):
            # Zero GVAs here are expected: blocks already committed in the
            # pool or excluded by masks are intentionally not allocated.
            # Real batch_alloc failures are logged as errors by the caller.
            zero_count = int(np.sum(base_gvas_arr <= 0))
            logger.debug(
                "[KVPOOL] build_transfer layer=%d detected %d zero/negative base_gvas "
                "(base_gvas_sample=%s); these blocks will be skipped in batch_copy",
                layer_id,
                zero_count,
                base_gvas_arr[:5].tolist(),
            )
        logger.debug(
            "[KVPOOL] build_transfer layer=%d page_size=%d caches_per_layer=%d "
            "rank_layer_offset=%d layer_block_len=%s layer_inner_offsets=%s "
            "base_gvas=%s",
            layer_id,
            self.page_size_bytes,
            end_offset - base_offset,
            rank_layer_offset,
            layer_block_len.tolist(),
            layer_inner_offsets.tolist(),
            base_gvas_arr.tolist(),
        )

        addr_arr = layer_base_addrs[None, :] + block_ids_arr[:, None] * layer_block_stride[None, :]
        size_arr = np.broadcast_to(layer_block_len, addr_arr.shape)
        gvas_arr = base_gvas_arr[:, None] + rank_layer_offset + layer_inner_offsets[None, :]

        return (
            addr_arr.ravel(),
            size_arr.ravel(),
            gvas_arr.ravel(),
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
            # Zero GVAs are the designed sentinel for blocks that are not
            # transfer targets this step (already committed in the pool,
            # hit-prefix blocks, or masked-out blocks). Real batch_alloc
            # failures are logged as errors by _alloc_gvas_for_save.
            skip_count = int(np.sum(~valid_mask))
            logger.debug(
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
        """Build block-key rows shared by all backend layer ranges."""
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
        return self.build_addrs(shared, task.layer_idx_in_group)


class KVCacheStoreLayerSendingThreadBase(KVTransferThread):
    """Track pending layer saves by request ID for both layerwise paths."""

    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        dcp_size: int = 1,
        ready_event: threading.Event | None = None,
        name: str = "KVCacheStoreLayerSendingThreadBase",
    ):
        super().__init__(m_store, token_database, block_size, tp_rank, tp_size, dcp_size, ready_event, name=name)
        self.stored_requests: defaultdict[str, int] = defaultdict(int)

    def add_stored_request(self, req_id: str) -> None:
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str) -> int | None:
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1
                return self.stored_requests[req_id]
            return None

    def try_finish_and_delete_stored_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            if req_id in self.stored_requests and self.stored_requests[req_id] == 0:
                del self.stored_requests[req_id]
                return True
            return False


class KVCacheStoreKeyLayerSendingThread(KVCacheStoreLayerSendingThreadBase):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        put_step: int,
        ready_event: threading.Event,
        num_layers: int,
        layer_save_finished_events: list[threading.Event],
        sync_save_events: list[torch.npu.Event],
        layer_offset: int = 0,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreKeyLayerSendingThread",
        )
        self.final_layer_id = num_layers - 1
        self.layer_offset = layer_offset
        self.put_step = put_step
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events

    def build_cached_process_tokens(self, task: LayerTransferTask) -> dict[int, list[tuple[int, int, list]]] | None:
        """Pre-compute process_tokens results for all layers (Key path).

        Returns a dict mapping block_range index to a list of
        (start, end, key_all_layers) tuples, where key_all_layers is the
        result of key.split_layers().
        """
        if not task.block_ranges:
            return None

        group_block_size = self._get_block_size(0)
        cache: dict[int, list[tuple[int, int, list]]] = {}

        for br_idx, block_range in enumerate(task.block_ranges):
            request = block_range.request
            mask_num = request.save_start_token // group_block_size * group_block_size
            entries = []
            for start, end, key in self.token_database.process_tokens(
                request.save_end_token,
                request.block_hashes,
                mask_num,
            ):
                block_index = start // group_block_size
                if block_index < block_range.start_block or block_index >= block_range.end_block:
                    continue
                key_all = key.split_layers(self.final_layer_id + 1, self.layer_offset)
                entries.append((start, end, key_all))
            cache[br_idx] = entries

        return cache

    def _handle_request(  # type: ignore[override]
        self, transfer_tasks: list[LayerTransferTask]
    ):
        if len(transfer_tasks) == 0:
            self.request_queue.task_done()
            return
        if len(transfer_tasks) > 1:
            raise ValueError(f"Expected at most one layer transfer task, got {len(transfer_tasks)}")

        transfer_task = transfer_tasks[0]
        layer_id = transfer_task.layer_id
        key_list = []
        addr_list = []
        size_list = []
        req_ids = []
        is_last_chunks = []

        # Reuse pre-computed process_tokens results if available
        cached_tokens = transfer_task.cached_process_tokens

        for br_idx, block_range in enumerate(transfer_task.block_ranges):
            request = block_range.request
            req_ids.append(request.req_id)
            is_last_chunks.append(request.is_last_chunk)
            starts = []
            ends = []
            keys = []
            group_block_size = self._get_block_size(0)

            if cached_tokens is not None:
                # Fast path: reuse cached (start, end, key_all) tuples
                for start, end, key_all in cached_tokens[br_idx]:
                    block_index = start // group_block_size
                    if block_index < block_range.start_block or block_index >= block_range.end_block:
                        continue
                    starts.append(start)
                    ends.append(end)
                    keys.append(key_all[layer_id])
            else:
                mask_num = request.save_start_token // group_block_size * group_block_size
                for start, end, key in self.token_database.process_tokens(
                    request.save_end_token,
                    request.block_hashes,
                    mask_num,
                ):
                    block_index = start // group_block_size
                    if block_index < block_range.start_block or block_index >= block_range.end_block:
                        continue
                    starts.append(start)
                    ends.append(end)
                    keys.append(key.split_layers(self.final_layer_id + 1, self.layer_offset)[layer_id])

            if not self.dcp_size > 1:
                starts = starts[self.tp_rank % self.put_step :: self.put_step]
                ends = ends[self.tp_rank % self.put_step :: self.put_step]
                keys = keys[self.tp_rank % self.put_step :: self.put_step]

            for index, key in enumerate(keys):
                key_list.append(key.to_string())
                addr, size, _ = self.token_database.prepare_value_layer(
                    starts[index],
                    ends[index],
                    request.block_ids,
                    layer_id,
                )
                addr_list.append(addr)
                size_list.append(size)

        for req_id in req_ids:
            self.dec_stored_request(req_id)

        if key_list:
            missing_indices = self._get_missing_indices(key_list)
            keys_to_put = [key_list[index] for index in missing_indices]
            addrs_to_put = [addr_list[index] for index in missing_indices]
            sizes_to_put = [size_list[index] for index in missing_indices]
            if keys_to_put:
                self.sync_save_events[layer_id].synchronize()
                self.m_store.put(keys_to_put, addrs_to_put, sizes_to_put)

        if layer_id == self.final_layer_id:
            for req_id, is_last_chunk in zip(req_ids, is_last_chunks):
                if is_last_chunk and self.try_finish_and_delete_stored_request(req_id):
                    self.set_finished_request(req_id)

        assert not self.layer_save_finished_events[layer_id].is_set(), f"thread: {layer_id} save failed "
        logger.debug("Key-based layer save event set: layer %d", layer_id)
        self.layer_save_finished_events[layer_id].set()
        transfer_tasks.clear()
        self.request_queue.task_done()


class KVCacheStoreKeyLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        ready_event: threading.Event,
        get_event: threading.Event,
        layer_load_finished_events: list[threading.Event],
        layer_save_finished_events: list[threading.Event],
        num_layers: int,
        layer_offset: int = 0,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreKeyLayerRecvingThread",
        )
        self.get_event = get_event
        self.layer_load_finished_events = layer_load_finished_events
        self.layer_save_finished_events = layer_save_finished_events
        self.final_layer_id = num_layers - 1
        self.layer_offset = layer_offset

    def _wait_for_save(self, layer_id: int) -> bool:
        if not self._wait_for_event(self.layer_save_finished_events[layer_id]):
            return False
        logger.debug("Key-based layer save event cleared: layer %d", layer_id)
        self.layer_save_finished_events[layer_id].clear()
        return True

    def _handle_request(  # type: ignore[override]
        self, data: LayerLoadTask
    ):
        wait_for_save = data.wait_for_save_layer
        layer_id = data.layer_id
        if wait_for_save is not None and not self._wait_for_save(wait_for_save):
            data.transfer_tasks.clear()

        if data.attention_start_gate is not None:
            if not self._wait_for_event(data.attention_start_gate):
                data.transfer_tasks.clear()

        key_list = []
        addr_list = []
        size_list = []
        req_ids = []
        is_last_chunks = []
        if len(data.transfer_tasks) > 1:
            raise ValueError(f"Expected at most one layer transfer task, got {len(data.transfer_tasks)}")
        if data.transfer_tasks:
            transfer_task = data.transfer_tasks[0]
            for block_range in transfer_task.block_ranges:
                request = block_range.request
                req_ids.append(request.req_id)
                is_last_chunks.append(request.is_last_chunk)
                for block_index in range(block_range.start_block, block_range.end_block):
                    if block_index >= len(request.block_hashes):
                        continue
                    block_hash = request.block_hashes[block_index]
                    chunk_hash = block_hash if isinstance(block_hash, str) else block_hash.hex()
                    key = self.token_database._make_key_by_hash(
                        chunk_hash,
                    ).split_layers(self.final_layer_id + 1, self.layer_offset)[layer_id]
                    group_block_size = self._get_block_size(0)
                    start = block_index * group_block_size
                    end = start + group_block_size
                    addr, size, _ = self.token_database.prepare_value_layer(
                        start,
                        end,
                        request.block_ids,
                        layer_id,
                    )
                    key_list.append(key.to_string())
                    addr_list.append(addr)
                    size_list.append(size)

        if key_list:
            shift = (self.tp_rank * len(key_list)) // self.tp_size
            key_list_c = _circular_shift(key_list, shift)
            addr_list_c = _circular_shift(addr_list, shift)
            size_list_c = _circular_shift(size_list, shift)
            self.m_store.get(key_list_c, addr_list_c, size_list_c)

        if layer_id == self.final_layer_id:
            for req_id, is_last_chunk in zip(req_ids, is_last_chunks):
                if is_last_chunk:
                    self.set_finished_request(req_id)

        assert not self.layer_load_finished_events[layer_id].is_set(), f"thread: {layer_id} load failed "
        logger.debug("Key-based layer load event set: layer %d", layer_id)
        self.layer_load_finished_events[layer_id].set()
        data.transfer_tasks.clear()
        self.request_queue.task_done()
        self.get_event.set()


class KVCacheStoreLayerSendingThread(KVCacheStoreLayerSendingThreadBase):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        page_size_bytes: int,
        ready_event: threading.Event,
        num_layers: int,
        layer_save_finished_events: list[threading.Event],
        sync_save_events: list[torch.npu.Event],
        max_transfer_blocks: int = 0,
        max_transfer_bytes: int = 0,
        group_builders: list[LayerBatchBuilder] | None = None,
        put_started_keys: set[str] | None = None,
        put_started_keys_lock: threading.Lock | None = None,
        session_tracker: LayerwiseSessionTracker | None = None,
        raise_on_transfer_failure: bool = True,
        invalid_block_ids: set[int] | None = None,
        invalid_block_ids_lock: threading.Lock | None = None,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreLayerSendingThread",
        )
        self.final_layer_id = num_layers - 1
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes
        self.write_results: dict[str, int] = {}
        self._put_started_keys = put_started_keys if put_started_keys is not None else set()
        self._put_started_keys_lock = put_started_keys_lock or threading.Lock()
        self._session_tracker = session_tracker
        self.raise_on_transfer_failure = raise_on_transfer_failure
        self._invalid_block_ids = invalid_block_ids if invalid_block_ids is not None else set()
        self._invalid_block_ids_lock = invalid_block_ids_lock or threading.Lock()
        self._active_put_keys: set[str] | None = None
        self._group_put_keys: dict[int, set[str] | None] = {}
        self.group_builders: list[LayerBatchBuilder] | None = group_builders
        if group_builders is not None:
            self.layer_batch_builder = group_builders[0]
        else:
            self.layer_batch_builder = LayerBatchBuilder(
                token_database,
                page_size_bytes,
                num_layers,
                group_id=0,
            )

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def build_shared_data(self, task: LayerTransferTask) -> SharedBlockData | None:
        """Pre-compute block rows shared by every layer in the task."""
        if self.group_builders is not None:
            builder = self.group_builders[task.group_id]
        else:
            builder = self.layer_batch_builder
        return builder.build_shared(task, is_save=True)

    def add_revoke_request(self, keys: list[str]) -> None:
        deduplicated_keys = tuple(dict.fromkeys(keys))
        if deduplicated_keys:
            self.add_request(_LayerRevokeTask(deduplicated_keys))

    def _remove_started_keys(self, keys: list[str]) -> None:
        with self._put_started_keys_lock:
            self._put_started_keys.difference_update(keys)

    def _revoke_range_keys(self, keys: list[str]) -> None:
        keys = list(dict.fromkeys(keys))
        if not keys:
            return
        try:
            results = require_aligned_batch_results("batch_revoke", keys, self.m_store.batch_revoke(keys))
            if any(result != 0 for result in results):
                logger.error("Layerwise backend revoke failed keys=%s results=%s", keys, results)
        except Exception as exc:
            logger.error("Layerwise backend revoke raised keys=%s error=%s", keys, exc)
        finally:
            self._remove_started_keys(keys)
            if self._session_tracker is not None:
                self._session_tracker.revoke_put_keys(keys)

    def _handle_range_request(self, req_meta: LayerRangeReqMeta, final_group_layer: bool | None = None) -> None:
        layer_id = req_meta.layer_id
        if self._active_put_keys is None or layer_id == 0:
            self._active_put_keys = set(req_meta.keys)
        assert self._active_put_keys is not None
        active_indices = [index for index, key in enumerate(req_meta.keys) if key in self._active_put_keys]
        active_keys = [req_meta.keys[index] for index in active_indices]
        if active_keys:
            self.sync_save_events[layer_id].synchronize()
            active_buffers = [req_meta.all_buffers[index] for index in active_indices]
            active_sizes = [req_meta.all_sizes[index] for index in active_indices]
            active_offsets = [req_meta.all_offsets[index] for index in active_indices]
            results: list[int] = []
            for keys, buffers, sizes, offsets in self._range_transfer_batches(
                active_keys,
                active_buffers,
                active_sizes,
                active_offsets,
                self.max_transfer_blocks,
                self.max_transfer_bytes,
            ):
                results.extend(
                    require_aligned_batch_results(
                        "batch_copy_put",
                        keys,
                        self.m_store.batch_copy_put(keys, buffers, sizes, offsets),
                    )
                )
            _emit_range_debug_event("save", layer_id, active_sizes, active_offsets, results)
            failed_keys = [key for key, result in zip(active_keys, results, strict=True) if result < 0]
            if failed_keys:
                self._revoke_range_keys(failed_keys)
                self._active_put_keys.difference_update(failed_keys)

        if not (layer_id == self.final_layer_id if final_group_layer is None else final_group_layer):
            return
        active_keys = [key for key in req_meta.keys if key in self._active_put_keys]
        if active_keys:
            try:
                commit_results = require_aligned_batch_results(
                    "batch_commit", active_keys, self.m_store.batch_commit(active_keys)
                )
                _emit_commit_debug_event(layer_id, len(active_keys), commit_results)
            except Exception:
                self._revoke_range_keys(active_keys)
                raise
            failed_keys = [key for key, result in zip(active_keys, commit_results, strict=True) if result != 0]
            if failed_keys:
                self._revoke_range_keys(failed_keys)
            committed_keys = [key for key, result in zip(active_keys, commit_results, strict=True) if result == 0]
            if self._session_tracker is not None:
                self._session_tracker.commit_put_keys(committed_keys)
            self._remove_started_keys(active_keys)
        self._active_put_keys = None

    def _handle_range_layer_tasks(self, transfer_tasks: list[LayerTransferTask]) -> None:
        layer_id = transfer_tasks[0].layer_id
        try:
            for task in transfer_tasks:
                shared = task.shared_block_data
                if shared is None:
                    raise RuntimeError("Layerwise range save requires shared block metadata")
                builder = self.group_builders[task.group_id] if self.group_builders else self.layer_batch_builder
                req_meta = builder.build_addrs(shared, task.layer_idx_in_group)
                if not isinstance(req_meta, LayerRangeReqMeta):
                    raise TypeError(f"Expected layerwise range metadata, got {type(req_meta).__name__}")
                req_meta.layer_id = layer_id
                self._active_put_keys = self._group_put_keys.get(task.group_id)
                try:
                    self._handle_range_request(req_meta, task.final_group_layer or layer_id == self.final_layer_id)
                finally:
                    self._group_put_keys[task.group_id] = self._active_put_keys
                for req_id in req_meta.req_ids:
                    self.dec_stored_request(req_id)
                    if self.try_finish_and_delete_stored_request(req_id):
                        self.set_finished_request(req_id)
        except Exception as exc:
            self._fatal_error = exc
            # All started objects belong to the failed forward step, including
            # groups whose first physical layer has not been visited yet.
            with self._put_started_keys_lock:
                keys_to_revoke = list(self._put_started_keys)
            self._revoke_range_keys(keys_to_revoke)
            self._group_put_keys.clear()
            raise
        finally:
            self.layer_save_finished_events[layer_id].set()
            transfer_tasks.clear()
            self.request_queue.task_done()

    def _handle_request(  # type: ignore[override]
        self, request: list[LayerTransferTask] | _LayerRevokeTask
    ):
        if isinstance(request, _LayerRevokeTask):
            try:
                self._revoke_range_keys(list(request.keys))
            finally:
                self.request_queue.task_done()
            return
        if not request:
            self.request_queue.task_done()
            return
        if request[0].use_key_major_ranges:
            self._handle_range_layer_tasks(request)
            return
        layer_id = request[0].layer_id
        try:
            if not self._copy_gva_layers(request):
                self.save_disabled.set()
                if self.raise_on_transfer_failure:
                    self._mark_failed_requests([br.request for task in request for br in task.block_ranges])
                # WRITE_FAIL removes the key even if batch_alloc returned
                # an existing allocation. Leave unpublished writes for eviction
                # after their lease expires instead of deleting shared keys.
                self.write_results.clear()
                for task in request:
                    task.write_finish_keys.clear()
        except Exception as exc:
            self._fatal_error = exc
            raise
        finally:
            for task in request:
                req_ids = [br.request.req_id for br in task.block_ranges]
                if not req_ids and task.shared_block_data is not None:
                    req_ids = task.shared_block_data.req_ids
                for req_id in req_ids:
                    self.dec_stored_request(req_id)
                    if self.try_finish_and_delete_stored_request(req_id):
                        self.set_finished_request(req_id)
            self.layer_save_finished_events[layer_id].set()
            request.clear()
            self.request_queue.task_done()

    def _copy_gva_layers(self, transfer_tasks: list[LayerTransferTask]) -> bool:
        if self.save_disabled.is_set():
            return False
        physical_layer = transfer_tasks[0].layer_id
        has_any_save = False
        all_gvas = []
        all_addrs = []
        all_sizes = []
        all_save_keys: list[str] = []
        write_finish_keys: list[str] = []
        for task in transfer_tasks:
            shared = task.shared_block_data
            if shared is None:
                continue
            has_any_save = True
            builder = self.group_builders[task.group_id] if self.group_builders else self.layer_batch_builder
            req_meta = builder.build_addrs(shared, task.layer_idx_in_group)
            if not isinstance(req_meta, LayerBatchReqMeta):
                raise TypeError(f"Expected GVA layer metadata, got {type(req_meta).__name__}")
            all_save_keys.extend(shared.save_keys)
            write_finish_keys.extend(task.write_finish_keys)
            all_gvas.append(req_meta.gvas_array)
            all_addrs.append(req_meta.addr_array)
            all_sizes.append(req_meta.size_array)
        if has_any_save:
            self.sync_save_events[physical_layer].synchronize()
            gvas_array = np.concatenate(all_gvas) if len(all_gvas) > 1 else all_gvas[0]
            addr_array = np.concatenate(all_addrs) if len(all_addrs) > 1 else all_addrs[0]
            size_array = np.concatenate(all_sizes) if len(all_sizes) > 1 else all_sizes[0]
            res = self._batch_copy_with_limits(
                gvas_array,
                addr_array,
                size_array,
                0,
                self.max_transfer_blocks,
                self.max_transfer_bytes,
            )
            if res != 0:
                return False
            if self.save_disabled.is_set():
                return False
            if all_save_keys:
                save_keys = list(dict.fromkeys(all_save_keys))
                for key in save_keys:
                    self.write_results[key] = self.write_results.get(key, 0) or res
            if write_finish_keys:
                finish_keys = list(dict.fromkeys(write_finish_keys))
                if any(self.write_results.pop(key, None) != 0 for key in finish_keys):
                    logger.error("Memcache publication lacks successful copies: keys=%s", finish_keys)
                    return False
                # A failed publication response does not prove WRITE_OK failed
                # remotely. Never compensate it with WRITE_FAIL (key removal).
                for task in transfer_tasks:
                    task.write_finish_keys.clear()
                finish_results = self.m_store.batch_write_finish(finish_keys, [0] * len(finish_keys))
                if (
                    finish_results is None
                    or len(finish_results) != len(finish_keys)
                    or any(result != 0 for result in finish_results)
                ):
                    logger.error("Memcache write publication failed: keys=%s results=%s", finish_keys, finish_results)
                    return False
        return True


class KVCacheStoreLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        page_size_bytes: int,
        ready_event: threading.Event,
        get_event: threading.Event,
        layer_load_finished_events: list[threading.Event],
        layer_save_finished_events: list[threading.Event],
        sync_save_events: list[torch.npu.Event],
        num_layers: int,
        h2d_stagger_us: int = 0,
        max_transfer_blocks: int = 0,
        max_transfer_bytes: int = 0,
        group_builders: list[LayerBatchBuilder] | None = None,
        external_slot_release_waiter: Callable[[int], None] | None = None,
        save_failure_checker: Callable[[], None] | None = None,
        invalid_block_ids: set[int] | None = None,
        invalid_block_ids_lock: threading.Lock | None = None,
        load_abort_event: threading.Event | None = None,
        raise_on_transfer_failure: bool = True,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreLayerRecvingThread",
        )
        self.get_event = get_event
        self.layer_load_finished_events = layer_load_finished_events
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events
        self.final_layer_id = num_layers - 1
        self.h2d_stagger_us = h2d_stagger_us
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes
        self.external_slot_release_waiter = external_slot_release_waiter
        self.save_failure_checker = save_failure_checker
        self.raise_on_transfer_failure = raise_on_transfer_failure
        self._invalid_block_ids = invalid_block_ids if invalid_block_ids is not None else set()
        self._invalid_block_ids_lock = invalid_block_ids_lock or threading.Lock()
        self._load_abort_event = load_abort_event or threading.Event()
        self._active_load_indices: set[int] | None = None
        self._group_load_indices: dict[int, set[int] | None] = {}
        self.group_builders: list[LayerBatchBuilder] | None = group_builders
        if group_builders is not None:
            self.layer_batch_builder = group_builders[0]
        else:
            self.layer_batch_builder = LayerBatchBuilder(
                token_database,
                page_size_bytes,
                num_layers,
                group_id=0,
            )

    def build_shared_data(self, task: LayerTransferTask) -> SharedBlockData | None:
        """Pre-compute block rows shared by every layer in the task."""
        if self.group_builders is not None:
            builder = self.group_builders[task.group_id]
        else:
            builder = self.layer_batch_builder
        return builder.build_shared(task, is_save=False)

    def _mark_invalid_transfer_task_blocks(self, transfer_tasks: list[LayerTransferTask]) -> None:
        block_ids: set[int] = set()
        for task in transfer_tasks:
            for block_range in task.block_ranges:
                request_block_ids = (
                    block_range.request.block_ids_by_group[task.group_id]
                    if task.group_id < len(block_range.request.block_ids_by_group)
                    else block_range.request.block_ids
                )
                start = max(0, block_range.start_block)
                end = min(block_range.end_block, len(request_block_ids))
                block_ids.update(request_block_ids[start:end])
                partial_index = block_range.partial_block_index
                if partial_index is not None and 0 <= partial_index < len(request_block_ids):
                    block_ids.add(request_block_ids[partial_index])
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(block_ids)

    def _mark_invalid_range_indices(self, req_meta: LayerRangeReqMeta, indices: list[int]) -> None:
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(req_meta.block_ids[index] for index in indices)

    def _handle_range_request(self, req_meta: LayerRangeReqMeta, shared: SharedBlockData) -> None:
        layer_id = req_meta.layer_id
        if self._active_load_indices is None or layer_id == 0:
            self._active_load_indices = set(range(len(req_meta.keys)))
        assert self._active_load_indices is not None
        # Read the abort flag once: this comprehension covers every key of the
        # layer, so an is_set() per key is pure overhead in the receive thread.
        aborted = self._load_abort_event.is_set()
        active_indices = [
            index for index in range(len(req_meta.keys)) if not aborted and index in self._active_load_indices
        ]
        active_keys = [req_meta.keys[index] for index in active_indices]
        if active_keys:
            self._stagger_h2d_submit(layer_id)
            active_buffers = [req_meta.all_buffers[index] for index in active_indices]
            active_sizes = [req_meta.all_sizes[index] for index in active_indices]
            active_offsets = [req_meta.all_offsets[index] for index in active_indices]
            results: list[int] = []
            for keys, buffers, sizes, offsets in self._range_transfer_batches(
                active_keys,
                active_buffers,
                active_sizes,
                active_offsets,
                self.max_transfer_blocks,
                self.max_transfer_bytes,
            ):
                results.extend(
                    require_aligned_batch_results(
                        "batch_copy_get",
                        keys,
                        self.m_store.batch_copy_get(keys, buffers, sizes, offsets),
                    )
                )
            _emit_range_debug_event("load", layer_id, active_sizes, active_offsets, results)
            failed_indices = [index for index, result in zip(active_indices, results, strict=True) if result < 0]
            if failed_indices:
                self._mark_invalid_range_indices(req_meta, failed_indices)
                self._active_load_indices.difference_update(failed_indices)

        if layer_id == self.final_layer_id:
            for req_id, is_last_chunk in zip(req_meta.req_ids, shared.is_last_chunks, strict=True):
                if is_last_chunk:
                    self.set_finished_request(req_id)
            self._active_load_indices = None

    def _handle_range_layer_task(self, data: LayerLoadTask) -> None:
        layer_id = data.layer_id
        try:
            wait_for_save = data.wait_for_save_layer
            if wait_for_save is not None:
                if not self._wait_for_event(self.layer_save_finished_events[wait_for_save], self.save_failure_checker):
                    return
                if self.save_failure_checker is not None:
                    self.save_failure_checker()
                self.sync_save_events[wait_for_save].synchronize()
                self.layer_save_finished_events[wait_for_save].clear()

            if data.attention_start_gate is not None:
                if not self._wait_for_event(data.attention_start_gate):
                    return
            if self.external_slot_release_waiter is not None:
                self.external_slot_release_waiter(layer_id)
            for task in data.transfer_tasks:
                shared = task.shared_block_data
                if shared is None:
                    raise RuntimeError("Layerwise range load requires shared block metadata")
                builder = self.group_builders[task.group_id] if self.group_builders else self.layer_batch_builder
                req_meta = builder.build_addrs(shared, task.layer_idx_in_group)
                if not isinstance(req_meta, LayerRangeReqMeta):
                    raise TypeError(f"Expected layerwise range metadata, got {type(req_meta).__name__}")
                req_meta.layer_id = layer_id
                self._active_load_indices = self._group_load_indices.get(task.group_id)
                self._handle_range_request(req_meta, shared)
                self._group_load_indices[task.group_id] = None if task.final_group_layer else self._active_load_indices
        except Exception as exc:
            self._fatal_error = exc
            if self._active_load_indices is not None:
                self._active_load_indices.clear()
            self._mark_invalid_transfer_task_blocks(data.transfer_tasks)
            self._load_abort_event.set()
            raise
        finally:
            if not self.layer_load_finished_events[layer_id].is_set():
                self.layer_load_finished_events[layer_id].set()
            self.request_queue.task_done()
            self.get_event.set()

    def _get_h2d_stagger_delay_us(self, layer_id: int) -> int:
        if self.h2d_stagger_us <= 0:
            return 0
        slot = (self.tp_rank + layer_id) % self.tp_size
        return slot * self.h2d_stagger_us

    def _stagger_h2d_submit(self, layer_id: int) -> None:
        delay_us = self._get_h2d_stagger_delay_us(layer_id)
        if delay_us <= 0:
            return
        deadline = time.perf_counter() + delay_us / 1_000_000
        while time.perf_counter() < deadline:
            pass

    def _handle_request(  # type: ignore[override]
        self, data: LayerLoadTask
    ):
        if data.transfer_tasks and data.transfer_tasks[0].use_key_major_ranges:
            self._handle_range_layer_task(data)
            return
        try:
            if not self._copy_gva_layers(data):
                self.load_disabled.set()
                self._mark_failed_requests([br.request for task in data.transfer_tasks for br in task.block_ranges])
        except Exception as exc:
            self._fatal_error = exc
            raise
        finally:
            self.layer_load_finished_events[data.layer_id].set()
            self.request_queue.task_done()
            self.get_event.set()

    def _copy_gva_layers(self, data: LayerLoadTask) -> bool:
        # Layerwise threads only run when the worker-side
        # protocol gate is on; every store call below sits on the Backend ABC.
        wait_for_save = data.wait_for_save_layer
        transfer_tasks = data.transfer_tasks
        layer_id = data.layer_id
        attention_start_gate = data.attention_start_gate

        if wait_for_save is not None:
            if not self._wait_for_event(self.layer_save_finished_events[wait_for_save], self.save_failure_checker):
                return False
            if self.save_failure_checker is not None:
                self.save_failure_checker()
            # Non-saving TP ranks have no D2H task to synchronize the event.
            # Their CPU save-finished signal only means the event was recorded;
            # wait for the NPU work before reusing the local HBM buffer.
            self.sync_save_events[wait_for_save].synchronize()
            logger.debug("Layer save event cleared: layer %d", wait_for_save)
            self.layer_save_finished_events[wait_for_save].clear()

        if self.load_disabled.is_set():
            if self.external_slot_release_waiter is not None:
                self.external_slot_release_waiter(layer_id)
            return False

        if len(transfer_tasks) == 0:
            if self.external_slot_release_waiter is not None:
                self.external_slot_release_waiter(layer_id)
            assert not self.layer_load_finished_events[layer_id].is_set()
            logger.debug("Layer load event set: layer %d", layer_id)
            return True

        # Build req_meta for all tasks first; if all are None, early return.
        task_metas: list[tuple[LayerTransferTask, LayerBatchReqMeta]] = []
        for task in transfer_tasks:
            shared = task.shared_block_data
            builder = self.group_builders[task.group_id] if self.group_builders else self.layer_batch_builder
            if shared is not None:
                built_meta = builder.build_addrs(shared, task.layer_idx_in_group)
                if not isinstance(built_meta, LayerBatchReqMeta):
                    raise TypeError(f"Expected GVA layer metadata, got {type(built_meta).__name__}")
                req_meta: LayerBatchReqMeta | None = built_meta
            else:
                candidate_meta = builder.build(task, is_save=False)
                if candidate_meta is not None and not isinstance(candidate_meta, LayerBatchReqMeta):
                    raise TypeError(f"Expected GVA layer metadata, got {type(candidate_meta).__name__}")
                req_meta = candidate_meta
            if req_meta is not None:
                task_metas.append((task, req_meta))

        if not task_metas:
            if self.external_slot_release_waiter is not None:
                self.external_slot_release_waiter(layer_id)
            assert not self.layer_load_finished_events[layer_id].is_set()
            logger.debug("Layer load event set: layer %d", layer_id)
            return True

        if attention_start_gate is not None:
            if not self._wait_for_event(attention_start_gate):
                return False

        all_req_ids: set[str] = set()
        last_chunk_req_ids: set[str] = set()
        all_gvas = []
        all_addrs = []
        all_sizes = []
        for task, req_meta in task_metas:
            for req_id, is_last_chunk in zip(req_meta.req_ids, req_meta.is_last_chunks):
                all_req_ids.add(req_id)
                if is_last_chunk:
                    last_chunk_req_ids.add(req_id)
            all_gvas.append(req_meta.gvas_array)
            all_addrs.append(req_meta.addr_array)
            all_sizes.append(req_meta.size_array)

        self._stagger_h2d_submit(layer_id)
        gvas_array = np.concatenate(all_gvas) if len(all_gvas) > 1 else all_gvas[0]
        addr_array = np.concatenate(all_addrs) if len(all_addrs) > 1 else all_addrs[0]
        size_array = np.concatenate(all_sizes) if len(all_sizes) > 1 else all_sizes[0]
        if self.external_slot_release_waiter is not None:
            self.external_slot_release_waiter(layer_id)
        res = self._batch_copy_with_limits(
            gvas_array,
            addr_array,
            size_array,
            1,
            self.max_transfer_blocks,
            self.max_transfer_bytes,
        )
        if layer_id <= 2 or res != 0:
            logger.debug(
                "load_thread: layer=%d groups=%d blocks=%d res=%d",
                layer_id,
                len(all_gvas),
                len(gvas_array),
                res,
            )
        if res != 0:
            return False

        if layer_id == self.final_layer_id:
            for req_id in all_req_ids:
                if req_id in last_chunk_req_ids:
                    self.set_finished_request(req_id)
        assert not self.layer_load_finished_events[layer_id].is_set(), f"thread: {layer_id} load failed "
        logger.debug("Layer load event set: layer %d", layer_id)
        # transfer_tasks aliases KVPoolWorker.layer_load_tasks[layer_id]. Do
        # not mutate the worker-owned list from this asynchronous thread. The
        # worker replaces all per-layer lists at the beginning of every step.
        return True
