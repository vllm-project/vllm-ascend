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
"""Layerwise GVA transfer threads (memcache backend).

This module centralizes the GVA layerwise send/receive transfer threads and
their batch builder that used to live inside ``kv_transfer.py``. It is a
behavior-preserving relocation: thread lifecycle, event ordering, and log
messages are kept byte-for-byte.

Ownership:
- :class:`LayerBatchBuilder`: shared block-data precompute and per-layer
  address/GVA array construction.
- :class:`KVCacheStoreLayerSendingThread` / :class:`KVCacheStoreLayerRecvingThread`:
  the GVA layerwise transfer threads.
- :class:`GVALayerwiseThreadContext` + factory functions: the wiring surface
  used by ``KVPoolWorker`` to construct the threads without knowing their
  per-thread parameter lists.

Dependency direction (acyclic):
``kv_transfer.py (generic base) <- import <- gva_threads.py <- import <- pool_worker.py``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import (
    Backend,
    GVALayerwiseCapable,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVTransferThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    LayerBatchReqMeta,
    LayerBlockRange,
    LayerLoadTask,
    LayerTransferTask,
    SharedBlockData,
)


class LayerBatchBuilder:
    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        page_size_bytes: int,
        num_layers: int,
        group_id: int = 0,
    ) -> None:
        self.page_size_bytes = page_size_bytes
        self.group_id = group_id
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
        rank_layer_offset = int(self._block_len_np[:base_offset].sum())
        if base_gvas_arr.size > 0 and np.any(base_gvas_arr <= 0):
            zero_count = int(np.sum(base_gvas_arr <= 0))
            logger.warning(
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

    def build_addrs(
        self,
        shared: SharedBlockData,
        layer_id: int,
    ) -> LayerBatchReqMeta:
        """Compute per-layer addresses from pre-computed shared block data."""
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

    def build(self, task: LayerTransferTask, is_save: bool = True) -> LayerBatchReqMeta | None:
        """Full build: shared data + per-layer addresses (backward compat)."""
        shared = self.build_shared(task, is_save)
        if shared is None:
            return None
        return self.build_addrs(shared, task.layer_idx_in_group)


class _GVALayerTransferThreadBase(KVTransferThread):
    """Shared plumbing for the GVA layerwise threads: transfer packet
    splitting and the rate-limited batch_copy loop."""

    def __init__(
        self,
        m_store: GVALayerwiseCapable,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        ready_event: threading.Event,
        name: str,
        max_transfer_blocks: int = 0,
        max_transfer_bytes: int = 0,
    ):
        # GVA threads only ever run on backends that are both a full Backend
        # and GVALayerwiseCapable (today: MemcacheBackend).
        assert isinstance(m_store, Backend)
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name=name,
        )
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes

    @property
    def _gva_store(self) -> GVALayerwiseCapable:
        assert isinstance(self.m_store, GVALayerwiseCapable)
        return self.m_store

    @staticmethod
    def _split_transfer_packets(
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        max_transfer_bytes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if max_transfer_bytes <= 0:
            return gvas, addrs, sizes

        split_counts: np.ndarray = (sizes + max_transfer_bytes - 1) // max_transfer_bytes
        total_splits = int(split_counts.sum())
        if total_splits == sizes.shape[0]:
            return gvas, addrs, sizes

        split_indices: np.ndarray = np.arange(int(split_counts.max()), dtype=np.int64)
        split_mask = split_indices[:, None] < split_counts[None, :]
        entry_indices = np.broadcast_to(
            np.arange(sizes.shape[0], dtype=np.int64),
            split_mask.shape,
        )[split_mask]
        transfer_offsets = np.broadcast_to(
            split_indices[:, None] * max_transfer_bytes,
            split_mask.shape,
        )[split_mask]

        split_gvas = gvas[entry_indices] + transfer_offsets
        split_addrs = addrs[entry_indices] + transfer_offsets
        split_sizes = np.minimum(
            max_transfer_bytes,
            sizes[entry_indices] - transfer_offsets,
        )
        return split_gvas, split_addrs, split_sizes

    def _batch_copy_with_limits(
        self,
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        direction: int,
        max_transfer_blocks: int,
        max_transfer_bytes: int,
    ) -> int:
        if len(gvas) == 0:
            return 0

        # direction: 0/SMEMB_COPY_L2G = save (write), 1/SMEMB_COPY_G2L = load (read)
        dir_name = "save(L2G)" if direction == 0 else "load(G2L)" if direction == 1 else f"dir{direction}"
        logger.debug(
            "[KVPOOL] batch_copy %s gvas=%d total_bytes=%d",
            dir_name,
            len(gvas),
            int(sizes.sum()) if len(sizes) else 0,
        )

        max_transfer_addrs = 0
        if max_transfer_blocks > 0:
            max_transfer_addrs = max_transfer_blocks * self.num_addrs_per_block
        if max_transfer_addrs <= 0:
            max_transfer_addrs = len(gvas)

        gva_store = self._gva_store
        for start in range(0, len(gvas), max_transfer_addrs):
            end = start + max_transfer_addrs
            split_gvas, split_addrs, split_sizes = self._split_transfer_packets(
                gvas[start:end],
                addrs[start:end],
                sizes[start:end],
                max_transfer_bytes,
            )
            logger.debug(
                "[KVPOOL] batch_copy %s split_gvas=%s split_sizes=%s",
                dir_name,
                split_gvas.tolist(),
                split_sizes.tolist(),
            )
            res = gva_store.batch_copy(
                split_gvas.tolist(),
                split_addrs.tolist(),
                split_sizes.tolist(),
                direction,
            )
            if res != 0:
                logger.error("[KVPOOL] batch_copy %s FAILED res=%d", dir_name, res)
                return res
        return 0


class KVCacheStoreLayerSendingThread(_GVALayerTransferThreadBase):
    def __init__(
        self,
        m_store: GVALayerwiseCapable,
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
            max_transfer_blocks=max_transfer_blocks,
            max_transfer_bytes=max_transfer_bytes,
        )
        self.final_layer_id = num_layers - 1
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events
        self.write_results: dict[str, int] = {}
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
        """Pre-compute shared block data for all layers (GVA path)."""
        if self.group_builders is not None:
            builder = self.group_builders[task.group_id]
        else:
            builder = self.layer_batch_builder
        return builder.build_shared(task, is_save=True)

    def _handle_request(  # type: ignore[override]
        self, transfer_tasks: list[LayerTransferTask]
    ):
        if len(transfer_tasks) == 0:
            self.request_queue.task_done()
            return
        physical_layer = transfer_tasks[0].layer_id
        has_any_save = False
        all_gvas = []
        all_addrs = []
        all_sizes = []
        all_req_ids = []
        all_save_keys: list[str] = []
        write_finish_keys: list[str] = []
        for task in transfer_tasks:
            shared = task.shared_block_data
            if shared is None:
                continue
            has_any_save = True
            builder = self.group_builders[task.group_id] if self.group_builders else self.layer_batch_builder
            req_meta = builder.build_addrs(shared, task.layer_idx_in_group)
            for req_id in req_meta.req_ids:
                self.dec_stored_request(req_id)
                all_req_ids.append(req_id)
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
                raise RuntimeError(f"Layerwise {physical_layer} save batch_copy failed with return code {res}")
            if all_save_keys:
                save_keys = list(dict.fromkeys(all_save_keys))
                for key in save_keys:
                    self.write_results[key] = self.write_results.get(key, 0) or res
            if write_finish_keys:
                finish_keys = list(dict.fromkeys(write_finish_keys))
                results = [self.write_results.pop(key) for key in finish_keys]
                finish_results = self._gva_store.batch_write_finish(finish_keys, results)
                if len(finish_results) != len(finish_keys) or any(result != 0 for result in finish_results):
                    raise RuntimeError(
                        f"Layerwise save batch_write_finish failed: "
                        f"expected={len(finish_keys)}, results={finish_results}"
                    )
            for req_id in all_req_ids:
                if self.try_finish_and_delete_stored_request(req_id):
                    self.set_finished_request(req_id)
        if not has_any_save:
            assert not self.layer_save_finished_events[physical_layer].is_set(), (
                f"thread: {physical_layer} save failed "
            )
            logger.debug("Layer save event set: layer %d", physical_layer)
            self.layer_save_finished_events[physical_layer].set()
            transfer_tasks.clear()
            self.request_queue.task_done()
            return
        assert not self.layer_save_finished_events[physical_layer].is_set(), f"thread: {physical_layer} save failed "
        logger.debug("Layer save event set: layer %d", physical_layer)
        self.layer_save_finished_events[physical_layer].set()
        transfer_tasks.clear()
        self.request_queue.task_done()


class KVCacheStoreLayerRecvingThread(_GVALayerTransferThreadBase):
    def __init__(
        self,
        m_store: GVALayerwiseCapable,
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
            max_transfer_blocks=max_transfer_blocks,
            max_transfer_bytes=max_transfer_bytes,
        )
        self.get_event = get_event
        self.layer_load_finished_events = layer_load_finished_events
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events
        self.final_layer_id = num_layers - 1
        self.h2d_stagger_us = h2d_stagger_us
        self.external_slot_release_waiter = external_slot_release_waiter
        self.save_failure_checker = save_failure_checker
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
        """Pre-compute shared block data for all layers (GVA path)."""
        if self.group_builders is not None:
            builder = self.group_builders[task.group_id]
        else:
            builder = self.layer_batch_builder
        return builder.build_shared(task, is_save=False)

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
        wait_for_save = data.wait_for_save_layer
        transfer_tasks = data.transfer_tasks
        layer_id = data.layer_id
        attention_start_gate = data.attention_start_gate

        if wait_for_save is not None:
            while not self.layer_save_finished_events[wait_for_save].wait(timeout=10):
                if self.save_failure_checker is not None:
                    self.save_failure_checker()
                logger.info("Layerwise %d save wait timed out, keep waiting before load", wait_for_save)
            if self.save_failure_checker is not None:
                self.save_failure_checker()
            # Non-saving TP ranks have no D2H task to synchronize the event.
            # Their CPU save-finished signal only means the event was recorded;
            # wait for the NPU work before reusing the local HBM buffer.
            self.sync_save_events[wait_for_save].synchronize()
            logger.debug("Layer save event cleared: layer %d", wait_for_save)
            self.layer_save_finished_events[wait_for_save].clear()

        if len(transfer_tasks) == 0:
            if self.external_slot_release_waiter is not None:
                self.external_slot_release_waiter(layer_id)
            assert not self.layer_load_finished_events[layer_id].is_set()
            logger.debug("Layer load event set: layer %d", layer_id)
            self.layer_load_finished_events[layer_id].set()
            self.request_queue.task_done()
            return

        # Build req_meta for all tasks first; if all are None, early return.
        task_metas: list[tuple[LayerTransferTask, LayerBatchReqMeta]] = []
        for task in transfer_tasks:
            shared = task.shared_block_data
            builder = self.group_builders[task.group_id] if self.group_builders else self.layer_batch_builder
            if shared is not None:
                req_meta: LayerBatchReqMeta | None = builder.build_addrs(shared, task.layer_idx_in_group)
            else:
                req_meta = builder.build(task, is_save=False)
            if req_meta is not None:
                task_metas.append((task, req_meta))

        if not task_metas:
            if self.external_slot_release_waiter is not None:
                self.external_slot_release_waiter(layer_id)
            assert not self.layer_load_finished_events[layer_id].is_set()
            logger.debug("Layer load event set: layer %d", layer_id)
            self.layer_load_finished_events[layer_id].set()
            self.request_queue.task_done()
            return

        if attention_start_gate is not None:
            while not attention_start_gate.wait(timeout=10):
                logger.info("Layerwise %d load waits for attention compute start", layer_id)

        all_load_keys: list[str] = []
        all_req_ids: set[str] = set()
        last_chunk_req_ids: set[str] = set()
        all_gvas = []
        all_addrs = []
        all_sizes = []
        for task, req_meta in task_metas:
            if req_meta.load_keys:
                all_load_keys.extend(req_meta.load_keys)
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
            raise RuntimeError(f"Layerwise {layer_id} load batch_copy failed with return code {res}")

        if layer_id == self.final_layer_id and all_load_keys:
            unique_load_keys = list(dict.fromkeys(all_load_keys))
            self._gva_store.batch_remove_lease(unique_load_keys)
            logger.debug(
                "[KVPOOL] load_thread released %d leases after final layer %d",
                len(unique_load_keys),
                layer_id,
            )
        if layer_id == self.final_layer_id:
            for req_id in all_req_ids:
                if req_id in last_chunk_req_ids:
                    self.set_finished_request(req_id)
        assert not self.layer_load_finished_events[layer_id].is_set(), f"thread: {layer_id} load failed "
        logger.debug("Layer load event set: layer %d", layer_id)
        self.layer_load_finished_events[layer_id].set()
        # transfer_tasks aliases KVPoolWorker.layer_load_tasks[layer_id]. Do
        # not mutate the worker-owned list from this asynchronous thread. The
        # worker replaces all per-layer lists at the beginning of every step.
        self.request_queue.task_done()
        self.get_event.set()


@dataclass
class GVALayerwiseThreadContext:
    """State snapshot used to construct the GVA layerwise threads.

    Captured once by ``KVPoolWorker`` when the transfer threads start; the
    factories below consume it so the worker does not need to know the
    per-thread parameter lists.
    """

    m_store: GVALayerwiseCapable
    token_database: ChunkedTokenDatabase
    block_size: int | list[int]
    tp_rank: int
    tp_size: int
    dcp_size: int
    page_size_bytes: int
    num_layers: int
    layer_save_finished_events: list[threading.Event]
    sync_save_events: list[torch.npu.Event]
    max_transfer_blocks: int = 0
    max_transfer_bytes: int = 0
    num_kv_cache_groups: int = 1
    group_num_layers: dict[int, int] | None = None
    group_block_len: dict[int, list[int]] | None = None


def build_group_layer_builders(ctx: GVALayerwiseThreadContext) -> list[LayerBatchBuilder]:
    """Build one independent LayerBatchBuilder per KV cache group.

    Each builder owns reusable numpy buffers, so send and receive threads
    must never share builder instances.
    """
    builders = []
    for group_id in range(ctx.num_kv_cache_groups):
        group_num_layers_map = ctx.group_num_layers or {}
        group_block_len_map = ctx.group_block_len or {}
        group_num_layers = group_num_layers_map.get(group_id, ctx.num_layers)
        group_block_len = group_block_len_map.get(group_id, group_block_len_map.get(0, []))
        group_page_size = sum(group_block_len) if group_block_len else ctx.page_size_bytes
        builders.append(
            LayerBatchBuilder(
                ctx.token_database,
                group_page_size,
                group_num_layers,
                group_id=group_id,
            )
        )
    return builders


def create_gva_sending_thread(
    ctx: GVALayerwiseThreadContext,
    ready_event: threading.Event,
) -> KVCacheStoreLayerSendingThread:
    return KVCacheStoreLayerSendingThread(
        ctx.m_store,
        ctx.token_database,
        ctx.block_size,
        ctx.tp_rank,
        ctx.tp_size,
        ctx.dcp_size,
        ctx.page_size_bytes,
        ready_event,
        ctx.num_layers,
        ctx.layer_save_finished_events,
        ctx.sync_save_events,
        ctx.max_transfer_blocks,
        ctx.max_transfer_bytes,
        group_builders=build_group_layer_builders(ctx),
    )


def create_gva_recving_thread(
    ctx: GVALayerwiseThreadContext,
    ready_event: threading.Event,
    get_event: threading.Event,
    layer_load_finished_events: list[threading.Event],
    h2d_stagger_us: int = 0,
    external_slot_release_waiter: Callable[[int], None] | None = None,
    save_failure_checker: Callable[[], None] | None = None,
) -> KVCacheStoreLayerRecvingThread:
    return KVCacheStoreLayerRecvingThread(
        ctx.m_store,
        ctx.token_database,
        ctx.block_size,
        ctx.tp_rank,
        ctx.tp_size,
        ctx.dcp_size,
        ctx.page_size_bytes,
        ready_event,
        get_event,
        layer_load_finished_events,
        ctx.layer_save_finished_events,
        ctx.sync_save_events,
        ctx.num_layers,
        h2d_stagger_us,
        ctx.max_transfer_blocks,
        ctx.max_transfer_bytes,
        group_builders=build_group_layer_builders(ctx),
        external_slot_release_waiter=external_slot_release_waiter,
        save_failure_checker=save_failure_checker,
    )
