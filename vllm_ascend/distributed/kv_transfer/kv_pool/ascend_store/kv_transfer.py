from __future__ import annotations

import ctypes
import queue
import threading
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend

# isort: off
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    LayerBatchReqMeta,
    LayerBlockRange,
    LayerLoadTask,
    LayerMultiBlockReqMeta,
    LayerTransferTask,
    ReqMeta,
    SharedBlockData,
    get_block_hashes,
)
# isort: on


def _circular_shift(lst: list, offset: int) -> list:
    if not lst or offset == 0:
        return lst
    return lst[offset:] + lst[:offset]


def _circular_shift_array(value: np.ndarray, offset: int) -> np.ndarray:
    length = len(value)
    if length == 0:
        return value
    offset %= length
    if offset == 0:
        return value
    return np.concatenate((value[offset:], value[:offset]))


class LayerBatchBuilder:
    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        my_key_index: int,
        num_ranks_per_layer: int,
        page_size_bytes: int,
        num_layers: int,
    ) -> None:
        self.my_key_index = my_key_index
        self.num_ranks_per_layer = num_ranks_per_layer
        self.page_size_bytes = page_size_bytes
        self.num_layers = num_layers
        self._block_len_np = np.asarray(token_database.group_block_len[0], dtype=np.int64)
        self._kv_caches_base_addr_np = np.asarray(
            token_database.group_kv_caches_base_addr[0],
            dtype=np.int64,
        )
        group_block_stride = token_database.group_block_stride.get(0, token_database.group_block_len[0])
        self._block_stride_np = np.asarray(group_block_stride, dtype=np.int64)
        # group_block_len[0] / kv_caches_base_addr[0] are laid out flat as
        # [layer0_caches..., layer1_caches..., ...]; the per-layer stride is the
        # total length divided by the number of layers (mirrors
        # ChunkedTokenDatabase caches_per_layer computation).
        self._caches_per_layer = max(1, self._block_len_np.shape[0] // max(1, num_layers))
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
        caches_per_layer = self._caches_per_layer
        # group_* arrays are laid out flat as [layer0_caches..., layer1_caches...];
        # slice the per-layer window for ``layer_id``. Using the full length as the
        # stride (the old behaviour) overshoots for layer_id >= 1 and yields empty
        # slices -> broadcast errors.
        base_offset = layer_id * caches_per_layer
        layer_base_addrs = self._kv_caches_base_addr_np[base_offset : base_offset + caches_per_layer]
        layer_block_len = self._block_len_np[base_offset : base_offset + caches_per_layer]
        layer_block_stride = self._block_stride_np[base_offset : base_offset + caches_per_layer]
        # Per-cache inner offsets within one layer's page: [0, len0, len0+len1, ...].
        layer_inner_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(layer_block_len[:-1], dtype=np.int64))
        )
        rank_layer_offset = (layer_id * self.num_ranks_per_layer + self.my_key_index) * self.page_size_bytes

        addr_arr = layer_base_addrs[None, :] + block_ids_arr[:, None] * layer_block_stride[None, :]
        size_arr = np.broadcast_to(layer_block_len, addr_arr.shape)
        gvas_arr = base_gvas_arr[:, None] + rank_layer_offset + layer_inner_offsets[None, :]

        return (
            addr_arr.ravel(),
            size_arr.ravel(),
            gvas_arr.ravel(),
        )

    @staticmethod
    def _require_request_arrays(
        block_range: LayerBlockRange,
    ) -> tuple[np.ndarray, np.ndarray]:
        request = block_range.request
        if request.block_ids_np is None or request.block_gvas_np is None:
            raise RuntimeError("ReqMeta numpy block metadata is not initialized")
        return request.block_ids_np, request.block_gvas_np

    def build_shared(self, task: LayerTransferTask) -> SharedBlockData | None:
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
        offset = 0

        for block_range in task.block_ranges:
            request = block_range.request
            req_ids.append(request.req_id)
            is_last_chunks.append(request.is_last_chunk)
            block_ids_np, block_gvas_np = self._require_request_arrays(block_range)

            num_blocks = block_range.end_block - block_range.start_block
            if num_blocks > 0:
                gva_start = block_range.start_block - request.gva_block_offset
                gva_end = block_range.end_block - request.gva_block_offset
                if gva_start < 0 or gva_end > len(block_gvas_np):
                    raise RuntimeError(
                        "ReqMeta GVA metadata does not cover requested block "
                        f"range [{block_range.start_block}, {block_range.end_block}) "
                        f"with offset {request.gva_block_offset}"
                    )
                end = offset + num_blocks
                block_ids_arr[offset:end] = block_ids_np[block_range.start_block : block_range.end_block]
                block_gvas_arr[offset:end] = block_gvas_np[gva_start:gva_end]
                offset = end

            if block_range.partial_block_index is not None:
                assert request.last_block_gva is not None
                block_ids_arr[offset] = block_ids_np[block_range.partial_block_index]
                block_gvas_arr[offset] = request.last_block_gva
                offset += 1

        block_ids_arr, block_gvas_arr = self._dedupe_transfer_blocks(block_ids_arr[:offset], block_gvas_arr[:offset])

        return SharedBlockData(
            block_ids_arr=block_ids_arr,
            block_gvas_arr=block_gvas_arr,
            req_ids=req_ids,
            is_last_chunks=is_last_chunks,
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
        )

    def build(self, task: LayerTransferTask) -> LayerBatchReqMeta | None:
        """Full build: shared data + per-layer addresses (backward compat)."""
        shared = self.build_shared(task)
        if shared is None:
            return None
        return self.build_addrs(shared, task.layer_id)


class KVTransferThread(threading.Thread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        dcp_size: int = 1,
        ready_event: threading.Event | None = None,
        name: str = "KVTransferThread",
    ):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event or threading.Event()
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.num_addrs_per_block = len(token_database.group_block_len[0])
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.stored_requests: defaultdict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []

    def _get_block_size(self, kv_cache_group_id: int = 0) -> int:
        if isinstance(self.block_size, list):
            if kv_cache_group_id >= len(self.block_size):
                return self.block_size[0]
            return self.block_size[kv_cache_group_id]
        return self.block_size

    def add_request(
        self,
        request: ReqMeta | LayerBatchReqMeta | LayerMultiBlockReqMeta,
    ) -> torch.Tensor:
        self.request_queue.put(request)

    def get_and_clear_finished_requests(
        self,
        req_ids: set[str] | None = None,
    ) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            if req_ids is None:
                finished_requests = self.finished_requests.copy()
                self.finished_requests.clear()
            else:
                finished_requests = self.finished_requests & req_ids
                self.finished_requests -= finished_requests
        return finished_requests

    def discard_finished_requests(self, req_ids: set[str]) -> None:
        with self.done_task_lock:
            self.finished_requests -= req_ids

    def set_finished_request(self, req_id):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def try_finish_and_delete_stored_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            if req_id in self.stored_requests and self.stored_requests[req_id] == 0:
                del self.stored_requests[req_id]
                return True
            return False

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

        max_transfer_addrs = 0
        if max_transfer_blocks > 0:
            max_transfer_addrs = max_transfer_blocks * self.num_addrs_per_block
        if max_transfer_addrs <= 0:
            max_transfer_addrs = len(gvas)

        assert self.m_store.store is not None
        for start in range(0, len(gvas), max_transfer_addrs):
            end = start + max_transfer_addrs
            split_gvas, split_addrs, split_sizes = self._split_transfer_packets(
                gvas[start:end],
                addrs[start:end],
                sizes[start:end],
                max_transfer_bytes,
            )
            res = self.m_store.store.batch_copy(
                split_gvas.tolist(),
                split_addrs.tolist(),
                split_sizes.tolist(),
                direction,
            )
            if res != 0:
                return res
        return 0

    def _set_os_thread_name(self) -> None:
        try:
            libc = ctypes.CDLL("libc.so.6")
            # Linux task comm is limited to 15 visible bytes plus NUL.
            libc.prctl(15, self.name[:15].encode(), 0, 0, 0)
        except Exception:
            pass

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self._set_os_thread_name()
        self.m_store.set_device()
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request. This indicates queue shutdown or invalid request.")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                logger.error(
                    "Error in KVCacheTransferThread. type=%s, error=%s. Check thread state and request processing.",
                    type(e).__name__,
                    e,
                )

    def _handle_request(self, req_meta: Any):
        pass

    def lookup(
        self,
        keys: list[str],
    ) -> list[bool]:
        """
        Check the existence of all keys from the cache engine.
        :return: A bool list where True means the key exists in store.
        """
        try:
            res = self.m_store.exists(keys)  # type: ignore[assignment]
            exists_list = [False] * len(keys)
            for index, value in enumerate(res):  # type: ignore[arg-type]
                exists_list[index] = value == 1
            return exists_list
        except Exception as e:
            logger.error(
                "Remote connection failed in lookup. type=%s, error=%s. Check network and remote store.",
                type(e).__name__,
                e,
            )
            return [False] * len(keys)

    def update_kv_event(self, event: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(event)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events

    @staticmethod
    def _skip_null_blocks(req_meta: ReqMeta, group_id: int, cache_role: str = "kv") -> bool:
        if cache_role != "kv":
            return False
        skip_flags = req_meta.skip_null_blocks_by_group
        return group_id < len(skip_flags) and skip_flags[group_id] if skip_flags else False

    def _process_tokens_with_block_ids(
        self,
        token_len: int,
        block_hashes,
        block_ids: list[int],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        cache_role: str = "kv",
    ):
        process_with_block_ids = getattr(self.token_database, "process_tokens_with_block_ids", None)
        if process_with_block_ids is not None:
            return process_with_block_ids(
                token_len,
                block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=kv_cache_group_id,
                skip_null_blocks=skip_null_blocks,
                cache_role=cache_role,
            )

        def iter_with_legacy_process_tokens():
            try:
                token_iter = self.token_database.process_tokens(token_len, block_hashes, mask_num)
            except TypeError:
                token_iter = self.token_database.process_tokens(token_len, block_hashes)
            group_block_size = self._get_block_size(kv_cache_group_id)
            for start, end, key in token_iter:
                block_idx = start // group_block_size
                if block_idx >= len(block_ids):
                    continue
                block_id = block_ids[block_idx]
                if skip_null_blocks and cache_role == "kv" and block_id <= 0:
                    continue
                yield start, end, key, block_id

        return iter_with_legacy_process_tokens()

    def _prepare_value(
        self,
        start: int,
        end: int,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
    ):
        try:
            return self.token_database.prepare_value(
                start,
                end,
                block_ids,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
            )
        except TypeError:
            return self.token_database.prepare_value(start, end, block_ids)

    def _decode_adaptor_prefill_pp(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
    ):
        try:
            return self.token_database.decode_adaptor_prefill_pp(
                keys,
                addrs,
                sizes,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
            )
        except TypeError:
            return self.token_database.decode_adaptor_prefill_pp(keys, addrs, sizes)


class KVCacheStoreSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        dcp_size: int = 1,
        put_step: int = 1,
        kv_role: str = "kv_producer",
        ready_event: threading.Event | None = None,
        group_uses_align_state: list[bool] | None = None,
        enable_kv_event: bool = False,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, tp_size, dcp_size, ready_event, name="KVCacheSendingThread"
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests = defaultdict[str, int](int)
        self.group_uses_align_state = group_uses_align_state or []
        self.enable_kv_event = enable_kv_event
        self.completed_events_lock = threading.Lock()
        self.completed_events: dict[int, int] = {}

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def mark_completed_events(self, event_id: int | None) -> None:
        if event_id is not None:
            with self.completed_events_lock:
                self.completed_events[event_id] = 1

    def get_completed_events(self):
        if not self.completed_events:
            return None
        with self.completed_events_lock:
            completed_events = self.completed_events.copy()
            self.completed_events.clear()
        return completed_events

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.token_len_chunk
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        try:
            if req_id not in self.stored_requests:
                self.request_queue.task_done()
                return

            for group_id in req_meta.kv_cache_group_ids or [0]:
                starts = []
                ends = []
                keys = []
                block_hashes = []
                block_ids = req_meta.block_ids_by_group[group_id]
                group_block_size = self._get_block_size(group_id)
                group_block_hashes = get_block_hashes(
                    req_meta.block_hashes,
                    group_block_size,
                    getattr(self.token_database, "hash_block_size", group_block_size),
                )

                for start, end, key, _ in self._process_tokens_with_block_ids(
                    token_len,
                    req_meta.block_hashes,
                    block_ids,
                    kv_cache_group_id=group_id,
                    skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                ):
                    starts.append(start)
                    ends.append(end)
                    keys.append(key.to_string())
                    block_hashes.append(group_block_hashes[start // group_block_size])

                if (
                    not self.dcp_size > 1
                    and not req_meta.disable_tp_key_sharding
                    and not self.group_uses_align_state[group_id]
                ):
                    starts = starts[self.tp_rank % self.put_step :: self.put_step]
                    ends = ends[self.tp_rank % self.put_step :: self.put_step]
                    keys = keys[self.tp_rank % self.put_step :: self.put_step]
                    block_hashes = block_hashes[self.tp_rank % self.put_step :: self.put_step]

                if not keys:
                    continue

                exists_states = self.lookup(keys)
                missing_indices = [index for index, exists in enumerate(exists_states) if not exists]

                if not missing_indices:
                    continue

                starts = [starts[index] for index in missing_indices]
                ends = [ends[index] for index in missing_indices]
                keys = [keys[index] for index in missing_indices]
                block_hashes = [block_hashes[index] for index in missing_indices]

                logger.info(
                    "Storing KV cache for %d out of %d blocks (missing_count=%d) for request %s in group %d",
                    len(keys),
                    token_len // group_block_size,
                    len(missing_indices),
                    req_id,
                    group_id,
                )
                logger.debug(
                    "KV pool put request=%s group=%d token_len=%d keys=%d sample_keys=%s",
                    req_id,
                    group_id,
                    token_len,
                    len(keys),
                    keys[:3],
                )

                addrs = []
                sizes = []
                stored_events: list[BlockStored] = []
                prev_key = None
                new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]
                for index, start in enumerate(starts):
                    addr, size, _ = self._prepare_value(
                        start,
                        ends[index],
                        block_ids,
                        kv_cache_group_id=group_id,
                    )
                    addrs.append(addr)
                    sizes.append(size)

                    # Create KV event
                    if self.enable_kv_event:
                        token_ids = req_meta.token_ids[start : ends[index]] if req_meta.token_ids is not None else None
                        block_size = (
                            req_meta.original_block_size[group_id]
                            if isinstance(req_meta.original_block_size, list)
                            else req_meta.original_block_size
                        )
                        if block_size is not None:
                            stored_event = BlockStored(
                                block_hashes=[new_block_hashes[index]],
                                parent_block_hash=prev_key,
                                token_ids=token_ids,
                                block_size=block_size,
                                lora_id=None,
                                medium="cpu",
                                lora_name=None,
                            )
                            stored_events.append(stored_event)
                            prev_key = new_block_hashes[index]
                            logger.debug("Added kv cache event '%s' to kv cache events queue", stored_event)

                if self.kv_role == "kv_consumer":
                    keys, addrs, sizes = self._decode_adaptor_prefill_pp(
                        keys,
                        addrs,
                        sizes,
                        kv_cache_group_id=group_id,
                    )

                if current_event is not None:
                    current_event.synchronize()
                self.m_store.put(keys, addrs, sizes)

                # TODO Query specific replica info to update the event
                if self.enable_kv_event and stored_events is not None:
                    self.update_kv_event(stored_events)
        finally:
            # always free blocks
            self.mark_completed_events(req_meta.event_id)
        self.dec_stored_request(req_id)
        if self.stored_requests.get(req_id, -1) == 0:
            self.delete_finished_stored_request(req_id)
            self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        dcp_size: int = 1,
        ready_event: threading.Event | None = None,
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
            name="KVCacheStoreRecvingThread",
        )
        self._invalid_block_ids = invalid_block_ids if invalid_block_ids is not None else set()
        self._invalid_block_ids_lock = invalid_block_ids_lock or threading.Lock()

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        addr_list = []
        size_list = []
        key_list = []
        block_id_list: list[int] = []
        group_ids = req_meta.kv_cache_group_ids or [0]
        for group_id in group_ids:
            block_ids = req_meta.block_ids_by_group[group_id]
            group_block_size = self._get_block_size(group_id)
            mask_num = (
                req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
                // group_block_size
                * group_block_size
            )
            for start, end, key, _ in self._process_tokens_with_block_ids(
                token_len,
                req_meta.block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=group_id,
                skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
            ):
                addr, size, block_id = self._prepare_value(
                    start,
                    end,
                    block_ids,
                    kv_cache_group_id=group_id,
                )
                key_list.append(key.to_string())
                addr_list.append(addr)
                size_list.append(size)
                block_id_list.append(block_id)
        if not key_list:
            self.set_finished_request(req_id)
            self.request_queue.task_done()
            return
        key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
        addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
        size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
        block_id_list_c = (
            block_id_list[self.tp_rank % len(block_id_list) :] + block_id_list[: self.tp_rank % len(block_id_list)]
        )
        logger.debug(
            "KV pool async recv calls backend get request=%s token_len=%d groups=%s keys=%d sample_keys=%s",
            req_id,
            token_len,
            req_meta.kv_cache_group_ids or [0],
            len(key_list_c),
            key_list_c[:3],
        )
        ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
        if ret is not None and any(r != 0 for r in ret):
            missing_block_ids = record_failed_blocks(
                block_id_list_c,
                ret,
            )
            if len(req_meta.block_ids_by_group) == 1:
                with self._invalid_block_ids_lock:
                    self._invalid_block_ids.update(missing_block_ids)
            elif missing_block_ids:
                logger.error(
                    "KV load failed for hybrid request %s. "
                    "Skip invalid-block fallback to avoid scheduler crash. "
                    "failed_blocks=%s",
                    req_id,
                    missing_block_ids,
                )
        elif ret is None:
            missing_block_ids = record_failed_blocks(
                block_id_list_c,
                [1] * len(block_id_list_c),
            )
            if len(req_meta.block_ids_by_group) == 1:
                with self._invalid_block_ids_lock:
                    self._invalid_block_ids.update(missing_block_ids)
            elif missing_block_ids:
                logger.error(
                    "KV load failed for hybrid request %s. "
                    "Skip invalid-block fallback to avoid scheduler crash. "
                    "failed_blocks=%s",
                    req_id,
                    missing_block_ids,
                )
        logger.debug(
            "KV pool async recv backend get returned request=%s token_len=%d groups=%s keys=%d",
            req_id,
            token_len,
            req_meta.kv_cache_group_ids or [0],
            len(key_list_c),
        )
        self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreKeyLayerSendingThread(KVTransferThread):
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
                key_all = key.split_layers(self.final_layer_id + 1)
                entries.append((start, end, key_all))
            cache[br_idx] = entries

        return cache

    def add_request(  # type: ignore[override]
        self, req_meta: list[LayerTransferTask]
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

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
                    keys.append(key.split_layers(self.final_layer_id + 1)[layer_id])

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
            exists_states = self.lookup(key_list)
            missing_indices = [index for index, exists in enumerate(exists_states) if not exists]
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

    def add_request(  # type: ignore[override]
        self, req_meta: LayerLoadTask
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _wait_for_save(self, layer_id: int) -> None:
        while not self.layer_save_finished_events[layer_id].wait(timeout=10):
            logger.info("Layerwise %d save wait timed out, keep waiting before load", layer_id)
        logger.debug("Key-based layer save event cleared: layer %d", layer_id)
        self.layer_save_finished_events[layer_id].clear()

    def _handle_request(  # type: ignore[override]
        self, data: LayerLoadTask
    ):
        wait_for_save = data.wait_for_save_layer
        layer_id = data.layer_id
        if wait_for_save is not None:
            self._wait_for_save(wait_for_save)

        if data.attention_start_gate is not None:
            while not data.attention_start_gate.wait(timeout=10):
                logger.info("Layerwise %d load waits for attention compute start", layer_id)

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
                    ).split_layers(self.final_layer_id + 1)[layer_id]
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


class KVCacheStoreLayerSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        put_step: int,
        my_key_index: int,
        num_ranks_per_layer: int,
        page_size_bytes: int,
        ready_event: threading.Event,
        num_layers: int,
        layer_save_finished_events: list[threading.Event],
        sync_save_events: list[torch.npu.Event],
        max_transfer_blocks: int = 0,
        max_transfer_bytes: int = 0,
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
        self.put_step = put_step
        self.stored_requests: defaultdict[str, int] = defaultdict(int)
        self.done_task_lock = threading.Lock()
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes
        self.layer_batch_builder = LayerBatchBuilder(
            token_database,
            my_key_index,
            num_ranks_per_layer,
            page_size_bytes,
            num_layers,
        )

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def build_shared_data(self, task: LayerTransferTask) -> SharedBlockData | None:
        """Pre-compute shared block data for all layers (GVA path)."""
        return self.layer_batch_builder.build_shared(task)

    def add_request(  # type: ignore[override]
        self, req_meta: list[LayerTransferTask]
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, transfer_tasks: list[LayerTransferTask]
    ):
        if len(transfer_tasks) == 0:
            self.request_queue.task_done()
            return
        if len(transfer_tasks) > 1:
            raise ValueError(f"Expected at most one layer transfer task, got {len(transfer_tasks)}")
        task = transfer_tasks[0]
        shared = task.shared_block_data
        if shared is None:
            layer_id = task.layer_id
            assert not self.layer_save_finished_events[layer_id].is_set(), f"thread: {layer_id} save failed "
            logger.debug("Layer save event set: layer %d", layer_id)
            self.layer_save_finished_events[layer_id].set()
            self.request_queue.task_done()
            return
        req_meta = self.layer_batch_builder.build_addrs(shared, task.layer_id)
        layer_id = req_meta.layer_id
        rank_start = self.tp_rank % self.put_step
        addr_array = req_meta.addr_array[rank_start :: self.put_step]
        size_array = req_meta.size_array[rank_start :: self.put_step]
        gvas_array = req_meta.gvas_array[rank_start :: self.put_step]
        for req_id in req_meta.req_ids:
            self.dec_stored_request(req_id)
        self.sync_save_events[layer_id].synchronize()
        res = self._batch_copy_with_limits(
            gvas_array,
            addr_array,
            size_array,
            0,
            self.max_transfer_blocks,
            self.max_transfer_bytes,
        )
        if res != 0:
            logger.error("Layerwise %d save batch_copy failed with return code %d", layer_id, res)
        for req_id in req_meta.req_ids:
            if self.try_finish_and_delete_stored_request(req_id):
                self.set_finished_request(req_id)
        assert not self.layer_save_finished_events[layer_id].is_set(), f"thread: {layer_id} save failed "
        logger.debug("Layer save event set: layer %d", layer_id)
        self.layer_save_finished_events[layer_id].set()
        transfer_tasks.clear()
        self.request_queue.task_done()


class KVCacheStoreLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        my_key_index: int,
        num_ranks_per_layer: int,
        page_size_bytes: int,
        ready_event: threading.Event,
        get_event: threading.Event,
        layer_load_finished_events: list[threading.Event],
        layer_save_finished_events: list[threading.Event],
        num_layers: int,
        h2d_stagger_us: int = 0,
        max_transfer_blocks: int = 0,
        max_transfer_bytes: int = 0,
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
        self.final_layer_id = num_layers - 1
        self.h2d_stagger_us = h2d_stagger_us
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes
        self.layer_batch_builder = LayerBatchBuilder(
            token_database,
            my_key_index,
            num_ranks_per_layer,
            page_size_bytes,
            num_layers,
        )

    def build_shared_data(self, task: LayerTransferTask) -> SharedBlockData | None:
        """Pre-compute shared block data for all layers (GVA path)."""
        return self.layer_batch_builder.build_shared(task)

    def add_request(  # type: ignore[override]
        self, req_meta: LayerLoadTask
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

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

        if len(transfer_tasks) == 0:
            if wait_for_save is not None:
                while not self.layer_save_finished_events[wait_for_save].wait(timeout=10):
                    logger.info("Layerwise %d save wait timed out, keep waiting before load", wait_for_save)
                logger.debug("Layer save event cleared: layer %d", wait_for_save)
                self.layer_save_finished_events[wait_for_save].clear()
            assert not self.layer_load_finished_events[layer_id].is_set()
            logger.debug("Layer load event set: layer %d", layer_id)
            self.layer_load_finished_events[layer_id].set()
            self.request_queue.task_done()
            return

        if len(transfer_tasks) > 1:
            raise ValueError(f"Expected at most one layer transfer task, got {len(transfer_tasks)}")
        task = transfer_tasks[0]
        shared = task.shared_block_data
        if shared is not None:
            req_meta: LayerBatchReqMeta | None = self.layer_batch_builder.build_addrs(shared, task.layer_id)
        else:
            req_meta = self.layer_batch_builder.build(task)
        if req_meta is None:
            assert not self.layer_load_finished_events[layer_id].is_set()
            logger.debug("Layer load event set: layer %d", layer_id)
            self.layer_load_finished_events[layer_id].set()
            self.request_queue.task_done()
            return
        layer_id = req_meta.layer_id

        if wait_for_save is not None:
            while not self.layer_save_finished_events[wait_for_save].wait(timeout=10):
                logger.info("Layerwise %d save wait timed out, keep waiting before load", wait_for_save)
            logger.debug("Layer save event cleared: layer %d", wait_for_save)
            self.layer_save_finished_events[wait_for_save].clear()

        if attention_start_gate is not None:
            while not attention_start_gate.wait(timeout=10):
                logger.info("Layerwise %d load waits for attention compute start", layer_id)

        gvas_array = _circular_shift_array(
            req_meta.gvas_array,
            (self.tp_rank * len(req_meta.gvas_array)) // self.tp_size,
        )
        addr_array = _circular_shift_array(
            req_meta.addr_array,
            (self.tp_rank * len(req_meta.addr_array)) // self.tp_size,
        )
        size_array = _circular_shift_array(
            req_meta.size_array,
            (self.tp_rank * len(req_meta.size_array)) // self.tp_size,
        )
        self._stagger_h2d_submit(layer_id)
        res = self._batch_copy_with_limits(
            gvas_array,
            addr_array,
            size_array,
            1,
            self.max_transfer_blocks,
            self.max_transfer_bytes,
        )
        if res != 0:
            logger.error("Layerwise %d load batch_copy failed with return code %d", layer_id, res)
        if layer_id == self.final_layer_id:
            for req_id, is_last_chunk in zip(req_meta.req_ids, req_meta.is_last_chunks):
                if is_last_chunk:
                    self.set_finished_request(req_id)
        assert not self.layer_load_finished_events[layer_id].is_set(), f"thread: {layer_id} load failed "
        logger.debug("Layer load event set: layer %d", layer_id)
        self.layer_load_finished_events[layer_id].set()
        transfer_tasks.clear()
        self.request_queue.task_done()
        self.get_event.set()


def record_failed_blocks(
    block_ids: list[int],
    ret_codes: list[int],
) -> set[int]:
    failed_blocks: set[int] = set()
    for block_id, code in zip(block_ids, ret_codes):
        if code != 0:
            failed_blocks.add(block_id)
    if failed_blocks:
        logger.error(
            "Failed to load blocks. failed_count=%d, failed_blocks=%s. Check block availability and memory state.",
            len(failed_blocks),
            failed_blocks,
        )
    return failed_blocks
