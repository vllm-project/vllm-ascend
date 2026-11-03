from __future__ import annotations

import ctypes
import os
import queue
import signal
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import AttentionComputeStartGate
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import (
    Backend,
    StoreCapacityError,
    TransferCompletionUnknown,
    require_aligned_batch_results,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    ReqMeta,
    get_block_hashes,
)

if TYPE_CHECKING:
    from .state_store.state_store_layout import AcceptedStateStoreLayout


_TRANSFER_SHUTDOWN_POLL_S = 0.1


class KVTransferThread(threading.Thread):
    _invalid_block_ids: set[int]
    _invalid_block_ids_lock: threading.Lock

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
        # Serialize request submission with the transition to a fatal state.
        # Otherwise a producer can enqueue immediately after the failed
        # consumer has drained the queue, leaving work with no live consumer.
        self._request_submit_lock = threading.Lock()
        self._stopping = threading.Event()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.finished_requests: set[str] = set()
        self.failed_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []
        self._fatal_error: BaseException | None = None
        # Keep the request and its fences alive until DMA completion is proven.
        self._inflight_request: ReqMeta | None = None
        self.save_disabled = threading.Event()
        self.load_disabled = threading.Event()

    def _mark_failed_requests(self, requests: list[ReqMeta]) -> None:
        logger.error("Memcache transfer failed for requests=%s", [request.req_id for request in requests])
        for request in requests:
            request.can_save = False
        with self.done_task_lock:
            self.failed_requests.update(request.req_id for request in requests if len(request.block_ids_by_group) > 1)
        self.save_disabled.set()
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(
                block_id
                for request in requests
                if len(request.block_ids_by_group) == 1
                for group in request.block_ids_by_group
                for block_id in group
                if block_id > 0
            )

    def get_and_clear_failed_requests(self) -> set[str]:
        with self.done_task_lock:
            failed = self.failed_requests.copy()
            self.failed_requests.clear()
        return failed

    def _get_block_size(self, kv_cache_group_id: int = 0) -> int:
        if isinstance(self.block_size, list):
            if kv_cache_group_id >= len(self.block_size):
                return self.block_size[0]
            return self.block_size[kv_cache_group_id]
        return self.block_size

    def add_request(self, request: Any) -> None:
        with self._request_submit_lock:
            if self._stopping.is_set():
                self._discard_request(request)
                return
            if self._fatal_error is not None:
                logger.error("Reject KV transfer submission to failed thread %s: %s", self.name, self._fatal_error)
                self._discard_request(request)
                return
            self.request_queue.put(request)

    def stop(self) -> None:
        with self._request_submit_lock:
            if not self._stopping.is_set():
                self._stopping.set()
                if self.is_alive():
                    self.request_queue.put(None)
                else:
                    self._discard_queued_requests()

    def _discard_request(self, request: Any) -> None:
        """Retire an item that has never submitted backend work."""

    def _discard_queued_requests(self) -> None:
        while True:
            try:
                request = self.request_queue.get_nowait()
            except queue.Empty:
                return
            self._discard_request(request)
            self.request_queue.task_done()

    def _wait_for_event(
        self,
        event: threading.Event | AttentionComputeStartGate,
        failure_checker: Callable[[], None] | None = None,
    ) -> bool:
        while not self._stopping.is_set():
            if event.wait(timeout=_TRANSFER_SHUTDOWN_POLL_S):
                return not self._stopping.is_set()
            if failure_checker is not None:
                failure_checker()
        return False

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

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(f"{self.name} failed during asynchronous transfer") from self._fatal_error

    def set_finished_request(self, req_id):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

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

        assert self.m_store.store is not None
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
            res = self.m_store.store.batch_copy(
                split_gvas.tolist(),
                split_addrs.tolist(),
                split_sizes.tolist(),
                direction,
            )
            if res != 0:
                logger.error("[KVPOOL] batch_copy %s FAILED res=%d", dir_name, res)
                return res
        return 0

    @staticmethod
    def _range_transfer_batches(
        keys: list[str],
        all_buffers: list[list[int]],
        all_sizes: list[list[int]],
        all_offsets: list[list[int]],
        max_transfer_blocks: int,
        max_transfer_bytes: int,
    ) -> list[tuple[list[str], list[list[int]], list[list[int]], list[list[int]]]]:
        """Split backend range calls by key count and per-segment byte size."""
        row_count = len(keys)
        if not (len(all_buffers) == len(all_sizes) == len(all_offsets) == row_count):
            raise ValueError("Layerwise range metadata must contain one buffer/size/offset row per key")

        # Default configuration (both limits unset) means one batch holding every
        # key and no per-segment splitting, so every row passes through as-is
        # unless it carries an empty or negative segment. Detect that case and
        # hand back the caller's own lists: this runs once per key, per layer,
        # per group, and rebuilding them element-wise was the receive thread's
        # largest single block of exposed time.
        if max_transfer_blocks <= 0 and max_transfer_bytes <= 0:
            for buffers, sizes, offsets in zip(all_buffers, all_sizes, all_offsets, strict=True):
                if not (len(buffers) == len(sizes) == len(offsets)):
                    raise ValueError("Layerwise range rows must align buffers, sizes, and offsets")
                if sizes and min(sizes) >= 0 and 0 not in sizes:
                    continue
                break
            else:
                return [(keys, all_buffers, all_sizes, all_offsets)]

        normalized_buffers: list[list[int]] = []
        normalized_sizes: list[list[int]] = []
        normalized_offsets: list[list[int]] = []
        for buffers, sizes, offsets in zip(all_buffers, all_sizes, all_offsets, strict=True):
            if not (len(buffers) == len(sizes) == len(offsets)):
                raise ValueError("Layerwise range rows must align buffers, sizes, and offsets")
            split_buffers: list[int] = []
            split_sizes: list[int] = []
            split_offsets: list[int] = []
            for buffer, size, offset in zip(buffers, sizes, offsets, strict=True):
                if size < 0:
                    raise ValueError(f"Layerwise range size must be non-negative, got {size}")
                if size == 0:
                    continue
                packet_size = max_transfer_bytes if max_transfer_bytes > 0 else size
                for packet_offset in range(0, size, packet_size):
                    split_buffers.append(buffer + packet_offset)
                    split_sizes.append(min(packet_size, size - packet_offset))
                    split_offsets.append(offset + packet_offset)
            normalized_buffers.append(split_buffers)
            normalized_sizes.append(split_sizes)
            normalized_offsets.append(split_offsets)

        rows_per_batch = max_transfer_blocks if max_transfer_blocks > 0 else max(1, row_count)
        return [
            (
                keys[start:end],
                normalized_buffers[start:end],
                normalized_sizes[start:end],
                normalized_offsets[start:end],
            )
            for start in range(0, row_count, rows_per_batch)
            for end in [min(start + rows_per_batch, row_count)]
        ]

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
        try:
            self.m_store.set_device()
        except Exception as exc:
            with self._request_submit_lock:
                self._fatal_error = exc
                self._handle_request_exception(None)
            return
        finally:
            self.ready_event.set()
        while True:
            request_data = None
            try:
                request_data = self.request_queue.get()
                if request_data is None or self._stopping.is_set():
                    self._discard_request(request_data)
                    self.request_queue.task_done()
                    if request_data is None:
                        return
                    continue
                self._handle_request(request_data)
            except Exception as e:
                if isinstance(e, TransferCompletionUnknown):
                    # DMA termination is unproven. Preserve buffers and SDK
                    # handles through process teardown without Python finalizers.
                    logger.critical(
                        "KV transfer completion unknown in %s; terminating worker with buffers retained",
                        self.name,
                        exc_info=True,
                    )
                    os.kill(os.getpid(), signal.SIGKILL)
                with self._request_submit_lock:
                    self._fatal_error = e
                    try:
                        self._handle_request_exception(request_data)
                    except Exception:
                        logger.exception("Failed to clean up KV transfer request after an unexpected dispatch error")
                logger.error(
                    "Error in KVCacheTransferThread(%s). type=%s, error=%s. Check thread state and request processing.",
                    self.name,
                    type(e).__name__,
                    e,
                )
                return

    def _handle_request(self, req_meta: Any):
        pass

    def _handle_request_exception(self, request_data: Any):
        """Allow subclasses to complete queue/request bookkeeping on errors."""
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

    def _get_missing_indices(self, keys: list[str], require_exists_check: bool = False) -> list[int]:
        """Filter existing keys unless the backend can do so during put.

        Callers that need the exact newly stored key set, such as KV event
        publishers, can force connector-side filtering.
        """
        if not require_exists_check and not self.m_store.requires_exists_before_put:
            return list(range(len(keys)))
        exists_states = self.lookup(keys)
        return [index for index, exists in enumerate(exists_states) if not exists]

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

    def _prepare_value(
        self,
        start: int,
        end: int,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        block_id: int | None = None,
    ):
        try:
            return self.token_database.prepare_value(
                start,
                end,
                block_ids,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
                block_id=block_id,
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


class KVCacheStoreBatch:
    """FIFO fence for a batch of asynchronous KV store requests."""

    def __init__(self) -> None:
        self.done = threading.Event()


# Poll interval for waiting on a specific request's queued save chunks
# (released-block fence; keep it small so scheduling stays responsive).
SAVE_REQUEST_POLL_INTERVAL_S = 0.001


class KVCacheStoreSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        pcp_rank: int = 0,
        pcp_size: int = 1,
        dcp_size: int = 1,
        put_step: int = 1,
        kv_role: str = "kv_producer",
        ready_event: threading.Event | None = None,
        group_uses_align_state: list[bool] | None = None,
        enable_kv_event: bool = False,
        worker: Any = None,
        *,
        state_store_layouts: dict[int, AcceptedStateStoreLayout] | None = None,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, tp_size, dcp_size, ready_event, name="KVCacheSendingThread"
        )
        self.pcp_rank = pcp_rank
        self.pcp_size = pcp_size
        self.put_step = put_step
        self.kv_role = kv_role
        # Request ids can be reused after preemption, so track each queued
        # store generation independently.
        self.stored_requests: dict[str, set[int]] = {}
        self._next_local_store_job_id = -1
        self.group_uses_align_state = group_uses_align_state or []
        self.enable_kv_event = enable_kv_event
        self.completed_events_lock = threading.Lock()
        self.completed_events: dict[int, int] = {}
        self.worker = worker
        self.state_store_layouts = dict(state_store_layouts or {})
        self._use_store_jobs = state_store_layouts is not None
        # Request-local durable high-water marks. They are advanced only after
        # the backend store succeeds, so a failed range is retried on the next
        # Decode boundary instead of being lost when scheduler metadata moves
        # forward. Keeping this on the worker also avoids rescanning the full
        # prefix on every successful Decode save.
        self._saved_offsets: dict[str, int] = {}
        # Token suffix retained only after a failed store. This lets a retry
        # publish correct KV events without copying the full request history on
        # the normal Decode path.
        self._retry_token_ids: dict[str, tuple[int, list[int]]] = {}

    def lookup_store_keys(self, keys: list[str], kv_cache_group_id: int = 0) -> list[bool] | None:
        """Return existence in the logical Decode-key domain.

        A Decode consumer may expand one logical block into one key per
        Prefill PP rank. The block is reusable only when every expanded key is
        present; checking only ``@pp_rank:0`` would turn a partial PUT into a
        false hit and permanently skip the missing partitions on retry.
        Return None after logging malformed results; callers must fail the job.
        """
        partitions = getattr(self.token_database, "partitions", None)
        if self.kv_role != "kv_consumer" or not keys or partitions is None or len(partitions) <= 1:
            return [bool(value) for value in self.lookup(keys)]

        empty_buffers: list[list[int]] = [[] for _ in keys]
        expanded_keys, _, _ = self._decode_adaptor_prefill_pp(
            keys,
            empty_buffers,
            empty_buffers,
            kv_cache_group_id=kv_cache_group_id,
        )
        if len(expanded_keys) % len(keys) != 0:
            logger.error(
                "Prefill-PP key adaptor expanded %s logical keys to an invalid count %s", len(keys), len(expanded_keys)
            )
            return None
        partitions_per_key = len(expanded_keys) // len(keys)
        if partitions_per_key == 0:
            logger.error("Prefill-PP key adaptor returned no store keys")
            return None
        expanded_exists = self.lookup(expanded_keys)
        if len(expanded_exists) != len(expanded_keys):
            logger.error(
                "KV store returned %s lookup results for %s Prefill-PP keys", len(expanded_exists), len(expanded_keys)
            )
            return None
        return [
            all(expanded_exists[start : start + partitions_per_key])
            for start in range(0, len(expanded_exists), partitions_per_key)
        ]

    def add_stored_request(self, req_meta: ReqMeta) -> None:
        with self.done_task_lock:
            if req_meta.store_job_id is None:
                req_meta.store_job_id = self._next_local_store_job_id
                self._next_local_store_job_id -= 1
            self.finished_requests.discard(req_meta.req_id)
            self.stored_requests.setdefault(req_meta.req_id, set()).add(req_meta.store_job_id)

    def add_save_batch(self, requests: list[ReqMeta]) -> KVCacheStoreBatch:
        """Queue requests followed by a fence that completes after the batch."""
        save_batch = KVCacheStoreBatch()
        # The lock makes registration, queue publication, and the transition
        # to a fatal state mutually exclusive. A failed consumer therefore
        # either drains this complete batch or rejects it before registration.
        with self._request_submit_lock:
            if self._stopping.is_set():
                for request in requests:
                    self._discard_request(request)
                save_batch.done.set()
                return save_batch
            if self._fatal_error is not None:
                logger.error("Reject KV save batch to failed thread %s: %s", self.name, self._fatal_error)
                # No source KV is read, but completion must release Scheduler pins.
                # Reject without registering jobs, advancing offsets, or emitting BlockStored.
                for request in requests:
                    self._discard_request(request)
                save_batch.done.set()
                return save_batch
            for request in requests:
                self.add_stored_request(request)
            for request in requests:
                self.request_queue.put(request)
            self.request_queue.put(save_batch)
        return save_batch

    def wait_for_requests_saved(self, req_ids: set[str] | None) -> None:
        """Block until every queued save chunk of these requests completed.

        Block-reuse fence: the scheduler frees a finished or preempted
        request's blocks immediately, so before this step's forward may
        reallocate them, all of that request's queued puts must drain. A
        request never queues new puts after it finishes or is preempted, so
        draining it once makes its blocks safe to reuse from then on.
        """
        if not req_ids:
            return
        pending = set(req_ids)
        wait_start = time.perf_counter()
        while True:
            self.raise_if_failed()
            with self.done_task_lock:
                pending &= self.stored_requests.keys()
                if not pending:
                    break
            time.sleep(SAVE_REQUEST_POLL_INTERVAL_S)
        elapsed = time.perf_counter() - wait_start
        logger.debug(
            "Released-block KV save fence (%s) completed after waiting %.3f ms tp_rank=%d",
            ",".join(sorted(req_ids))[:200],
            elapsed * 1000,
            self.tp_rank,
        )

    def is_live_store_job(self, req_meta: ReqMeta) -> bool:
        with self.done_task_lock:
            return req_meta.store_job_id in self.stored_requests.get(req_meta.req_id, ())

    def is_stored_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            return bool(self.stored_requests.get(req_id))

    def get_stored_request_count(self, req_id: str) -> int | None:
        with self.done_task_lock:
            jobs = self.stored_requests.get(req_id)
            return None if jobs is None else len(jobs)

    def get_stored_requests_snapshot(self) -> dict[str, int]:
        with self.done_task_lock:
            return {req_id: len(jobs) for req_id, jobs in self.stored_requests.items()}

    def finish_store_job(self, req_meta: ReqMeta) -> int | None:
        with self.done_task_lock:
            jobs = self.stored_requests.get(req_meta.req_id)
            if jobs is None or req_meta.store_job_id not in jobs:
                return None
            jobs.remove(req_meta.store_job_id)
            remaining = len(jobs)
            if remaining == 0:
                del self.stored_requests[req_meta.req_id]
            return remaining

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests.pop(req_id, None)

    def get_saved_offset(self, req_id: str) -> int:
        with self.done_task_lock:
            return self._saved_offsets.get(req_id, 0)

    def record_saved_offset(self, req_meta: ReqMeta, token_len: int) -> None:
        with self.done_task_lock:
            if req_meta.store_job_id in self.stored_requests.get(req_meta.req_id, ()):
                self._saved_offsets[req_meta.req_id] = max(self._saved_offsets.get(req_meta.req_id, 0), token_len)

    def reset_saved_request(self, req_id: str) -> None:
        with self.done_task_lock:
            self._saved_offsets.pop(req_id, None)
            self._retry_token_ids.pop(req_id, None)

    def get_event_token_ids(self, req_meta: ReqMeta, start: int, end: int) -> list[int] | None:
        if req_meta.token_ids is None:
            return None
        token_ids_start = req_meta.save_start_token
        token_ids = req_meta.token_ids
        with self.done_task_lock:
            if req_meta.store_job_id not in self.stored_requests.get(req_meta.req_id, ()):
                return []
            retry = self._retry_token_ids.get(req_meta.req_id)
            if retry is not None:
                retry_start, retry_ids = retry
                if retry_start + len(retry_ids) == token_ids_start:
                    token_ids_start = retry_start
                    token_ids = retry_ids + token_ids
        token_ids_end = token_ids_start + len(token_ids)
        if start < token_ids_start or end > token_ids_end:
            return []
        return token_ids[start - token_ids_start : end - token_ids_start]

    def _remember_retry_token_ids(self, req_meta: ReqMeta) -> None:
        if not self.enable_kv_event or req_meta.token_ids is None:
            return
        token_ids_start = req_meta.save_start_token
        token_ids = req_meta.token_ids.copy()
        with self.done_task_lock:
            if req_meta.store_job_id not in self.stored_requests.get(req_meta.req_id, ()):
                return
            retry = self._retry_token_ids.get(req_meta.req_id)
            if retry is not None:
                retry_start, retry_ids = retry
                if retry_start + len(retry_ids) == token_ids_start:
                    token_ids_start = retry_start
                    token_ids = retry_ids + token_ids
            self._retry_token_ids[req_meta.req_id] = (token_ids_start, token_ids)

    def _clear_retry_token_ids(self, req_meta: ReqMeta) -> None:
        with self.done_task_lock:
            if req_meta.store_job_id in self.stored_requests.get(req_meta.req_id, ()):
                self._retry_token_ids.pop(req_meta.req_id, None)

    def get_completed_events(self):
        with self.completed_events_lock:
            if not self.completed_events:
                return None
            completed_events = self.completed_events.copy()
            self.completed_events.clear()
        return completed_events

    def _finish_state_job(self, request: ReqMeta) -> None:
        remaining = self.finish_store_job(request)
        if remaining == 0:
            self.set_finished_request(request.req_id)
        assert request.store_job_id is not None
        with self.completed_events_lock:
            self.completed_events[request.store_job_id] = 1

    def _discard_request(self, request: Any) -> None:
        if isinstance(request, KVCacheStoreBatch):
            request.done.set()
        elif isinstance(request, ReqMeta):
            if self._use_store_jobs:
                self._finish_state_job(request)
            else:
                remaining = self.finish_store_job(request)
                if remaining == 0:
                    self.set_finished_request(request.req_id)
                if request.event_id is not None:
                    with self.completed_events_lock:
                        self.completed_events[request.event_id] = self.completed_events.get(request.event_id, 0) + 1

    def _handle_request_exception(self, request_data: Any):
        def finish_failed_request(failed_request: Any) -> None:
            if failed_request is self._inflight_request:
                return
            if self._use_store_jobs:
                self._discard_request(failed_request)
                return
            if isinstance(failed_request, ReqMeta):
                remaining = self.finish_store_job(failed_request)
                if remaining == 0:
                    self.set_finished_request(failed_request.req_id)
                if remaining is not None and failed_request.event_id is not None:
                    with self.completed_events_lock:
                        self.completed_events[failed_request.event_id] = (
                            self.completed_events.get(failed_request.event_id, 0) + 1
                        )
            elif isinstance(failed_request, KVCacheStoreBatch):
                # Complete the batch fence even after a fatal dispatch error.
                # Request fences observe _fatal_error before allowing block reuse.
                failed_request.done.set()

        # run() holds _request_submit_lock, and this thread is the only consumer.
        # unfinished_tasks counts queued items plus the current item if unacknowledged.
        # Equal counts mean the current item was already acknowledged; do not repeat it.
        queued_request_count = self.request_queue.qsize()
        current_needs_ack = self.request_queue.unfinished_tasks > queued_request_count
        if request_data is not None:
            finish_failed_request(request_data)
            if current_needs_ack:
                self.request_queue.task_done()
        # Acknowledge each dequeued item exactly once, including batch fences.
        for _ in range(queued_request_count):
            try:
                queued_request = self.request_queue.get_nowait()
            except queue.Empty:
                break
            finish_failed_request(queued_request)
            self.request_queue.task_done()

    def _handle_request(self, req_meta: ReqMeta | KVCacheStoreBatch):
        if isinstance(req_meta, KVCacheStoreBatch):
            req_meta.done.set()
            self.request_queue.task_done()
            return

        if self._use_store_jobs:
            self._handle_store_job(req_meta)
            return

        if self.worker is not None and getattr(self.worker, "tp_mismatch", False):
            req_id = req_meta.req_id
            tracked_request = self.is_live_store_job(req_meta)
            try:
                succeeded = self.worker._store_kv_tp_mismatch(req_meta)
                if tracked_request:
                    if succeeded is True:
                        self.record_saved_offset(req_meta, req_meta.token_len_chunk)
                        self._clear_retry_token_ids(req_meta)
                    else:
                        self._remember_retry_token_ids(req_meta)
            except Exception:
                self._remember_retry_token_ids(req_meta)
                logger.exception("Failed to store KV cache for TP-mismatch request %s", req_id)
            finally:
                remaining = self.finish_store_job(req_meta) if tracked_request else None
                if tracked_request and remaining == 0:
                    self.set_finished_request(req_id)
                if req_meta.event_id is not None:
                    with self.completed_events_lock:
                        self.completed_events[req_meta.event_id] = 1
                self.request_queue.task_done()
            return

        req_id = req_meta.req_id
        tracked_request = False
        try:
            tracked_request = self.is_live_store_job(req_meta)
            if not tracked_request:
                return
            # Only explicit success may advance the saved offset; failures return False.
            succeeded = self._handle_stored_request(req_meta)
            if succeeded is True:
                self.record_saved_offset(req_meta, req_meta.token_len_chunk)
                self._clear_retry_token_ids(req_meta)
            else:
                self._remember_retry_token_ids(req_meta)
        except Exception:
            self._remember_retry_token_ids(req_meta)
            logger.exception("Failed to store KV cache for request %s", req_id)
        finally:
            remaining = self.finish_store_job(req_meta) if tracked_request else None
            if tracked_request and remaining == 0:
                self.set_finished_request(req_id)
            if req_meta.event_id is not None:
                with self.completed_events_lock:
                    self.completed_events[req_meta.event_id] = 1
            self.request_queue.task_done()

    def _handle_store_job(self, request: ReqMeta) -> None:
        """Finish only after checked DMA completion, including ordinary pages."""
        self._inflight_request = request
        try:
            # Record durable progress at the same full-page boundary used by
            # the sender, including a prefix shortened by skipped drafting.
            coordinator = self.token_database.cache_coordinator
            assert coordinator is not None
            granularity = coordinator.lcm_block_size
            request.token_len_chunk = request.token_len_chunk // granularity * granularity
            request.save_end_token = min(request.save_end_token, request.token_len_chunk)
            try:
                if self.is_live_store_job(request) and request.can_save is not False:
                    source_ready = False
                    if request.boundary_puts:
                        source_ready = self._store_boundary_states(request)
                    if request.token_len_chunk:
                        if self._handle_stored_request(request, source_ready=source_ready):
                            self.record_saved_offset(request, request.token_len_chunk)
                            self._clear_retry_token_ids(request)
                        else:
                            self._remember_retry_token_ids(request)
            except StoreCapacityError:
                self._remember_retry_token_ids(request)
                logger.warning(
                    "Store capacity rejected save job %s for request %s",
                    request.store_job_id,
                    request.req_id,
                )
            # Retired queued jobs still report their own references releasable.
            self._finish_state_job(request)
            self._inflight_request = None
        finally:
            # Queue completion never substitutes for DMA completion. An unknown
            # SDK result propagates to run's fatal path and retains job refs.
            self.request_queue.task_done()

    def _store_boundary_states(self, request: ReqMeta) -> bool:
        """Store scheduler-pinned states using registered layout templates.

        The scheduler supplies committed dependency keys. This thread owns
        the copy fence and backend I/O before retiring source references.
        Return whether the source event was synchronized for this job.
        """
        if request.event_id is None or request.current_event is None:
            raise ValueError("boundary save requires pinned blocks and a copy-completion event")
        hashes = request.cache_key_hashes
        if hashes is None:
            raise ValueError("boundary save lacks committed model-dependent keys")
        keys: list[str] = []
        addresses: list[list[int]] = []
        sizes: list[list[int]] = []
        for put in request.boundary_puts or ():
            layout = self.state_store_layouts.get(put.group_id)
            if layout is None:
                raise ValueError(f"state Store layout is not registered for group {put.group_id}")
            key_hash = hashes.get(put.num_tokens)
            if not isinstance(key_hash, bytes) or not key_hash:
                raise ValueError("boundary hash is missing or not committed bytes")
            shards = layout.local_shard_ids
            block_ids = [put.block_id] * len(shards)
            group_addresses, group_sizes = layout.prepare_boundary_values(block_ids, shards)
            keys.extend(layout.key_for(shard, key_hash) for shard in shards)
            addresses.extend(group_addresses)
            sizes.extend(group_sizes)
        if not keys:
            raise ValueError("boundary save has no state objects")
        if len(set(keys)) != len(keys):
            raise ValueError("boundary save contains duplicate Store objects")
        # Match the native boundary-state save path: only missing objects
        # need a source fence and DMA, including partially published states.
        missing = self._get_missing_indices(keys, require_exists_check=True)
        if not missing:
            return False
        keys = [keys[i] for i in missing]
        addresses = [addresses[i] for i in missing]
        sizes = [sizes[i] for i in missing]
        request.current_event.synchronize()
        self.m_store.put_checked(keys, addresses, sizes)
        return True

    def _handle_stored_request(self, req_meta: ReqMeta, *, source_ready: bool = False) -> bool:
        """Store missing KV chunks; log validation/PUT failures and return False."""
        if req_meta.can_save is False:
            return False
        token_len = req_meta.token_len_chunk
        if self._use_store_jobs:
            coord = self.token_database.cache_coordinator
            assert coord is not None
            token_len = token_len // coord.lcm_block_size * coord.lcm_block_size
            if token_len == 0:
                return True
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        try:
            store_masks = self.token_database.store_mask(token_len, req_meta.num_prompt_tokens)
        except AssertionError as exc:
            if self._use_store_jobs:
                raise
            logger.debug("Skip AscendStore store mask for unaligned request %s: %s", req_id, exc)
            store_masks = None
        load_spec = req_meta.load_spec
        skip_start = load_spec.vllm_cached_tokens if load_spec is not None else 0
        skip_end = (
            (
                load_spec.kvpool_store_skip_tokens
                if load_spec.kvpool_store_skip_tokens is not None
                else load_spec.kvpool_cached_tokens
            )
            if load_spec is not None
            else 0
        )
        save_start_token = self.get_saved_offset(req_id)

        def should_skip(start: int, end: int, group_save_start: int) -> bool:
            # This is the worker's durable high-water mark and intentionally
            # may lag ReqMeta.save_start_token. In Decode-only mode the first
            # job therefore checks the complete prefix and can restore missing
            # ancestors of the new Decode block. After a successful store it
            # advances, avoiding a full-prefix lookup on later boundaries.
            if end <= group_save_start:
                return True
            return skip_end > skip_start and start >= skip_start and end <= skip_end

        for group_id in req_meta.kv_cache_group_ids or [0]:
            if self._use_store_jobs and group_id in self.state_store_layouts:
                continue
            group_block_size = self._get_block_size(group_id)

            group_store_mask = (
                store_masks[group_id] if store_masks is not None and group_id < len(store_masks) else None
            )
            if group_store_mask is not None and not group_store_mask:
                continue

            starts: list[int] = []
            ends: list[int] = []
            keys: list[str] = []
            block_hashes = []
            key_block_ids: list[int] = []
            block_ids = req_meta.block_ids_by_group[group_id]
            skip_null_blocks = self._skip_null_blocks(req_meta, group_id)
            align_state_group = group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]
            # Align-state slots below the saved extent can become valid later.
            save_start = 0 if align_state_group else save_start_token

            def chunk_filter(
                start: int,
                group_block_size=group_block_size,
                save_start=save_start,
            ) -> bool:
                return not should_skip(start, start + group_block_size, save_start)

            tp_replicas = self.put_step if self.dcp_size <= 1 and not align_state_group else 1
            if self._use_store_jobs:
                tp_replicas = self.tp_size
            # PCP=2, TP=4, KV heads=2, tp_replicas=2:
            # PCP  TP (KV 0 / KV 1)  shard_rank  filtered candidates
            #  0        0 / 2            0      0, 4, ...
            #  0        1 / 3            1      1, 5, ...
            #  1        0 / 2            2      2, 6, ...
            #  1        1 / 3            3      3, 7, ...
            iterator = self.token_database.process_token_key_strings_with_block_ids(
                token_len,
                req_meta.block_hashes,
                block_ids,
                kv_cache_group_id=group_id,
                skip_null_blocks=skip_null_blocks,
                chunk_filter=chunk_filter,
                shard_rank=self.pcp_rank * tp_replicas + self.tp_rank % tp_replicas,
                shard_size=self.pcp_size * tp_replicas,
                chunk_mask=group_store_mask,
            )
            for start, end, key, block_hash, block_id in iterator:
                if self._use_store_jobs:
                    bound_hash = (req_meta.cache_key_hashes or {}).get(end)
                    if bound_hash is None:
                        raise ValueError("page save boundary lacks its bound identity")
                    key = self.token_database.key_for(bound_hash, kv_cache_group_id=group_id)
                starts.append(start)
                ends.append(end)
                keys.append(key)
                if self.enable_kv_event:
                    block_hashes.append(block_hash)
                key_block_ids.append(block_id)

            if not keys:
                continue
            partitions = getattr(self.token_database, "partitions", None)
            if self.kv_role == "kv_consumer" and partitions is not None and len(partitions) > 1:
                # Decode consumers split one logical block across Prefill PP
                # ranks. A partial prior PUT must be retried, so require every
                # partition key to exist before skipping the logical block.
                exists_states = self.lookup_store_keys(keys, kv_cache_group_id=group_id)
                if exists_states is None:
                    return False
                missing_indices = [index for index, exists in enumerate(exists_states) if not exists]
            else:
                missing_indices = self._get_missing_indices(keys, require_exists_check=self.enable_kv_event)
            if not missing_indices:
                continue
            starts = [starts[index] for index in missing_indices]
            ends = [ends[index] for index in missing_indices]
            keys = [keys[index] for index in missing_indices]
            if self.enable_kv_event:
                block_hashes = [block_hashes[index] for index in missing_indices]
            key_block_ids = [key_block_ids[index] for index in missing_indices]

            logger.debug(
                "Storing KV cache for %d out of %d blocks for request %s in group %d",
                len(keys),
                token_len // group_block_size,
                req_id,
                group_id,
            )
            addrs = []
            sizes = []
            stored_events: list[BlockStored] = []
            all_hashes = []
            if self.enable_kv_event:
                group_block_hashes = get_block_hashes(
                    req_meta.block_hashes,
                    group_block_size,
                    getattr(self.token_database, "hash_block_size", group_block_size),
                )
                all_hashes = [maybe_convert_block_hash(bh) for bh in group_block_hashes]
            logger.debug(
                "KV pool put request=%s group=%d token_len=%d keys=%d sample_keys=%s",
                req_id,
                group_id,
                token_len,
                len(keys),
                keys[:3],
            )
            for index, start in enumerate(starts):
                addr, size, _ = self._prepare_value(
                    start,
                    ends[index],
                    block_ids,
                    kv_cache_group_id=group_id,
                    block_id=key_block_ids[index],
                )
                addrs.append(addr)
                sizes.append(size)
                if self.enable_kv_event:
                    token_ids = self.get_event_token_ids(req_meta, start, ends[index])
                    block_size = (
                        req_meta.original_block_size[group_id]
                        if isinstance(req_meta.original_block_size, list)
                        else req_meta.original_block_size
                    )
                    if block_size is not None:
                        block_idx = start // group_block_size
                        if block_idx >= len(all_hashes):
                            continue
                        current_hash = all_hashes[block_idx]
                        parent_hash = all_hashes[block_idx - 1] if block_idx > 0 else None
                        stored_event = BlockStored(
                            block_hashes=[current_hash],
                            parent_block_hash=parent_hash,
                            token_ids=token_ids,
                            block_size=block_size,
                            lora_id=None,
                            medium="cpu",
                            lora_name=None,
                        )
                        stored_events.append(stored_event)
                        logger.debug("Added kv cache event '%s' to kv cache events queue", stored_event)

            logical_key_count = len(keys)
            if self.kv_role == "kv_consumer":
                keys, addrs, sizes = self._decode_adaptor_prefill_pp(
                    keys,
                    addrs,
                    sizes,
                    kv_cache_group_id=group_id,
                )
            if not keys or len(keys) % logical_key_count != 0:
                logger.error(
                    "Invalid PP key expansion for request %s: logical=%d physical=%d",
                    req_id,
                    logical_key_count,
                    len(keys),
                )
                return False
            if current_event is not None and not source_ready:
                current_event.synchronize()
                source_ready = True
            if self._use_store_jobs:
                try:
                    self.m_store.put_checked(keys, addrs, sizes)
                except StoreCapacityError as exc:
                    # Preserve per-key event reporting for completed PP shards.
                    shards_per_key = len(keys) // logical_key_count
                    if self.enable_kv_event and len(stored_events) == logical_key_count:
                        self.update_kv_event(
                            [
                                event
                                for index, event in enumerate(stored_events)
                                if all(
                                    status == 0
                                    for status in exc.results[index * shards_per_key : (index + 1) * shards_per_key]
                                )
                            ]
                        )
                    raise
                put_result = None
            else:
                put_result = self.m_store.put(keys, addrs, sizes)
            if isinstance(put_result, list):
                if len(put_result) != len(keys):
                    logger.error("KV store backend returned %s results for %s keys", len(put_result), len(keys))
                    return False
                if not all(put_result):
                    if len(put_result) % logical_key_count != 0:
                        logger.error(
                            "KV store returned %s physical results for %s logical keys",
                            len(put_result),
                            logical_key_count,
                        )
                        return False
                    physical_keys_per_logical = len(put_result) // logical_key_count
                    logical_success = [
                        all(put_result[start : start + physical_keys_per_logical])
                        for start in range(0, len(put_result), physical_keys_per_logical)
                    ]
                    if self.enable_kv_event and len(stored_events) == len(logical_success):
                        self.update_kv_event(
                            [
                                event
                                for event, succeeded in zip(stored_events, logical_success, strict=True)
                                if succeeded
                            ]
                        )
                    logger.error("KV store backend partially failed to put request %s", req_id)
                    return False
            elif put_result is False:
                logger.error("KV store backend failed to put request %s", req_id)
                return False
            if self.enable_kv_event and stored_events:
                self.update_kv_event(stored_events)
        return True


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
        worker: Any = None,
        record_operation: Callable[[str, float, int], None] | None = None,
        *,
        state_store_layouts: dict[int, AcceptedStateStoreLayout] | None = None,
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
        self.worker = worker
        self._record_operation_cb = record_operation
        self.state_store_layouts = dict(state_store_layouts or {})
        self._use_state_layouts = state_store_layouts is not None

    def take_receive_results(self) -> tuple[set[str], set[str]]:
        """One atomic snapshot: a failed receive must also finish its wait."""
        with self.done_task_lock:
            finished, failed = self.finished_requests, self.failed_requests
            self.finished_requests, self.failed_requests = set(), set()
            return finished, failed

    def _receive_states_and_pages(self, request: ReqMeta) -> None:
        """Native load_mask + StoreLayout addressing over core-owned targets."""
        load = request.load_spec
        if load is None or load.cache_key_hashes is None:
            raise ValueError("state receive lacks the lookup's bound identities")
        token_len = load.kvpool_cached_tokens
        masks = cast(tuple[list[bool], ...], self.token_database.load_mask(request.block_hashes, token_len))
        keys: list[str] = []
        addresses: list[list[int]] = []
        sizes: list[list[int]] = []
        state_destinations: dict[int, list[int]] = {}
        for group_id in request.kv_cache_group_ids or [0]:
            group_mask = masks[group_id]
            if not any(group_mask):
                continue
            blocks = request.block_ids_by_group[group_id]
            block_size = self._get_block_size(group_id)
            if group_id in self.state_store_layouts:
                layout = self.state_store_layouts[group_id]
                chunks, shards = [], []
                for index, present in enumerate(group_mask):
                    boundary = (index + 1) * block_size
                    if not present or boundary <= load.vllm_cached_tokens:
                        continue
                    key_hash = load.cache_key_hashes.get(boundary)
                    if key_hash is None:
                        raise ValueError("receive boundary was not bound during lookup")
                    for shard in layout.local_shard_ids:
                        keys.append(layout.key_for(shard, key_hash))
                        chunks.append((boundary - block_size, boundary))
                        shards.append(shard)
                group_addresses, group_sizes, destinations = layout.prepare_values(chunks, blocks, shards)
                addresses.extend(group_addresses)
                sizes.extend(group_sizes)
                state_destinations[group_id] = destinations
                continue

            def chunk_filter(start, mask=group_mask, size=block_size):
                index = start // size
                return index < len(mask) and mask[index]

            for start, end, key, _, block_id in self.token_database.process_token_key_strings_with_block_ids(
                token_len,
                request.block_hashes,
                blocks,
                load.vllm_cached_tokens,
                kv_cache_group_id=group_id,
                chunk_filter=chunk_filter,
            ):
                bound_hash = load.cache_key_hashes.get(end)
                if bound_hash is None:
                    raise ValueError("page receive boundary lacks its bound identity")
                key = self.token_database.key_for(bound_hash, kv_cache_group_id=group_id)
                addr, size, _ = self._prepare_value(start, end, blocks, kv_cache_group_id=group_id, block_id=block_id)
                keys.append(key)
                addresses.append(addr)
                sizes.append(size)
        if not keys:
            raise ValueError("external hit produced no receive descriptors")
        if request.current_event is None:
            raise ValueError("receive requires the destination initialization fence")
        request.current_event.synchronize()
        started = time.perf_counter() if self._record_operation_cb is not None else 0.0
        counts = require_aligned_batch_results("get_checked", keys, self.m_store.get_checked(keys, addresses, sizes))
        if self._record_operation_cb is not None:
            self._record_operation_cb("load_get", time.perf_counter() - started, len(keys))
        expected = [sum(parts) for parts in sizes]
        if any(actual > length for actual, length in zip(counts, expected, strict=True)):
            raise TransferCompletionUnknown("GET reported more bytes than the registered destination")
        failed = counts != expected
        if not failed:
            device_work = False
            for group_id, destinations in state_destinations.items():
                device_work |= self.state_store_layouts[group_id].initialize_received_scratch(destinations)
            if device_work:
                torch.npu.current_stream().synchronize()
        with self.done_task_lock:
            if failed:
                self.failed_requests.add(request.req_id)
            self.finished_requests.add(request.req_id)

    def _discard_request(self, request: Any) -> None:
        if self._use_state_layouts and isinstance(request, ReqMeta):
            with self.done_task_lock:
                self.failed_requests.add(request.req_id)
                self.finished_requests.add(request.req_id)

    def _handle_request_exception(self, request_data: Any):
        self._discard_queued_requests()

    def _handle_request(self, req_meta: ReqMeta):
        try:
            if self._use_state_layouts:
                self._inflight_request = req_meta
                self._receive_states_and_pages(req_meta)
                self._inflight_request = None
                return
            load_spec = req_meta.load_spec
            req_id = req_meta.req_id
            if load_spec is None:
                logger.error("KV pool async recv request %s has no load spec; skip load.", req_id)
                self.set_finished_request(req_id)
                return

            token_len = load_spec.token_len
            if self.worker is not None and getattr(self.worker, "tp_mismatch", False):
                group_block_size = self._get_block_size(0)
                mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size
                loaded = self.worker._load_kv_tp_mismatch(
                    req_meta.block_hashes,
                    req_meta.block_ids_by_group[0],
                    token_len,
                    mask_num,
                )
                if loaded is False:
                    self._mark_failed_requests([req_meta])
                self.set_finished_request(req_id)
                return

            addr_list = []
            size_list = []
            key_list = []
            block_id_list: list[int] = []
            group_ids = req_meta.kv_cache_group_ids or [0]
            load_masks = self.token_database.load_mask(req_meta.block_hashes, token_len)
            for group_id in group_ids:
                block_ids = req_meta.block_ids_by_group[group_id]
                group_block_size = self._get_block_size(group_id)
                mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size

                def chunk_filter(start: int, group_id=group_id) -> bool:
                    return self.token_database.mask_allows_chunk(load_masks, group_id, start)

                token_iter = self.token_database.process_token_key_strings_with_block_ids(
                    token_len,
                    req_meta.block_hashes,
                    block_ids,
                    mask_num,
                    kv_cache_group_id=group_id,
                    skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                    chunk_filter=chunk_filter,
                )
                for start, end, key, _block_hash, block_id in token_iter:
                    addr, size, block_id = self._prepare_value(
                        start,
                        end,
                        block_ids,
                        kv_cache_group_id=group_id,
                        block_id=block_id,
                    )
                    key_list.append(key)
                    addr_list.append(addr)
                    size_list.append(size)
                    block_id_list.append(block_id)
            if not key_list:
                self.set_finished_request(req_id)
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
            load_get_start = time.perf_counter() if self._record_operation_cb is not None else 0.0
            ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
            if self._record_operation_cb is not None:
                self._record_operation_cb(
                    "load_get",
                    time.perf_counter() - load_get_start,
                    len(key_list_c),
                )
            if isinstance(self.m_store, MemcacheBackend) and (
                ret is None or len(ret) != len(key_list_c) or any(r != 0 for r in ret)
            ):
                self._mark_failed_requests([req_meta])
                self.set_finished_request(req_id)
                return
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
        finally:
            self.request_queue.task_done()


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
