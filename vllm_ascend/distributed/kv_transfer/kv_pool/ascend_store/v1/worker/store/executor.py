"""FIFO execution of Store tasks on one background thread."""

from __future__ import annotations

import queue
import threading
from collections import defaultdict

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

from .task import StoreChunk, StoreTask

STORE_BARRIER_POLL_INTERVAL_S = 1.0


class StoreBatchBarrier:
    """FIFO marker following every task in one scheduled Store batch."""

    def __init__(self) -> None:
        self.completed = threading.Event()


class StoreExecutor(threading.Thread):
    """Execute Store tasks in FIFO order and track batch completion."""

    def __init__(self, backend: Backend) -> None:
        super().__init__(daemon=True, name="KVCacheSendingThread")
        self._backend = backend
        self._ready = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._has_started = False
        self._closed = False
        self._completion_lock = threading.Lock()
        self._task_queue: queue.Queue[StoreTask | StoreBatchBarrier | None] = queue.Queue()
        self._pending_task_counts: defaultdict[str, int] = defaultdict(int)
        self._completed_request_ids: set[str] = set()
        self._fatal_error: BaseException | None = None
        self._previous_batch_barrier: StoreBatchBarrier | None = None

    def start_and_wait_ready(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError(f"{self.name} is closed")
            if not self._has_started:
                self.start()
                self._has_started = True
        self._ready.wait()
        self.raise_if_failed()

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            if not self._has_started:
                return
            if self.is_alive():
                self._task_queue.put(None)
        self.join()
        self.raise_if_failed()

    def submit_batch(self, tasks: list[StoreTask]) -> None:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            batch_barrier = StoreBatchBarrier()
            # Register every task before the thread may complete the first one.
            with self._completion_lock:
                for task in tasks:
                    self._completed_request_ids.discard(task.request_id)
                    self._pending_task_counts[task.request_id] += 1
            for task in tasks:
                self._task_queue.put(task)
            self._task_queue.put(batch_barrier)
            self._previous_batch_barrier = batch_barrier

    def wait_for_previous_store(self) -> None:
        batch_barrier = self._previous_batch_barrier
        if batch_barrier is None:
            return
        while True:
            self.raise_if_failed()
            if batch_barrier.completed.wait(timeout=STORE_BARRIER_POLL_INTERVAL_S):
                break
        self._previous_batch_barrier = None

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(f"{self.name} failed during asynchronous transfer") from self._fatal_error

    def _raise_if_not_running(self) -> None:
        self.raise_if_failed()
        if not self._has_started:
            raise RuntimeError(f"{self.name} has not started")
        if self._closed:
            raise RuntimeError(f"{self.name} is closed")

    def finish_step(self, preempted_request_ids: frozenset[str]) -> None:
        """Forget preempted Stores and consume completion bookkeeping for this step."""
        for request_id in preempted_request_ids:
            self._discard_pending_request(request_id)
        self._discard_completed_requests(preempted_request_ids)
        self._take_completed_request_ids()

    def _discard_completed_requests(self, request_ids: frozenset[str]) -> None:
        with self._completion_lock:
            self._completed_request_ids -= request_ids

    def _take_completed_request_ids(self) -> set[str]:
        with self._completion_lock:
            completed_request_ids = self._completed_request_ids.copy()
            self._completed_request_ids.clear()
            return completed_request_ids

    def _discard_pending_request(self, request_id: str) -> None:
        with self._completion_lock:
            self._pending_task_counts.pop(request_id, None)

    def run(self) -> None:
        try:
            self._backend.set_device()
        except BaseException as error:
            self._fatal_error = error
            logger.exception("Failed to start KVCacheSendingThread")
        finally:
            self._ready.set()
        if self._fatal_error is not None:
            return

        while True:
            task = self._task_queue.get()
            try:
                if task is None:
                    return
                self._handle_task(task)
            except Exception as error:
                self._fatal_error = error
                logger.exception("Error in KVCacheSendingThread")
                return
            finally:
                self._task_queue.task_done()

    def _handle_task(self, task: StoreTask | StoreBatchBarrier) -> None:
        if isinstance(task, StoreBatchBarrier):
            task.completed.set()
            return

        request_id = task.request_id
        with self._completion_lock:
            tracked_request = request_id in self._pending_task_counts
        try:
            if tracked_request:
                self._execute_task(task)
        except Exception:
            logger.exception("Failed to store KV cache for request %s", request_id)
        finally:
            with self._completion_lock:
                if tracked_request and request_id in self._pending_task_counts:
                    self._pending_task_counts[request_id] -= 1
                    if self._pending_task_counts[request_id] == 0:
                        del self._pending_task_counts[request_id]
                        self._completed_request_ids.add(request_id)

    def _execute_task(self, task: StoreTask) -> None:
        chunks = self._select_missing_chunks(task)
        if not chunks:
            return

        task.source_ready_event.synchronize()
        self._backend.put(
            [chunk.backend_key for chunk in chunks],
            [list(chunk.addresses) for chunk in chunks],
            [list(chunk.sizes) for chunk in chunks],
        )

    def _select_missing_chunks(self, task: StoreTask) -> tuple[StoreChunk, ...]:
        if not task.chunks or not self._backend.requires_exists_before_put:
            return task.chunks

        keys = [chunk.backend_key for chunk in task.chunks]
        try:
            present = self._backend.exists(keys)
            exists = [False] * len(keys)
            for index, value in enumerate(present):
                exists[index] = value == 1
            return tuple(chunk for index, chunk in enumerate(task.chunks) if not exists[index])
        except Exception:
            return task.chunks
