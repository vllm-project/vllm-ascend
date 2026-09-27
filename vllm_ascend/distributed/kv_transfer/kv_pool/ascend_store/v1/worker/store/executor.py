"""FIFO execution of Store tasks on one background thread."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

from .task import StoreChunk, StoreTask

STORE_BARRIER_POLL_INTERVAL_S = 1.0


@dataclass(frozen=True, slots=True)
class StoreExecutionResult:
    """Raw Backend result and source-release evidence for one Store task."""

    request_id: str
    result_codes: tuple[int, ...] | None
    error: Exception | None = None

    @property
    def source_released(self) -> bool:
        return self.error is None and self.result_codes is not None and all(code == 0 for code in self.result_codes)


@dataclass(slots=True)
class StoreBatch:
    """Own one submitted Store batch and its asynchronous completion facts."""

    tasks: tuple[StoreTask, ...]
    completed: threading.Event = field(default_factory=threading.Event)
    results: list[StoreExecutionResult] = field(default_factory=list)


class StoreExecutor(threading.Thread):
    """Execute Store tasks in FIFO order and track batch completion."""

    def __init__(self, backend: Backend) -> None:
        super().__init__(daemon=True, name="KVCacheSendingThread")
        self._backend = backend
        self._ready = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._has_started = False
        self._closed = False
        self._task_queue: queue.Queue[StoreBatch | None] = queue.Queue()
        self._fatal_error: BaseException | None = None
        self._previous_batch: StoreBatch | None = None

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
            batch = StoreBatch(tuple(tasks))
            self._task_queue.put(batch)
            self._previous_batch = batch

    def wait_for_previous_store(self) -> tuple[StoreExecutionResult, ...]:
        batch = self._previous_batch
        if batch is None:
            return ()
        while True:
            self.raise_if_failed()
            if batch.completed.wait(timeout=STORE_BARRIER_POLL_INTERVAL_S):
                break
        self.raise_if_failed()
        self._previous_batch = None
        return tuple(batch.results)

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(f"{self.name} failed during asynchronous transfer") from self._fatal_error

    def _raise_if_not_running(self) -> None:
        self.raise_if_failed()
        if not self._has_started:
            raise RuntimeError(f"{self.name} has not started")
        if self._closed:
            raise RuntimeError(f"{self.name} is closed")

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
            batch = self._task_queue.get()
            try:
                if batch is None:
                    return
                self._execute_batch(batch)
            except Exception as error:
                self._fatal_error = error
                logger.exception("Error in KVCacheSendingThread")
                return
            finally:
                if batch is not None:
                    batch.completed.set()
                self._task_queue.task_done()

    def _execute_batch(self, batch: StoreBatch) -> None:
        for task in batch.tasks:
            result = self._execute_task(task)
            batch.results.append(result)
            self._raise_for_incomplete_store(result)

    def _execute_task(self, task: StoreTask) -> StoreExecutionResult:
        try:
            chunks = self._select_missing_chunks(task)
            if not chunks:
                return StoreExecutionResult(task.request_id, ())

            task.source_ready_event.synchronize()
            result_codes = self._backend.put(
                [chunk.backend_key for chunk in chunks],
                [list(chunk.addresses) for chunk in chunks],
                [list(chunk.sizes) for chunk in chunks],
            )
            codes = None if result_codes is None else tuple(result_codes)
            if codes is not None and len(codes) != len(chunks):
                error = RuntimeError(f"Store returned {len(codes)} results for {len(chunks)} chunks")
                return StoreExecutionResult(task.request_id, codes, error)
            return StoreExecutionResult(task.request_id, codes)
        except Exception as error:
            return StoreExecutionResult(task.request_id, None, error)

    def _select_missing_chunks(self, task: StoreTask) -> tuple[StoreChunk, ...]:
        if not task.chunks or not self._backend.requires_exists_before_put:
            return task.chunks

        keys = [chunk.backend_key for chunk in task.chunks]
        present = self._backend.exists(keys)
        if len(present) != len(keys):
            raise RuntimeError(f"Store exists returned {len(present)} results for {len(keys)} chunks")
        if any(value not in (0, 1) for value in present):
            raise RuntimeError("Store exists returned states other than 0 or 1")
        return tuple(chunk for chunk, value in zip(task.chunks, present, strict=True) if value != 1)

    @staticmethod
    def _raise_for_incomplete_store(result: StoreExecutionResult) -> None:
        if result.source_released:
            return
        if result.error is not None:
            raise RuntimeError(f"Store failed for request {result.request_id}") from result.error
        if result.result_codes is None:
            raise RuntimeError(f"Store result is unknown for request {result.request_id}")
        failed_codes = [code for code in result.result_codes if code != 0]
        raise RuntimeError(f"Store failed for request {result.request_id} with result codes {failed_codes}")
