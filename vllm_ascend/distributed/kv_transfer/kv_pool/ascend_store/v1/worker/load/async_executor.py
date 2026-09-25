"""Background execution for fully resolved classic Load tasks."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

from .executor import LoadExecutor, LoadTaskResult
from .task import LoadTask


class AsyncLoadExecutor(threading.Thread):
    """Execute Load tasks on one background thread and retain their results."""

    def __init__(self, backend: Backend) -> None:
        super().__init__(daemon=True, name="KVCacheLoadThread")
        self._backend = backend
        self._load_executor = LoadExecutor(backend)
        self._ready = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._has_started = False
        self._closed = False
        self._completed_lock = threading.Lock()
        self._task_queue: queue.Queue[LoadTask | None] = queue.Queue()
        self._completed: dict[str, LoadTaskResult] = {}
        self._fatal_error: BaseException | None = None

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

    def submit(self, tasks: list[LoadTask]) -> Iterable[LoadTaskResult]:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            for task in tasks:
                self._task_queue.put(task)
        return ()

    def collect(self) -> list[LoadTaskResult]:
        self.raise_if_failed()
        with self._completed_lock:
            completed = list(self._completed.values())
            self._completed.clear()
        return completed

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(f"{self.name} failed during asynchronous Load") from self._fatal_error

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
            logger.exception("Failed to start KVCacheLoadThread")
        finally:
            self._ready.set()
        if self._fatal_error is not None:
            return

        while True:
            task = self._task_queue.get()
            try:
                if task is None:
                    return
                result = self._load_executor.execute(task)
                with self._completed_lock:
                    self._completed[task.request_id] = (task, result)
            except Exception as error:
                self._fatal_error = error
                logger.exception("Error in KVCacheLoadThread")
                return
            finally:
                self._task_queue.task_done()
