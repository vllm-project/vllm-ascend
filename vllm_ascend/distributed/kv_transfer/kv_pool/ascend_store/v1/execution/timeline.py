"""Temporal policies for scheduling fixed KV Pool operations."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from vllm.logger import logger

from ..graph.elements import BindingBatch, KVBinding
from .io import BindingEvidence, StoreEvidence

_STORE_FENCE_POLL_INTERVAL_S = 1.0


@dataclass(frozen=True, slots=True)
class LoadTransfer:
    """One request's immutable Load work after spatial projection."""

    request_id: str
    traversal: tuple[KVBinding, ...]


@dataclass(frozen=True, slots=True)
class LoadCompletion:
    """Load evidence that became visible at one point in time."""

    request_id: str
    binding_evidence: tuple[BindingEvidence, ...]


class LoadTimeline(Protocol):
    """Control when fixed Load work is executed and made visible.

    Unexpected execution exceptions terminate the timeline. They are not
    request-level Backend outcomes and the KV Pool graph must stop accepting work.
    """

    is_deferred: bool

    def attach_operation(self, operation: Callable[[LoadTransfer], LoadCompletion]) -> None: ...

    def start(self) -> None: ...

    def submit(self, transfers: list[LoadTransfer]) -> Iterable[LoadCompletion]: ...

    def collect(self) -> Iterable[LoadCompletion]: ...

    def close(self) -> None: ...


class SynchronousLoadTimeline:
    """Execute and publish Load completions in the submitting call."""

    is_deferred = False

    def __init__(self) -> None:
        self._operation: Callable[[LoadTransfer], LoadCompletion] | None = None

    def attach_operation(self, operation: Callable[[LoadTransfer], LoadCompletion]) -> None:
        if self._operation is not None:
            raise RuntimeError("Synchronous Load timeline operation is already attached")
        self._operation = operation

    def start(self) -> None:
        return

    def submit(self, transfers: list[LoadTransfer]) -> Iterable[LoadCompletion]:
        if self._operation is None:
            raise RuntimeError("Synchronous Load timeline has not started")
        return tuple(self._operation(transfer) for transfer in transfers)

    def collect(self) -> tuple[LoadCompletion, ...]:
        return ()

    def close(self) -> None:
        return


class AsynchronousLoadTimeline(threading.Thread):
    """Execute fixed Load work on one background timeline."""

    is_deferred = True

    def __init__(self, thread_initializer: Callable[[], None]) -> None:
        super().__init__(daemon=True, name="KVCacheLoadThread")
        self._thread_initializer = thread_initializer
        self._operation: Callable[[LoadTransfer], LoadCompletion] | None = None
        self._ready = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._has_started = False
        self._closed = False
        self._completed_lock = threading.Lock()
        self._queue: queue.Queue[LoadTransfer | None] = queue.Queue()
        self._completed: list[LoadCompletion] = []
        self._fatal_error: BaseException | None = None
        self._aborted_request_ids: tuple[str, ...] = ()

    def attach_operation(self, operation: Callable[[LoadTransfer], LoadCompletion]) -> None:
        with self._lifecycle_lock:
            if self._operation is not None:
                raise RuntimeError(f"{self.name} operation is already attached")
            self._operation = operation

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError(f"{self.name} is closed")
            if not self._has_started:
                if self._operation is None:
                    raise RuntimeError(f"{self.name} has no bound operation")
                super().start()
                self._has_started = True
        self._ready.wait()
        self._raise_if_failed()

    def submit(self, transfers: list[LoadTransfer]) -> tuple[LoadCompletion, ...]:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            for transfer in transfers:
                self._queue.put(transfer)
        return ()

    def collect(self) -> list[LoadCompletion]:
        self._raise_if_failed()
        with self._completed_lock:
            completed = self._completed
            self._completed = []
        return completed

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            if not self._has_started:
                return
            if self.is_alive():
                self._queue.put(None)
        self.join()
        self._raise_if_failed()

    def run(self) -> None:
        try:
            self._thread_initializer()
        except BaseException as error:
            self._terminate(error)
            logger.exception("Failed to start KVCacheLoadThread")
        finally:
            self._ready.set()
        if self._fatal_error is not None:
            return

        while True:
            transfer = self._queue.get()
            try:
                if transfer is None:
                    return
                if self._operation is None:
                    raise RuntimeError("Asynchronous Load timeline has no operation")
                completion = self._operation(transfer)
                with self._completed_lock:
                    self._completed.append(completion)
            except Exception as error:
                self._terminate(error, transfer.request_id if transfer is not None else None)
                logger.exception("Error in KVCacheLoadThread")
                return
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            requests = f"; unfinished requests: {list(self._aborted_request_ids)}" if self._aborted_request_ids else ""
            raise RuntimeError(f"{self.name} terminated during asynchronous Load{requests}") from self._fatal_error

    def _raise_if_not_running(self) -> None:
        self._raise_if_failed()
        if not self._has_started:
            raise RuntimeError(f"{self.name} has not started")
        if self._closed:
            raise RuntimeError(f"{self.name} is closed")

    def _terminate(self, error: BaseException, current_request_id: str | None = None) -> None:
        with self._lifecycle_lock:
            aborted_request_ids = {current_request_id} if current_request_id is not None else set()
            while True:
                try:
                    transfer = self._queue.get_nowait()
                except queue.Empty:
                    break
                if transfer is not None:
                    aborted_request_ids.add(transfer.request_id)
                self._queue.task_done()
            self._aborted_request_ids = tuple(sorted(aborted_request_ids))
            self._fatal_error = error


@dataclass(frozen=True, slots=True)
class StoreTransfer:
    """One request's immutable Store work and source-ready fence."""

    request_id: str
    batches: tuple[BindingBatch, ...]
    source_ready_event: Any


@dataclass(frozen=True, slots=True)
class StoreCompletion:
    """Store completion with transfer and source-release facts kept separate."""

    request_id: str
    evidence: StoreEvidence


@dataclass(slots=True)
class StoreBatch:
    """One accepted step batch and its exact completion fence."""

    transfers: tuple[StoreTransfer, ...]
    completed: threading.Event = field(default_factory=threading.Event)
    completions: list[StoreCompletion] = field(default_factory=list)


class StoreTimeline(threading.Thread):
    """Preserve FIFO Store submission and next-step fencing."""

    def __init__(self, thread_initializer: Callable[[], None]) -> None:
        super().__init__(daemon=True, name="KVCacheStoreThread")
        self._thread_initializer = thread_initializer
        self._operation: Callable[[StoreTransfer], StoreCompletion] | None = None
        self._ready = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._has_started = False
        self._closed = False
        self._queue: queue.Queue[StoreBatch | None] = queue.Queue()
        self._fatal_error: BaseException | None = None

    def attach_operation(self, operation: Callable[[StoreTransfer], StoreCompletion]) -> None:
        with self._lifecycle_lock:
            if self._operation is not None:
                raise RuntimeError(f"{self.name} operation is already attached")
            self._operation = operation

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError(f"{self.name} is closed")
            if not self._has_started:
                if self._operation is None:
                    raise RuntimeError(f"{self.name} has no bound operation")
                super().start()
                self._has_started = True
        self._ready.wait()
        self._raise_if_failed()

    def submit(self, transfers: list[StoreTransfer]) -> StoreBatch:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            batch = StoreBatch(tuple(transfers))
            self._queue.put(batch)
        return batch

    def wait(self, batch: StoreBatch) -> tuple[StoreCompletion, ...]:
        while True:
            self._raise_if_failed()
            if batch.completed.wait(timeout=_STORE_FENCE_POLL_INTERVAL_S):
                break
        self._raise_if_failed()
        return tuple(batch.completions)

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            if not self._has_started:
                return
            if self.is_alive():
                self._queue.put(None)
        self.join()
        self._raise_if_failed()

    def run(self) -> None:
        try:
            self._thread_initializer()
        except BaseException as error:
            self._fatal_error = error
            logger.exception("Failed to start KVCacheStoreThread")
        finally:
            self._ready.set()
        if self._fatal_error is not None:
            return

        while True:
            batch = self._queue.get()
            try:
                if batch is None:
                    return
                if self._operation is None:
                    raise RuntimeError("Store timeline has no operation")
                for transfer in batch.transfers:
                    batch.completions.append(self._operation(transfer))
            except BaseException as error:
                self._fatal_error = error
                logger.exception("Error in KVCacheStoreThread")
                return
            finally:
                if batch is not None:
                    batch.completed.set()
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(f"{self.name} failed during asynchronous Store") from self._fatal_error

    def _raise_if_not_running(self) -> None:
        self._raise_if_failed()
        if not self._has_started:
            raise RuntimeError(f"{self.name} has not started")
        if self._closed:
            raise RuntimeError(f"{self.name} is closed")
