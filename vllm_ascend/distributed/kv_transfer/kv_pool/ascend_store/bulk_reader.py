# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-level asynchronous restore with synchronous, drainable backend I/O."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.executor import PoolBackendExecutor, PoolReadRange


@dataclass(frozen=True)
class PoolLoadResult:
    request_id: str
    attempt_id: str
    success: bool
    cancelled: bool
    block_ids: frozenset[int]
    error: str = ""
    copied_bytes: int = 0


@dataclass
class PoolReadHandle:
    request_id: str
    attempt_id: str
    block_ids: frozenset[int]
    cancelled: threading.Event
    future: Future[None] | None = None
    planned_bytes: int = 0


class BulkReader:
    def __init__(self, executor: PoolBackendExecutor):
        self.executor = executor
        self._lock = threading.RLock()
        self._active: dict[str, PoolReadHandle] = {}
        self._closed = False
        self._queue = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pool-bulk-read", initializer=executor.backend.set_device
        )

    def submit(
        self,
        request_id: str,
        attempt_id: str,
        plan: list[PoolReadRange],
        block_ids: frozenset[int],
    ) -> PoolReadHandle:
        if not all(isinstance(value, str) and value for value in (request_id, attempt_id)):
            raise ValueError("Pool reads require request and attempt identities")
        if not isinstance(block_ids, frozenset) or any(type(block) is not int or block < 0 for block in block_ids):
            raise ValueError("Pool reads require an immutable set of destination block IDs")
        handle = PoolReadHandle(request_id, attempt_id, block_ids, threading.Event())
        handle.planned_bytes = sum(item.size for item in plan)
        with self._lock:
            if self._closed:
                raise RuntimeError("Pool bulk reader is closed")
            if request_id in self._active:
                raise ValueError("Previous request restore must drain and be collected before retry")
            self._active[request_id] = handle
            try:
                handle.future = self._queue.submit(self._restore, handle, tuple(plan))
            except BaseException:
                del self._active[request_id]
                raise
        return handle

    def _restore(self, handle: PoolReadHandle, plan: tuple[PoolReadRange, ...]) -> None:
        if handle.cancelled.is_set():
            return
        # Executor.restore only returns once all native copies and read-session
        # cleanup have completed. Cancellation therefore never reports a block
        # reusable while native I/O might still write to that block.
        self.executor.restore(list(plan))

    def cancel(self, request_id: str) -> None:
        with self._lock:
            handle = self._active.get(request_id)
            if handle is not None:
                handle.cancelled.set()

    def active_attempt(self, request_id: str) -> str | None:
        with self._lock:
            handle = self._active.get(request_id)
            return handle.attempt_id if handle is not None else None

    def poll_results(self) -> list[PoolLoadResult]:
        results = []
        with self._lock:
            for request_id, handle in list(self._active.items()):
                future = handle.future
                if future is None or not future.done():
                    continue
                error = future.exception()
                cancelled = handle.cancelled.is_set()
                results.append(
                    PoolLoadResult(
                        request_id,
                        handle.attempt_id,
                        error is None and not cancelled,
                        cancelled,
                        handle.block_ids,
                        str(error) if error is not None else "",
                        handle.planned_bytes if error is None and not cancelled else 0,
                    )
                )
                del self._active[request_id]
        return results

    def close(self) -> list[PoolLoadResult]:
        with self._lock:
            self._closed = True
            for handle in self._active.values():
                handle.cancelled.set()
        self._queue.shutdown(wait=True)
        return self.poll_results()
