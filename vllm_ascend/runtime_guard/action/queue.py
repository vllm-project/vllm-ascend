#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Background worker for runtime_guard async sinks (file write)."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

# Sentinel to stop the worker loop.
_STOP = object()


class ActionQueue:
    """Single daemon thread consuming sync-prepared sink jobs.

    Hot path should only ``submit`` CPU-side payloads (already snapshotted).
    Heavy disk work runs here so inference is not blocked.

    ``start`` / ``stop`` / ``submit`` share one lock so stop can drain and
    deliver ``_STOP`` without racing concurrent puts, and so submit cannot
    resurrect a worker mid-teardown.

    Optional ``dedupe_key``: while a job with that key is queued or running,
    another submit with the same key is dropped (same wave + req_id reports).
    """

    def __init__(self, *, maxsize: int = 64, name: str = "runtime-guard-actions") -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._name = name
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopping = False
        self._lock = threading.Lock()
        self._pending_keys: set[Any] = set()

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        with self._lock:
            self._start_unlocked()

    def _start_unlocked(self) -> None:
        if self._started or self._stopping:
            return
        self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
        self._thread.start()
        self._started = True
        logger.info(
            "[runtime_guard] action worker started name=%s maxsize=%d",
            self._name,
            self._queue.maxsize,
        )

    def stop(self, *, timeout: float = 2.0) -> None:
        with self._lock:
            if not self._started:
                return
            self._stopping = True
            # Stop is explicit teardown: pending jobs are expendable. Drain to
            # make room for the sentinel so the worker always gets it.
            dropped = 0
            while True:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    dropped += 1
                except queue.Empty:
                    break
            self._pending_keys.clear()
            if dropped:
                logger.warning(
                    "[runtime_guard] action worker stop: dropped %d pending job(s) name=%s",
                    dropped,
                    self._name,
                )
            # Holding the lock: no concurrent submit can refill between drain
            # and this put, so the sentinel cannot be blocked by Full.
            self._queue.put_nowait(_STOP)
            thread = self._thread
            self._thread = None
            self._started = False
        if thread is not None:
            thread.join(timeout=timeout)
        with self._lock:
            self._stopping = False

    def submit(
        self,
        job: Callable[[], None],
        *,
        heavy: bool = False,
        drop_on_full: bool | None = None,
        dedupe_key: Any | None = None,
    ) -> bool:
        """Enqueue ``job``. Return True if queued or run; False if dropped.

        Queue full: heavy jobs (dump / torch.save — seconds of blocking I/O)
        are always dropped, never run inline on the inference thread; light
        sinks fall back to inline execution to preserve their effect.
        ``drop_on_full=True`` also drops (after-sample CPU detect must not
        block ``get_output`` / request finish).

        ``dedupe_key``: if another job with the same key is already queued or
        running, return False without enqueueing.
        """
        drop = bool(heavy if drop_on_full is None else drop_on_full)
        run_inline = False
        with self._lock:
            if self._stopping:
                logger.warning(
                    "[runtime_guard] action queue stopping; dropping job name=%s",
                    self._name,
                )
                return False
            if not self._started:
                self._start_unlocked()
            if not self._started:
                return False
            if dedupe_key is not None and dedupe_key in self._pending_keys:
                logger.info(
                    "[runtime_guard] action queue skip enqueue: duplicate key=%s name=%s "
                    "(same job already queued or running)",
                    dedupe_key,
                    self._name,
                )
                return False
            if dedupe_key is not None:
                self._pending_keys.add(dedupe_key)

            def _wrapped(
                _job: Callable[[], None] = job,
                _key: Any = dedupe_key,
            ) -> None:
                try:
                    _job()
                finally:
                    if _key is not None:
                        with self._lock:
                            self._pending_keys.discard(_key)

            try:
                self._queue.put_nowait(_wrapped)
                return True
            except queue.Full:
                if dedupe_key is not None:
                    self._pending_keys.discard(dedupe_key)
                if drop:
                    logger.warning(
                        "[runtime_guard] action queue full (maxsize=%d); dropping job (never inline)",
                        self._queue.maxsize,
                    )
                    return False
                logger.warning(
                    "[runtime_guard] action queue full (maxsize=%d); running light sink inline",
                    self._queue.maxsize,
                )
                run_inline = True
        if run_inline:
            try:
                job()
            except Exception:
                logger.exception("[runtime_guard] inline sink job failed")
            return True
        return False

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                item()
            except Exception:
                logger.exception("[runtime_guard] async sink job failed")
            finally:
                self._queue.task_done()
