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

"""Background worker for the last-PP TP merged config+dump due bus.

C2 hot path (reload on, detectors off) pays ``sync_due_bits_from_src`` every
step. Running that source broadcast (and rare due-lane broadcasts) on a
dedicated daemon
thread lets the inference thread overlap the collective with forward:

  wave head:  submit(local bits + dump jobs)  → return immediately
  forward:    BusWorker due-broadcast / optional bcasts
  end-of-wave: wait → apply config / stash deferred dump → D2H

Only this thread touches the ProcessGroup for the merged bus (queue depth 1).
ActionQueue stays separate (disk IO must not stall collectives).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

_STOP = object()


@dataclass
class MergedBusRequest:
    """One wave's merged-bus work for the background thread."""

    sync_group: Any
    config_due_local: bool
    dump_due_local: bool
    dump_jobs: list[dict[str, Any]]
    hot_reload_enabled: bool
    is_first_rank: bool
    # Leader-only: build config JSON once the bus says config_due.
    build_config_payload: Callable[[], tuple[Any, bool]] | None = None
    # Rank-local wave counter, carried in the due-broadcast payload so
    # receivers can assert same-wave alignment (fail fast on skips).
    wave_idx: int = -1


@dataclass
class MergedBusResult:
    """Collective outcomes; main thread applies config / extends deferred."""

    config_due: bool = False
    dump_due: bool = False
    config_payload: Any | None = None
    leader_changed: bool = False
    dump_jobs: list[dict[str, Any]] | None = None
    error: BaseException | None = None


@dataclass
class _Slot:
    """Single in-flight request/result pair (queue depth 1)."""

    request: MergedBusRequest
    done: threading.Event = field(default_factory=threading.Event)
    result: MergedBusResult | None = None


class DueBitsBusWorker:
    """Daemon thread: merged-bus due-broadcast (+ due bcasts).
    Main thread never calls collectives."""

    def __init__(self, *, name: str = "runtime-guard-bus") -> None:
        self._name = name
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopping = False
        self._lock = threading.Lock()
        self._inflight: _Slot | None = None

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        with self._lock:
            if self._started or self._stopping:
                return
            self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
            self._thread.start()
            self._started = True
            logger.info("[runtime_guard] bus worker started name=%s", self._name)

    def stop(self, *, timeout: float = 2.0) -> None:
        with self._lock:
            if not self._started:
                return
            self._stopping = True
            # Drop a pending request so the sentinel always fits.
            with suppress(queue.Empty):
                self._queue.get_nowait()
            with suppress(queue.Full):
                self._queue.put_nowait(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        with self._lock:
            self._thread = None
            self._started = False
            self._stopping = False
            self._inflight = None
            logger.info("[runtime_guard] bus worker stopped name=%s", self._name)

    def submit(self, request: MergedBusRequest) -> None:
        """Enqueue one merged-bus wave. Raises if a prior wave is still inflight."""
        with self._lock:
            if not self._started or self._stopping:
                raise RuntimeError("DueBitsBusWorker is not started")
            if self._inflight is not None and not self._inflight.done.is_set():
                raise RuntimeError("DueBitsBusWorker queue depth is 1; drain the previous wave before submit")
            slot = _Slot(request=request)
            self._inflight = slot
        self._queue.put(slot)

    def poll_ready(self) -> bool:
        """True when the inflight wave has a result (or nothing inflight)."""
        slot = self._inflight
        if slot is None:
            return True
        return slot.done.is_set()

    def wait_result(self, *, timeout: float | None = None) -> MergedBusResult | None:
        """Block until the inflight result is ready; clears inflight.

        Returns ``None`` when nothing was submitted.
        """
        with self._lock:
            slot = self._inflight
        if slot is None:
            return None
        if not slot.done.wait(timeout=timeout):
            raise TimeoutError("DueBitsBusWorker wait timed out")
        with self._lock:
            if self._inflight is slot:
                self._inflight = None
        assert slot.result is not None
        return slot.result

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, _Slot)
                item.result = self._run_merged_bus(item.request)
                item.done.set()
            except Exception as exc:
                if isinstance(item, _Slot):
                    item.result = MergedBusResult(error=exc)
                    item.done.set()
                else:
                    raise
            finally:
                self._queue.task_done()

    @staticmethod
    def _run_merged_bus(req: MergedBusRequest) -> MergedBusResult:
        from vllm_ascend.observability.runtime_config.dist import (
            broadcast_when_due,
            sync_due_bits_from_src,
        )

        config_due, dump_due = sync_due_bits_from_src(
            req.sync_group,
            [req.config_due_local, req.dump_due_local],
            wave_idx=req.wave_idx,
        )
        leader_changed = False
        config_payload: Any | None = None
        if req.hot_reload_enabled:

            def _build() -> Any:
                nonlocal leader_changed
                assert req.build_config_payload is not None
                payload, ch = req.build_config_payload()
                leader_changed = bool(ch)
                return payload

            config_payload = broadcast_when_due(
                req.sync_group,
                due=config_due,
                build_payload=_build if req.is_first_rank else None,
                src=0,
            )
        dump_out = broadcast_when_due(
            req.sync_group,
            due=dump_due,
            payload=req.dump_jobs if req.is_first_rank else None,
            src=0,
        )
        return MergedBusResult(
            config_due=bool(config_due),
            dump_due=bool(dump_due),
            config_payload=config_payload,
            leader_changed=leader_changed,
            dump_jobs=list(dump_out) if dump_due and dump_out else None,
        )
