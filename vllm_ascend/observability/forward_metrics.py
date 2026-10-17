#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Async device forward time metrics collector for Model Runner V1.

Records a pair of NPU timing events around each model forward call and
aggregates device time by ``(rank, role, phase, batch_size)`` over periodic
windows. The whole path is non-blocking:

* ``record()`` only enqueues work on the current NPU stream;
* ``elapsed_time()`` is only allowed after ``query()`` returned True;
* ``synchronize()``/``wait_event()`` are never called by this module.

Events are recycled through a bounded pool; pending samples are kept in a
FIFO queue whose total in-flight count (pending + active) is capped at
``window_size``. Samples settle asynchronously, possibly after their window
has closed, and are always attributed to the window they were admitted into.

Counting unit: a target sample brackets one model forward; a draft sample
brackets one draft propose invocation, i.e. the whole merged draft chain
(k model forwards when ``num_speculative_tokens > 1``) plus any sampling
inside the propose entry. This is the only boundary that stays consistent
between eager and ACL graph mode, where a single replay covers the whole
chain, so draft and target averages must not be compared 1:1.
"""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal

from vllm.logger import logger

ModelRole = Literal["target", "draft"]
ForwardPhase = Literal["prefill", "decode", "mixed", "verify"]

DEFAULT_WINDOW_SIZE = 1000
_EVENTS_PER_SAMPLE = 2
_WARN_EVERY = 100
# Bounded backlog of emitted WindowSummary objects kept for phase-2 consumers
# polling flush_window(); logs already carry the same information.
_SUMMARY_BACKLOG = 256


def _default_event_factory() -> Any:
    """Create a timing NPU event.

    Imported lazily so this module stays importable on CPU-only test runners
    (tests inject their own fake event factory).
    """
    import torch

    return torch.npu.Event(enable_timing=True)


def npu_stream_id() -> int:
    """Compare native handles, not transient Python stream wrapper identities."""
    import torch

    return torch.npu.current_stream().npu_stream


@dataclass(frozen=True)
class MetricKey:
    rank: int
    role: ModelRole
    phase: ForwardPhase
    batch_size: int


@dataclass
class ForwardTimeHandle:
    key: MetricKey
    window_id: int
    start_event: Any
    stream_id: Any = None


@dataclass
class PendingSample:
    key: MetricKey
    window_id: int
    start_event: Any
    end_event: Any


@dataclass
class RunningStats:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = math.inf
    max_ms: float = 0.0

    def update(self, elapsed_ms: float) -> None:
        self.count += 1
        self.total_ms += elapsed_ms
        self.min_ms = min(self.min_ms, elapsed_ms)
        self.max_ms = max(self.max_ms, elapsed_ms)


@dataclass(frozen=True)
class WindowSummary:
    window_id: int
    key: MetricKey
    count: int
    avg_ms: float
    min_ms: float
    max_ms: float
    dropped: int
    errors: int


@dataclass
class _WindowState:
    window_id: int
    accepted: int = 0
    in_flight: int = 0
    closed: bool = False
    emitted: bool = False
    stats: dict[MetricKey, RunningStats] = field(default_factory=dict)
    dropped: dict[MetricKey, int] = field(default_factory=dict)
    errors: dict[MetricKey, int] = field(default_factory=dict)


class ForwardTimeCollector:
    """Bounded, non-blocking collector; one per single-threaded worker."""

    def __init__(
        self,
        rank: int,
        window_size: int = DEFAULT_WINDOW_SIZE,
        target_batch_sizes: frozenset[int] | set[int] = frozenset(),
        event_factory: Callable[[], Any] | None = None,
        emit_log: Callable[[str], None] | None = None,
        enabled: bool = True,
        stream_getter: Callable[[], Any] | None = None,
        labels: dict[str, int] | None = None,
        collect_draft: bool = True,
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be a positive integer, got {window_size}")
        self._rank = rank
        self._window_size = window_size
        self._target_bs = frozenset(target_batch_sizes)
        self._event_factory = event_factory or _default_event_factory
        self._emit_log = emit_log or (lambda msg: logger.info(msg))
        self._enabled = enabled
        self._stream_getter = stream_getter
        self._labels = dict(labels or {})
        self.collect_draft = collect_draft

        self._pool: list[Any] = []
        self._pool_cap = _EVENTS_PER_SAMPLE * window_size
        self._pending: deque[PendingSample] = deque()
        self._active_handles: dict[int, ForwardTimeHandle] = {}
        self._windows: OrderedDict[int, _WindowState] = OrderedDict()
        self._next_window_id = 1
        self._finalized = False
        self._summaries: deque[WindowSummary] = deque(maxlen=_SUMMARY_BACKLOG)
        self._warn_counts: dict[tuple[str, str], int] = {}

        if self._enabled:
            self._open_window()
            logger.info(
                "ForwardTimeCollector enabled: rank=%s window_size=%s target_batch_sizes=%s",
                rank,
                window_size,
                sorted(target_batch_sizes) if target_batch_sizes else "all",
            )

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def start(self, role: ModelRole, phase: ForwardPhase, batch_size: int) -> ForwardTimeHandle | None:
        if not self._enabled or self._finalized:
            return None
        self.drain_ready()
        if self._target_bs and batch_size not in self._target_bs:
            return None

        key = MetricKey(rank=self._rank, role=role, phase=phase, batch_size=batch_size)
        window = self._admission_window()
        if len(self._pending) + len(self._active_handles) >= self._window_size:
            window.dropped[key] = window.dropped.get(key, 0) + 1
            return None

        try:
            stream_id = self._stream_getter() if self._stream_getter else None
            start_event = self._acquire_event()
            start_event.record()
        except Exception as exc:
            window.errors[key] = window.errors.get(key, 0) + 1
            self._warn_limited("start", exc)
            return None

        window.accepted += 1
        window.in_flight += 1
        if window.accepted >= self._window_size:
            window.closed = True
        handle = ForwardTimeHandle(key=key, window_id=window.window_id, start_event=start_event, stream_id=stream_id)
        self._active_handles[id(handle)] = handle
        return handle

    def finish(self, handle: ForwardTimeHandle | None) -> None:
        if handle is None or self._active_handles.get(id(handle)) is not handle:
            return
        if self._stream_getter is not None:
            try:
                if self._stream_getter() != handle.stream_id:
                    raise RuntimeError("execution stream changed across the timed boundary")
            except Exception as exc:
                self._warn_limited("stream", exc)
                self.abort(handle)
                return
        del self._active_handles[id(handle)]
        window = self._windows.get(handle.window_id)
        if window is None or window.emitted:
            # Defensive: the bound window is already gone; drop the sample.
            return
        try:
            end_event = self._acquire_event()
            end_event.record()
        except Exception as exc:
            window.in_flight -= 1
            window.errors[handle.key] = window.errors.get(handle.key, 0) + 1
            self._warn_limited("finish", exc)
            # The accounting change may have made a closed window idle; emit
            # it now instead of waiting for the next settled sample.
            self._emit_idle_windows()
            return
        self._pending.append(
            PendingSample(
                key=handle.key,
                window_id=handle.window_id,
                start_event=handle.start_event,
                end_event=end_event,
            )
        )

    def abort(self, handle: ForwardTimeHandle | None) -> None:
        """Forward raised: release the active slot, count as dropped, keep the exception."""
        if handle is None or self._active_handles.get(id(handle)) is not handle:
            return
        del self._active_handles[id(handle)]
        window = self._windows.get(handle.window_id)
        if window is not None and not window.emitted:
            window.in_flight -= 1
            window.dropped[handle.key] = window.dropped.get(handle.key, 0) + 1
            # Emitting here keeps a window whose last in-flight sample aborted
            # from lingering until the next successful settlement (or teardown).
            self._emit_idle_windows()
        # The recorded start_event is not pooled: it may not be complete yet.

    def drain_ready(self) -> None:
        """Settle completed pending samples; stop at the first incomplete one.

        Valid because all events are recorded on the same NPU stream and thus
        complete in submission order.
        """
        settled = False
        while self._pending:
            head = self._pending[0]
            try:
                done = bool(head.end_event.query())
            except Exception as exc:
                self._pending.popleft()
                self._fail_sample(head, exc)
                settled = True
                continue
            if not done:
                break
            self._pending.popleft()
            self._settle(head)
            settled = True
        if settled:
            self._emit_idle_windows()

    def flush_window(self, final: bool = False) -> list[WindowSummary]:
        summaries: list[WindowSummary] = []
        if not self._enabled:
            return summaries
        self.drain_ready()
        if final:
            self._finalized = True
            # Teardown: unfinished samples are dropped into their bound windows.
            while self._pending:
                sample = self._pending.popleft()
                window = self._windows.get(sample.window_id)
                if window is not None and not window.emitted:
                    window.in_flight -= 1
                    window.dropped[sample.key] = window.dropped.get(sample.key, 0) + 1
                # Events are NOT recycled here: they may still be incomplete.
            for handle in list(self._active_handles.values()):
                self.abort(handle)
            for window in list(self._windows.values()):
                if not window.emitted:
                    self._emit_window(window)
            self._windows.clear()
        summaries.extend(self._summaries)
        self._summaries.clear()
        return summaries

    # ------------------------------------------------------------------ #
    # Read-only state for tests and operational metrics
    # ------------------------------------------------------------------ #

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def active_count(self) -> int:
        return len(self._active_handles)

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    @property
    def live_window_count(self) -> int:
        return len(self._windows)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _open_window(self) -> _WindowState:
        window = _WindowState(window_id=self._next_window_id)
        self._next_window_id += 1
        self._windows[window.window_id] = window
        return window

    def _admission_window(self) -> _WindowState:
        """Window a new sample would be admitted into; rolls if the current one is full."""
        if not self._windows:
            return self._open_window()
        current = next(reversed(self._windows.values()))
        if current.accepted >= self._window_size:
            current.closed = True
            return self._open_window()
        return current

    def _acquire_event(self) -> Any:
        if self._pool:
            return self._pool.pop()
        return self._event_factory()

    def _recycle(self, *events: Any) -> None:
        for event in events:
            if len(self._pool) < self._pool_cap:
                self._pool.append(event)

    def _settle(self, sample: PendingSample) -> None:
        window = self._windows.get(sample.window_id)
        if window is None or window.emitted:
            # Defensive: window already emitted; just recycle the events.
            self._recycle(sample.start_event, sample.end_event)
            return
        window.in_flight -= 1
        try:
            elapsed_ms = sample.start_event.elapsed_time(sample.end_event)
        except Exception as exc:
            window.errors[sample.key] = window.errors.get(sample.key, 0) + 1
            self._warn_limited("elapsed_time", exc)
            self._recycle(sample.start_event, sample.end_event)
            return
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            window.errors[sample.key] = window.errors.get(sample.key, 0) + 1
        else:
            window.stats.setdefault(sample.key, RunningStats()).update(elapsed_ms)
        self._recycle(sample.start_event, sample.end_event)

    def _fail_sample(self, sample: PendingSample, exc: Exception) -> None:
        window = self._windows.get(sample.window_id)
        if window is not None and not window.emitted:
            window.in_flight -= 1
            window.errors[sample.key] = window.errors.get(sample.key, 0) + 1
        self._warn_limited("query", exc)
        # Events may be in an unknown state; do not recycle them.

    def _emit_idle_windows(self) -> None:
        # Emit strictly in window_id order: stop at the first window that is
        # not emittable so a later idle window can never be logged before an
        # earlier one.
        while self._windows:
            window = next(iter(self._windows.values()))
            if window.emitted or not window.closed or window.in_flight != 0:
                break
            self._emit_window(window)
            del self._windows[window.window_id]

    def _emit_window(self, window: _WindowState) -> None:
        window.emitted = True
        ordered_keys = sorted(
            (set(window.stats) | set(window.dropped) | set(window.errors)),
            key=lambda key: (-window.stats.get(key, RunningStats()).count, key.role, key.phase, key.batch_size),
        )
        for key in ordered_keys:
            stats = window.stats.get(key, RunningStats())
            try:
                self._emit_log(
                    f"Forward timing: window={window.window_id} rank={key.rank} "
                    + "".join(f"{name}={value} " for name, value in sorted(self._labels.items()))
                    + f"role={key.role} "
                    f"phase={key.phase} bs={key.batch_size} count={stats.count} "
                    f"avg_ms={stats.total_ms / stats.count if stats.count else 0.0:.3f} "
                    f"min_ms={stats.min_ms if stats.count else 0.0:.3f} "
                    f"max_ms={stats.max_ms:.3f} dropped={window.dropped.get(key, 0)} "
                    f"errors={window.errors.get(key, 0)}"
                )
            except Exception as exc:
                # Fail-open emission: one bad callback or key must not abort
                # the remaining log lines or the summaries below.
                self._warn_limited("emit_log", exc)
        for key in set(window.stats) | set(window.dropped) | set(window.errors):
            stats = window.stats.get(key)
            count = stats.count if stats is not None else 0
            self._summaries.append(
                WindowSummary(
                    window_id=window.window_id,
                    key=key,
                    count=count,
                    avg_ms=stats.total_ms / count if stats is not None and count > 0 else 0.0,
                    min_ms=stats.min_ms if stats is not None and count > 0 else 0.0,
                    max_ms=stats.max_ms if stats is not None and count > 0 else 0.0,
                    dropped=window.dropped.get(key, 0),
                    errors=window.errors.get(key, 0),
                )
            )

    def _warn_limited(self, op: str, exc: BaseException) -> None:
        counter_key = (type(exc).__name__, op)
        count = self._warn_counts.get(counter_key, 0) + 1
        self._warn_counts[counter_key] = count
        if count == 1 or count % _WARN_EVERY == 0:
            # Logging failure must not escape the metrics failure handler.
            with suppress(Exception):
                logger.warning(
                    "ForwardTimeCollector rank=%s: %s during %s (occurrence %d): %s",
                    self._rank,
                    type(exc).__name__,
                    op,
                    count,
                    exc,
                )


def run_timed_forward(
    collector: ForwardTimeCollector,
    role: ModelRole,
    phase_fn: Callable[[], ForwardPhase],
    batch_size: int,
    run_forward: Callable[[], Any],
) -> Any:
    """Fail-open timed execution of one business forward.

    Whatever the collector does — classify, start, abort or finish — the model
    forward itself must always run, a successful forward must never be turned
    into a failed one, and a business exception must never be masked:

    * ``phase_fn`` or ``collector.start`` raising → the forward runs
      uninstrumented (a sample is skipped, not the forward);
    * ``run_forward`` raising → ``collector.abort`` runs inside its own guard
      so cleanup failures cannot replace the original exception;
    * ``collector.finish`` raising after a successful forward → swallowed.

    Both integration paths (model runner target/verify forwards and the draft
    wrapper) go through this single guard, so the fail-open contract lives in
    exactly one place.
    """
    try:
        phase = phase_fn()
    except Exception as exc:
        collector._warn_limited("phase", exc)
        return run_forward()
    try:
        handle = collector.start(role, phase, batch_size)
    except Exception as exc:
        collector._warn_limited("start", exc)
        return run_forward()
    try:
        result = run_forward()
    except BaseException:
        try:
            collector.abort(handle)
        except Exception as exc:
            collector._warn_limited("abort", exc)
        raise
    try:
        collector.finish(handle)
    except Exception as exc:
        collector._warn_limited("finish", exc)
    return result
