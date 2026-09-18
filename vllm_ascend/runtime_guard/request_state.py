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

"""Per-request runtime-guard memory state (shared across IO / filter / wave).

Cross-module per-req fields live on :class:`RequestGuardState`. Detectors keep
their own private dicts for now; finish cleanup is:

1. :meth:`mark_finished` (from ``RuntimeGuardProcessor.mark_finished``)
2. :meth:`list_reapable` when ``_drain_probe`` (WaveTracker.pending) is clear,
   or past ``max_deferred_waves``
3. :meth:`clear` from ``RuntimeGuardProcessor._reap_finished_requests`` (also clears
   detectors)

Lifecycle (async-safe)::

    mark_finished(req)  # scheduler finished; do NOT pop yet
    record_sample_waves  # last sample stamp still allowed
    check_after_sample  # logits .item() / append / enqueue CPU detect
    _reap_finished_requests  # waits until cpu_jobs==0 (or max_deferred_waves)

While ``finished=True``, writers reuse the same state object (no second create).
After :meth:`clear`, a later create is treated as a **new** request with the
same id (id reuse).

Do **not** add parallel ``_xxx_by_req`` maps for new shared per-req fields —
extend :class:`RequestGuardState` instead.

Intentionally **not** stored here (survive request finish): dump totals and
on-disk anomaly report files.

Optional :meth:`register_on_clear` hooks let report helpers (e.g. I/O snapshot
wave cache) drop per-req scratch without Store importing those modules.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.runtime_guard.detector.manager import DetectorManager

logger = init_logger_ascend(__name__)

# If finished but sample_waves still non-empty this many real-steps later,
# force reap (stuck / dropped AsyncOutput). Prefer wave-empty as the signal.
DEFAULT_MAX_DEFERRED_WAVES = 8
# Post-reap late async appends: remember recently cleared ids so a zombie
# recreate in append_output_ids can be stamped finished=True for reap.
REAPED_RING_MAX = 1024


@dataclass
class RequestGuardState:
    """All shared runtime-guard memory for one ``req_id`` until :meth:`RequestGuardStore.clear`."""

    req_id: str
    output_token_ids: list[int] = field(default_factory=list)
    detection_stopped: bool = False
    # Same-wave append dedupe frontier; not sample-wave stamps.
    last_append_chunk: tuple[int, ...] | None = None
    # Scheduler finished; keep state until reap (last get_output / idle sweep).
    finished: bool = False
    finish_mark_wave: int | None = None
    # Cached prompt token ids captured from scheduler_output.scheduled_new_reqs
    # on the first prefill wave. Avoids reading v2 RequestState.all_token_ids
    # StagedWriteTensor host mirror before apply_staged_writes commits (Bug #9:
    # mirror initialized to 0, snapshot returned 19 zeros / empty after clear).
    prompt_token_ids: list[int] | None = None
    # In-flight after-sample CPU detect jobs. Reap waits until 0 so Store /
    # detector windows survive until the ActionQueue worker finishes.
    cpu_jobs: int = 0


class RequestGuardStore:
    """Process-wide ``req_id → RequestGuardState`` with deferred finish clear."""

    _instance: RequestGuardStore | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_req: dict[str, RequestGuardState] = {}
        # Soft deps (snapshot wave cache, …): called after state is popped.
        self._on_clear: list[Callable[[str], None]] = []
        self.max_deferred_waves = DEFAULT_MAX_DEFERRED_WAVES
        self._reaped_ring: deque[str] = deque()
        self._reaped_set: set[str] = set()
        # B1: replaces the dead store-side sample-wave FIFO — the processor
        # injects WaveTracker.pending so reap waits for async output drains.
        self._drain_probe: Callable[[str], bool] | None = None

    def _note_reaped_locked(self, rid: str) -> None:
        """Remember a cleared id for post-reap zombie detection (caller holds lock)."""
        if rid in self._reaped_set:
            return
        while len(self._reaped_ring) >= REAPED_RING_MAX:
            old = self._reaped_ring.popleft()
            self._reaped_set.discard(old)
        self._reaped_ring.append(rid)
        self._reaped_set.add(rid)

    def _discard_reaped_locked(self, rid: str) -> None:
        """Drop reaped tracking when a new live request starts (caller holds lock)."""
        self._reaped_set.discard(rid)

    @classmethod
    def get(cls) -> RequestGuardStore:
        # B11: double-checked lock — two threads racing the first get() must
        # not build two stores.
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def set_drain_probe(self, probe: Callable[[str], bool] | None) -> None:
        self._drain_probe = probe

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop singleton (unit tests only)."""
        cls._instance = None

    def register_on_clear(self, hook: Callable[[str], None]) -> None:
        """Register a per-req cleanup hook (idempotent by function identity)."""
        if hook not in self._on_clear:
            self._on_clear.append(hook)

    def get_state(self, req_id: str) -> RequestGuardState | None:
        if not req_id:
            return None
        with self._lock:
            return self._by_req.get(str(req_id))

    def get_or_create(self, req_id: str) -> RequestGuardState:
        """Return existing state, or create once for a live request.

        Never allocates a second object for an id that already exists (including
        ``finished=True`` deferred states).
        """
        rid = str(req_id)
        with self._lock:
            state = self._by_req.get(rid)
            if state is None:
                self._discard_reaped_locked(rid)
                state = RequestGuardState(req_id=rid)
                self._by_req[rid] = state
            return state

    def mark_finished(self, req_ids: Iterable[str] | None, *, wave: int) -> None:
        """Mark requests finished; keep state until :meth:`try_reap` / reap sweep.

        Safe to call before the last ``record_sample_waves`` /
        ``check_after_sample`` (current runner order).
        """
        if not req_ids:
            return
        w = int(wave)
        with self._lock:
            for raw in req_ids:
                if not raw:
                    continue
                rid = str(raw)
                state = self._by_req.get(rid)
                if state is None:
                    state = RequestGuardState(req_id=rid)
                    self._by_req[rid] = state
                state.finished = True
                if state.finish_mark_wave is None:
                    state.finish_mark_wave = w

    def add_cpu_jobs(self, req_ids: Iterable[str] | None) -> None:
        """Pin finished reqs until the after-sample CPU job runs (or is dropped)."""
        if not req_ids:
            return
        with self._lock:
            for raw in req_ids:
                if not raw:
                    continue
                rid = str(raw)
                state = self._by_req.get(rid)
                if state is None:
                    self._discard_reaped_locked(rid)
                    state = RequestGuardState(req_id=rid)
                    self._by_req[rid] = state
                state.cpu_jobs += 1

    def finish_cpu_jobs(self, req_ids: Iterable[str] | None) -> None:
        if not req_ids:
            return
        with self._lock:
            for raw in req_ids:
                if not raw:
                    continue
                state = self._by_req.get(str(raw))
                if state is not None and state.cpu_jobs > 0:
                    state.cpu_jobs -= 1

    def kv_dump_allowed(self, req_id: str) -> bool:
        """False when the request has finished or been reaped (KV may be reused).

        Unknown ids (never in Store) return True so unit tests / manual dumps
        without Store state still proceed.
        """
        if not req_id:
            return False
        rid = str(req_id)
        with self._lock:
            state = self._by_req.get(rid)
            if state is not None:
                return not state.finished
            return rid not in self._reaped_set

    def _ready_to_reap_locked(self, state: RequestGuardState, *, current_wave: int) -> bool:
        # Defer-cap first: a finished req must not linger past it even if the
        # drain probe is stuck (dropped AsyncOutput / dead consumer).
        mark = state.finish_mark_wave
        if state.cpu_jobs > 0:
            if mark is None or int(current_wave) - int(mark) < int(self.max_deferred_waves):
                return False
        if mark is not None and int(current_wave) - int(mark) >= int(self.max_deferred_waves):
            return True
        probe = self._drain_probe
        if probe is not None:
            try:
                return not probe(state.req_id)
            except Exception:
                logger.debug("[runtime_guard reap] drain probe failed for %s", state.req_id, exc_info=True)
                return True
        return True

    def list_reapable(self, *, current_wave: int) -> list[str]:
        """Finished reqs whose drain probe is clear (or past defer cap)."""
        w = int(current_wave)
        with self._lock:
            out: list[str] = []
            for rid, state in self._by_req.items():
                if state.finished and self._ready_to_reap_locked(state, current_wave=w):
                    out.append(rid)
            return out

    def clear(
        self,
        req_id: str,
        *,
        detectors: DetectorManager | None = None,
    ) -> RequestGuardState | None:
        """Pop shared state and clear detector private per-req maps."""
        if not req_id:
            return None
        rid = str(req_id)
        if detectors is not None:
            detectors.clear_finished(rid)
        with self._lock:
            state = self._by_req.pop(rid, None)
            if state is not None:
                self._note_reaped_locked(rid)
            hooks = list(self._on_clear)
        for hook in hooks:
            try:
                hook(rid)
            except Exception:
                logger.exception(
                    "[runtime_guard clear] on_clear hook failed req_id=%s hook=%r",
                    rid,
                    hook,
                )
        return state

    def clear_many(
        self,
        req_ids: Iterable[str],
        *,
        detectors: DetectorManager | None = None,
    ) -> None:
        for req_id in req_ids:
            if req_id:
                self.clear(str(req_id), detectors=detectors)

    # ---- IO helpers -------------------------------------------------------

    def append_output_ids(self, req_id: str, token_ids: list[int]) -> None:
        if not req_id or not token_ids:
            return
        rid = str(req_id)
        chunk = tuple(token_ids)
        with self._lock:
            state = self._by_req.get(rid)
            if state is None:
                # New live request, or post-reap late async append (zombie).
                state = RequestGuardState(req_id=rid)
                if rid in self._reaped_set:
                    # S14/R1: clear() already ran; stamp finished so list_reapable
                    # can reap this zombie instead of leaving finished=False forever.
                    state.finished = True
                    self._reaped_set.discard(rid)
                self._by_req[rid] = state
            if state.last_append_chunk == chunk:
                return
            state.output_token_ids.extend(token_ids)
            state.last_append_chunk = chunk

    def clear_wave_append_frontier(self) -> None:
        """Reset same-wave dedupe so identical chunks across steps are kept."""
        with self._lock:
            for state in self._by_req.values():
                state.last_append_chunk = None

    def new_output_ids_since(self, req_id: str, consumed: int) -> tuple[int, list[int]]:
        """Return ``(total_len, ids[consumed:])``, copying only the new tail.

        Detectors fold the cumulative output stream with a per-req cursor; a
        full ``list(output_token_ids)`` copy every step is O(total) per step.
        This materializes only ``ids[consumed:]`` (O(new)) under ``self._lock``
        so the slice never races ``append_output_ids``.
        """
        if not req_id:
            return 0, []
        rid = str(req_id)
        with self._lock:
            state = self._by_req.get(rid)
            if state is None:
                return 0, []
            ids = state.output_token_ids
            total = len(ids)
            if consumed >= total:
                return total, []
            start = max(consumed, 0)
            return total, ids[start:]

    # ---- prompt token ids cache (Bug #9 fix) -----------------------------

    def set_prompt_token_ids(self, req_id: str, token_ids: Sequence[int] | None) -> None:
        """Cache prompt token ids captured from scheduler_output on first wave.

        Idempotent: first non-None value wins. Later calls with a different list
        are ignored so prompt ids stay stable for the request's lifetime.
        """
        if not req_id or token_ids is None:
            return
        rid = str(req_id)
        ids = [int(x) for x in token_ids]
        if not ids:
            return
        with self._lock:
            state = self._by_req.get(rid)
            if state is None:
                self._discard_reaped_locked(rid)
                state = RequestGuardState(req_id=rid)
                self._by_req[rid] = state
            if state.prompt_token_ids is None:
                state.prompt_token_ids = ids

    # ---- sample waves -----------------------------------------------------
    # B1: the store-side sample-wave FIFO was dead code (never written by the
    # processor, which stamps WaveTracker instead). Drain gating now flows
    # through the injected ``_drain_probe`` (WaveTracker.pending).

    # ---- detection_stopped (report.max_per_req write-full) ----------------

    def stopped_req_ids(self) -> set[str]:
        with self._lock:
            return {rid for rid, st in self._by_req.items() if st.detection_stopped}

    def mark_detection_stopped(self, req_id: str | None) -> None:
        """Stop all detectors for ``req_id`` (report ``max_per_req`` reached).

        No-op when the id is already gone (reaped / never seen). Report commit
        can race finish→reap on ActionQueue; allocating here would resurrect an
        unfinished orphan and, on req_id reuse, stop-detect the new request.
        """
        if not req_id:
            return
        rid = str(req_id)
        with self._lock:
            state = self._by_req.get(rid)
            if state is None:
                return
            state.detection_stopped = True
