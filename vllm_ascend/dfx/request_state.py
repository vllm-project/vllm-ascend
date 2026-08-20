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

"""Per-request DFX memory state (shared across IO / filter / wave / dump_finish).

Cross-module per-req fields live on :class:`RequestDfxState`. Detectors keep
their own private dicts for now; finish cleanup is:

1. :meth:`mark_finished` (from ``DfxProcessor.mark_finished``)
2. :meth:`list_reapable` when ``sample_waves`` is empty (or deferred-wave cap)
3. :meth:`clear` from ``DfxProcessor._reap_finished_requests`` (also clears
   detectors)

Lifecycle (async-safe)::

    mark_finished(req)  # scheduler finished; do NOT pop yet
    record_sample_waves  # last sample stamp still allowed
    check_after_sample  # take stamp / detect / append still allowed
    _reap_finished_requests  # sample_waves empty (or max_deferred_waves)

While ``finished=True``, writers reuse the same state object (no second create).
After :meth:`clear`, a later create is treated as a **new** request with the
same id (id reuse).

Do **not** add parallel ``_xxx_by_req`` maps for new shared per-req fields —
extend :class:`RequestDfxState` instead.

Intentionally **not** stored here (survive request finish):
``Dumper._msprobe_dumped_req_ids``, dump totals, on-disk anomaly / dump_finish
files, and open dump arm batches (``Dumper._open_dump_*``).

Optional :meth:`register_on_clear` hooks let report helpers (e.g. I/O snapshot
wave cache) drop per-req scratch without Store importing those modules.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm_ascend.dfx.dfx_types import DumpFinishMeta
from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.dfx.detector.manager import DetectorManager

logger = init_logger_ascend(__name__)

# If finished but sample_waves still non-empty this many real-steps later,
# force reap (stuck / dropped AsyncOutput). Prefer wave-empty as the signal.
DEFAULT_MAX_DEFERRED_WAVES = 8


@dataclass
class RequestDfxState:
    """All shared DFX memory for one ``req_id`` until :meth:`RequestDfxStore.clear`."""

    req_id: str
    output_token_ids: list[int] = field(default_factory=list)
    filter_allowed: bool | None = None
    sample_waves: deque[int] = field(default_factory=deque)
    stopped_after_alert: bool = False
    dump_finish: DumpFinishMeta | None = None
    # Same-wave append dedupe frontier; not sample-wave stamps.
    last_append_chunk: tuple[int, ...] | None = None
    # Scheduler finished; keep state until reap (last get_output / idle sweep).
    finished: bool = False
    finish_mark_wave: int | None = None


class RequestDfxStore:
    """Process-wide ``req_id → RequestDfxState`` with deferred finish clear."""

    _instance: RequestDfxStore | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_req: dict[str, RequestDfxState] = {}
        # Soft deps (snapshot wave cache, …): called after state is popped.
        self._on_clear: list[Callable[[str], None]] = []
        self.max_deferred_waves = DEFAULT_MAX_DEFERRED_WAVES

    @classmethod
    def get(cls) -> RequestDfxStore:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop singleton (unit tests only)."""
        cls._instance = None

    def register_on_clear(self, hook: Callable[[str], None]) -> None:
        """Register a per-req cleanup hook (idempotent by function identity)."""
        if hook not in self._on_clear:
            self._on_clear.append(hook)

    def get_state(self, req_id: str) -> RequestDfxState | None:
        if not req_id:
            return None
        with self._lock:
            return self._by_req.get(str(req_id))

    def get_or_create(self, req_id: str) -> RequestDfxState:
        """Return existing state, or create once for a live request.

        Never allocates a second object for an id that already exists (including
        ``finished=True`` deferred states).
        """
        rid = str(req_id)
        with self._lock:
            state = self._by_req.get(rid)
            if state is None:
                state = RequestDfxState(req_id=rid)
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
                    state = RequestDfxState(req_id=rid)
                    self._by_req[rid] = state
                state.finished = True
                if state.finish_mark_wave is None:
                    state.finish_mark_wave = w

    def is_finished(self, req_id: str) -> bool:
        state = self.get_state(req_id)
        return bool(state is not None and state.finished)

    def ready_to_reap(self, req_id: str, *, current_wave: int) -> bool:
        """True when finished and sample-wave FIFO is drained (or overdue)."""
        with self._lock:
            state = self._by_req.get(str(req_id)) if req_id else None
            if state is None or not state.finished:
                return False
            return self._ready_to_reap_locked(state, current_wave=int(current_wave))

    def _ready_to_reap_locked(self, state: RequestDfxState, *, current_wave: int) -> bool:
        if not state.sample_waves:
            return True
        mark = state.finish_mark_wave
        if mark is None:
            return False
        return int(current_wave) - int(mark) >= int(self.max_deferred_waves)

    def list_reapable(self, *, current_wave: int) -> list[str]:
        """Finished reqs whose sample-wave queue is empty (or past defer cap)."""
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
    ) -> RequestDfxState | None:
        """Pop shared state and clear detector private per-req maps.

        Prefer :meth:`take_dump_finish` / finish sidecars before clear.
        """
        if not req_id:
            return None
        rid = str(req_id)
        if detectors is not None:
            detectors.clear_finished(rid)
        with self._lock:
            state = self._by_req.pop(rid, None)
            hooks = list(self._on_clear)
            force_waves = int(len(state.sample_waves)) if state is not None else 0
            max_defer = int(self.max_deferred_waves)
        if state is not None and force_waves:
            # Small leftover: likely dropped AsyncOutput / stuck get_output on
            # the consuming rank → WARNING. Large backlog was the old async
            # non-TP0 pattern (stamp every step, never take); stamping is now
            # rank-gated, so this should be rare — keep DEBUG for noise.
            msg = (
                "[DFX reap] clearing req_id=%s with %d leftover sample_waves "
                "(async stamp not consumed; deferred-wave cap or idle sweep)"
            )
            if force_waves <= max_defer:
                logger.warning(msg, rid, force_waves)
            else:
                logger.debug(msg, rid, force_waves)
        for hook in hooks:
            try:
                hook(rid)
            except Exception:
                logger.exception(
                    "[DFX clear] on_clear hook failed req_id=%s hook=%r",
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
                # New live request, or first DFX touch. Deferred clear avoids the
                # common "clear then late async append" double-create; a true
                # post-reap append is rare (force-reap / dropped output).
                state = RequestDfxState(req_id=rid)
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

    def cumulative_output_ids(self, req_id: str) -> list[int]:
        state = self.get_state(req_id)
        return list(state.output_token_ids) if state is not None else []

    def cumulative_output_count(self, req_id: str) -> int:
        state = self.get_state(req_id)
        return len(state.output_token_ids) if state is not None else 0

    # ---- filter -----------------------------------------------------------

    def get_filter_allowed(self, req_id: str) -> bool | None:
        state = self.get_state(req_id)
        return None if state is None else state.filter_allowed

    def set_filter_allowed(self, req_id: str, allowed: bool) -> None:
        if not req_id:
            return
        self.get_or_create(req_id).filter_allowed = bool(allowed)

    def clear_filter_allowed(self, req_id: str) -> None:
        state = self.get_state(req_id)
        if state is not None:
            state.filter_allowed = None

    def clear_all_filter_allowed(self) -> None:
        with self._lock:
            for state in self._by_req.values():
                state.filter_allowed = None

    # ---- sample waves -----------------------------------------------------

    def record_sample_waves(self, req_ids: Iterable[str] | None, wave: int) -> None:
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
                    state = RequestDfxState(req_id=rid)
                    self._by_req[rid] = state
                # Finished reqs still accept the last stamp (runner marks finish
                # before record_sample_waves).
                state.sample_waves.append(w)

    def take_sample_wave(self, req_id: str) -> int | None:
        if not req_id:
            return None
        with self._lock:
            state = self._by_req.get(str(req_id))
            if state is None or not state.sample_waves:
                return None
            return int(state.sample_waves.popleft())

    def clear_sample_waves(self, req_id: str) -> None:
        state = self.get_state(req_id)
        if state is not None:
            state.sample_waves.clear()

    def sample_wave_pending(self, req_id: str) -> int:
        state = self.get_state(req_id)
        return len(state.sample_waves) if state is not None else 0

    # ---- stop_after_alert -------------------------------------------------

    def mark_stopped_after_alert(self, req_id: str) -> None:
        if not req_id:
            return
        self.get_or_create(req_id).stopped_after_alert = True

    def stopped_req_ids(self) -> set[str]:
        with self._lock:
            return {rid for rid, st in self._by_req.items() if st.stopped_after_alert}

    # ---- dump_finish meta -------------------------------------------------

    def set_dump_finish(self, req_id: str, meta: DumpFinishMeta) -> None:
        if not req_id:
            return
        self.get_or_create(req_id).dump_finish = meta

    def take_dump_finish(self, req_id: str) -> DumpFinishMeta | None:
        """Pop committed dump_finish meta without deleting the whole state."""
        if not req_id:
            return None
        with self._lock:
            state = self._by_req.get(str(req_id))
            if state is None or state.dump_finish is None:
                return None
            meta = state.dump_finish
            state.dump_finish = None
            return meta

    def dump_arm_wave_for_req(self, req_id: str) -> int | None:
        state = self.get_state(req_id)
        if state is None or state.dump_finish is None:
            return None
        arm = state.dump_finish.dump_arm_wave
        return int(arm) if isinstance(arm, int) else None

    def has_dump_finish_meta(self) -> bool:
        with self._lock:
            return any(st.dump_finish is not None for st in self._by_req.values())

    def __contains__(self, req_id: object) -> bool:
        if not isinstance(req_id, str) or not req_id:
            return False
        with self._lock:
            return req_id in self._by_req

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_req)
