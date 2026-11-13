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

"""Process-local runtime_guard state: incidents, waves, dump quota, per-req store."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig
    from vllm_ascend.observability.runtime_guard.detector.manager import DetectorManager


logger = init_logger_ascend(__name__)

# ---- incident types ----

# Align with msprobe response_anomaly ILLDetector ill_type codes.
ILL_TYPE_NONE = 0
ILL_TYPE_REPEAT = 3
ILL_TYPE_NAN = 4

ILL_TYPE_NAME: dict[int, str] = {
    ILL_TYPE_NONE: "none",
    ILL_TYPE_REPEAT: "repetition",
    ILL_TYPE_NAN: "nan",
}


@dataclass(slots=True)
class Incident:
    """One runtime finding handed to the action executor."""

    incident_type: str
    req_id: str
    is_ill: bool = True
    ill_type: int = ILL_TYPE_NONE
    req_idx: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    consume_quota: bool = True
    block_ids: list[int] = field(default_factory=list)
    wave: int | None = None
    log_context: dict[str, Any] = field(default_factory=dict)

    @property
    def ill_type_name(self) -> str:
        return ILL_TYPE_NAME.get(self.ill_type, f"unknown({self.ill_type})")

    def to_report_detail(self) -> dict[str, Any]:
        out = dict(self.detail)
        if self.ill_type != ILL_TYPE_NONE:
            out.setdefault("ill_type", self.ill_type)
            out.setdefault("ill_type_name", self.ill_type_name)
        out.setdefault("is_ill", self.is_ill)
        # ``block_ids`` is emitted solely by RuntimeGuardProcessor
        # ``_enrich_detail_with_block_meta`` (always attached). Emitting it
        # here would duplicate / race the processor enrichment path.
        return out


MANUAL_TRIGGER_REQ_ID = "__manual_trigger__"
MANUAL_TRIGGER_TYPE = "manual_trigger"


def iter_local_request_rows(
    runner: Any,
    scheduler_output: Any | None = None,
) -> list[tuple[str, int]]:
    """``(req_id, req_idx)`` for local live requests (v2 req_states / batch).

    Prefer ``input_batch.req_ids``; before prepare_inputs fall back to
    ``execute_model_state.input_batch``, ``req_states``, and
    ``scheduler_output.num_scheduled_tokens`` so manual_dump can arm on the
    first real prefill wave.
    """
    input_batch = getattr(runner, "input_batch", None)
    req_ids = getattr(input_batch, "req_ids", None) if input_batch is not None else None
    if req_ids:
        rows = [(str(req_id), idx) for idx, req_id in enumerate(req_ids) if req_id]
        if rows:
            return rows

    state = getattr(runner, "execute_model_state", None)
    state_batch = getattr(state, "input_batch", None) if state is not None else None
    state_ids = getattr(state_batch, "req_ids", None) if state_batch is not None else None
    if state_ids:
        rows = [(str(req_id), idx) for idx, req_id in enumerate(state_ids) if req_id]
        if rows:
            return rows

    requests = getattr(runner, "requests", None)
    if isinstance(requests, dict) and requests:
        return [(str(req_id), -1) for req_id in requests if req_id]

    req_states = getattr(runner, "req_states", None)
    id_map = getattr(req_states, "req_id_to_index", None) if req_states is not None else None
    if isinstance(id_map, dict) and id_map:
        return sorted(
            ((str(rid), int(idx)) for rid, idx in id_map.items() if rid),
            key=lambda item: item[1],
        )

    if scheduler_output is not None:
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if isinstance(num_scheduled, dict) and num_scheduled:
            return [(str(req_id), -1) for req_id, n_tok in num_scheduled.items() if req_id and int(n_tok or 0) > 0]
    return []


@dataclass(slots=True)
class TriggerEvent:
    """One control-plane trigger consumed from runtime_config."""

    trigger_type: str
    req_id: str
    detail: dict[str, Any] = field(default_factory=dict)
    consume_quota: bool = False

    def to_report_detail(self) -> dict[str, Any]:
        out = dict(self.detail)
        out.setdefault("source", self.trigger_type)
        return out


# ---- wave tracker ----


class WaveTracker:
    """Process-local wave counter + per-req sample stamps.

    ``record`` runs on the inference thread; async ``take`` / ``pending`` may
    run on the output copy thread — all stamp mutations share ``_lock``.

    Stamps are a **FIFO per req** (W1-3): under async scheduling, step N+1 may
    ``record`` before step N's ``get_output`` ``take``. A single overwrite slot
    made the later take miss (``missing sample-wave stamp``) and polluted
    ``arm_wave``. Ordered ``deque`` FIFOs keep stamp/take paired.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wave = 0
        self._sample_waves: dict[str, deque[int]] = {}

    def advance(self, *, allow_manual_dump: bool = True) -> None:
        if not allow_manual_dump:
            return
        with self._lock:
            self._wave += 1

    def current_wave(self) -> int:
        with self._lock:
            return self._wave

    def record_sample_waves(self, req_ids: list[str] | None) -> None:
        if not req_ids:
            return
        with self._lock:
            wave = self._wave
            for req_id in req_ids:
                if not req_id:
                    continue
                rid = str(req_id)
                q = self._sample_waves.get(rid)
                if q is None:
                    q = deque()
                    self._sample_waves[rid] = q
                q.append(wave)

    def take_sample_wave(self, req_id: str) -> int | None:
        with self._lock:
            rid = str(req_id)
            q = self._sample_waves.get(rid)
            if not q:
                return None
            wave = q.popleft()
            if not q:
                self._sample_waves.pop(rid, None)
            return wave

    def pending(self, req_id: str) -> bool:
        """True when a stamp is still unconsumed (async output not yet materialized)."""
        with self._lock:
            q = self._sample_waves.get(str(req_id))
            return bool(q)

    def discard(self, req_id: str) -> None:
        with self._lock:
            self._sample_waves.pop(str(req_id), None)

    def discard_many(self, req_ids: list[str] | None) -> None:
        # B2: reap-time cleanup — finished requests must not accumulate stamps.
        if not req_ids:
            return
        with self._lock:
            for req_id in req_ids:
                if req_id:
                    self._sample_waves.pop(str(req_id), None)


# ---- dump quota ----


class DumpQuota:
    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self._runtime_config = runtime_config
        self._total_count = 0
        self._last_ts: float | None = None
        self._lock = threading.Lock()
        self.sync_from_config()

    def sync_from_config(self) -> None:
        self._max_times = int(self._runtime_config.dump_max_times())
        self._cooldown = float(self._runtime_config.dump_cooldown_seconds())

    @property
    def total_count(self) -> int:
        return self._total_count

    @property
    def max_times(self) -> int:
        return self._max_times

    def can_consume(self, *, consume_quota: bool = True) -> bool:
        if not consume_quota:
            return True
        if self._max_times <= 0:
            return False
        if self._total_count >= self._max_times:
            return False
        if self._last_ts is not None and self._cooldown > 0:
            if time.time() - self._last_ts < self._cooldown:
                return False
        return True

    def try_consume(self, *, consume_quota: bool = True) -> bool:
        """Atomic check + consume (B'4).

        ``can_consume()`` followed by a separate ``consume()`` leaves a window
        where a hot-reloaded quota/cooldown boundary flips between the two —
        snapshots already captured then get dropped without being counted.
        """
        with self._lock:
            if not self.can_consume(consume_quota=consume_quota):
                return False
            if consume_quota:
                self._total_count += 1
                self._last_ts = time.time()
            return True

    def refund(self, *, consume_quota: bool = True) -> None:
        """Give back one unit when a consumed dump captured nothing.

        Also clears cooldown (``_last_ts``): empty queue / all-skip arms must
        not block the next auto dump for ``auto_cooldown_seconds``.
        """
        if not consume_quota:
            return
        with self._lock:
            if self._total_count > 0:
                self._total_count -= 1
            self._last_ts = None

    def snapshot(self) -> tuple[int, int]:
        return self._total_count, self._max_times


# ---- request store ----

# Finished but sample_waves still non-empty this many real-steps later → force reap.
MAX_DEFERRED_REAP_WAVES = 8
# Post-reap late async appends: remember recently cleared ids.
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
        # Finished-but-not-yet-reaped ids (subset of ``_by_req`` keys) so the
        # per-step reap scan is O(finished) instead of O(all requests).
        # Invariant (under ``_lock``): rid in _finished ⇔ state.finished.
        self._finished: set[str] = set()
        # Soft deps (snapshot wave cache, …): called after state is popped.
        self._on_clear: list[Callable[[str], None]] = []
        self.max_deferred_waves = MAX_DEFERRED_REAP_WAVES
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
        """Mark requests finished; keep state until :meth:`list_reapable` / reap sweep.

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
                self._finished.add(rid)
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
        # Post-reap late append stamps finished=True without a mark (zombie).
        # Start the defer clock on first reap scan so max_deferred_waves can
        # still force-reap when cpu_jobs is stuck.
        if mark is None:
            if not state.finished:
                return True
            mark = int(current_wave)
            state.finish_mark_wave = mark
        if state.cpu_jobs > 0:
            if int(current_wave) - mark < int(self.max_deferred_waves):
                return False
        if int(current_wave) - mark >= int(self.max_deferred_waves):
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
        """Finished reqs whose drain probe is clear (or past defer cap).

        Scans only the finished index (O(finished)), not every live request.
        """
        w = int(current_wave)
        with self._lock:
            out: list[str] = []
            for rid in self._finished:
                if self._ready_to_reap_locked(self._by_req[rid], current_wave=w):
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
                self._finished.discard(rid)
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
                    self._finished.add(rid)
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
