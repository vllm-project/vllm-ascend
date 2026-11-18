#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import TYPE_CHECKING, Any
from weakref import ReferenceType, ref

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_config.schema import ConfigField, DetectorSchema, validate_detector_section
from vllm_ascend.observability.runtime_guard.detector.base import ConfigBackedDetector, resolve_batch_req_ids
from vllm_ascend.observability.runtime_guard.io import RequestIoSnapshotManager
from vllm_ascend.observability.runtime_guard.state import ILL_TYPE_REPEAT, Incident, RequestGuardStore

if TYPE_CHECKING:
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig
    from vllm_ascend.observability.runtime_guard.detector.pattern_backend import PatternBackend
    from vllm_ascend.observability.runtime_guard.state import RequestGuardState

logger = init_logger_ascend(__name__)


class TokenRepeatDetector(ConfigBackedDetector):
    """Pattern-based local token repetition detector (native CPU extension).

    Consumes the cumulative output stream from ``RequestIoSnapshotManager``
    (IO is appended once in after-sample, not in ``check_after_spec``). A
    per-req cursor tracks how far the stream has been folded so each id is
    pushed once. Uses the in-tree ``native_pattern`` CPU extension built with
    vllm-ascend.
    """

    incident_type = "token_repeat"
    section_key = "token_repeat"
    schema = DetectorSchema(
        section_key="token_repeat",
        stage="after_sample",
        help="Pattern-based token repetition (native CPU; no logprobs).",
        fields=(
            ConfigField("enabled", False, "bool", help="Master switch"),
            ConfigField(
                "think_end",
                -1,
                "int",
                min_value=-1,
                help="Explicit think-end token ID; -1 for chat mode",
            ),
        ),
    )

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig | None = None,
        runner: Any | None = None,
    ) -> None:
        self._lock = RLock()
        self._epoch = 0
        self._think_end = -1
        self._backend: PatternBackend | None = None
        self._states: dict[str, Any] = {}  # native PatternDetector state objects
        self._owners: dict[str, ReferenceType[RequestGuardState]] = {}
        self._start_offset: dict[str, int] = {}
        self._failed: set[str] = set()
        super().__init__(runtime_config=runtime_config, runner=runner, enabled=False)
        # How many cumulative output ids have already been pushed for each req.
        self._consumed_len: dict[str, int] = {}
        self._alerted: set[str] = set()
        if runtime_config is not None:
            self.refresh_from_config()

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def refresh_from_config(self) -> None:
        with self._lock:
            super().refresh_from_config()

    def _enable_pattern(self) -> None:
        # Re-enable/config changes begin a new monitored suffix. Shared IO may
        # still contain history retained for other enabled detectors.
        store = RequestGuardStore.get()
        for rid in store.list_req_ids():
            state = store.get_state(rid)
            if state is not None:
                self._consumed_len[rid] = len(state.output_token_ids)
                self._owners[rid] = ref(state)
        self.warmup()

    def _apply_detector_section(self, section: dict[str, Any] | None) -> None:
        if not isinstance(section, dict):
            return
        candidate = dict(section)
        validate_detector_section(self.schema, candidate)
        was_enabled, epoch = self.enabled, self._epoch
        super()._apply_detector_section(candidate)
        if was_enabled and not self.enabled and epoch == self._epoch:
            self.on_disabled()
        if self.enabled and (not was_enabled or epoch != self._epoch):
            self._enable_pattern()

    def warmup(self) -> None:
        """Prepare model-specific CPU tables at enable time, outside detection batches."""
        if not self.enabled or self._backend is not None:
            return
        from vllm_ascend.observability.runtime_guard.detector.pattern_backend import load_pattern
        from vllm_ascend.observability.runtime_guard.rank_gate import should_run_anomaly_check_on_rank

        if self._runner is not None and not should_run_anomaly_check_on_rank(self._runner):
            return
        try:
            self._backend = load_pattern(self._runner, self._think_end)
        except Exception:
            # Optional detection must not break model initialization or the bus.
            logger.exception(
                "[runtime_guard: token_repeat] pattern warmup failed; detection unavailable until fixed and re-enabled"
            )

    def rebind_runner(self, runner: Any, *, runtime_config: RuntimeConfig | None = None) -> None:
        with self._lock:
            self.on_disabled()
            self._enabled = False
            super().rebind_runner(runner, runtime_config=runtime_config)

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        think_end = int(getter("think_end", -1))
        if think_end != self._think_end:
            self.on_disabled()
            self._think_end = think_end

    def clear_finished(self, req_id: str, *, expected_state: RequestGuardState | None = None) -> None:
        with self._lock:
            if expected_state is not None and (
                (owner := self._owners.get(req_id)) is None or owner() is not expected_state
            ):
                return
            self._consumed_len.pop(req_id, None)
            self._alerted.discard(req_id)
            self._states.pop(req_id, None)
            self._owners.pop(req_id, None)
            self._start_offset.pop(req_id, None)
            self._failed.discard(req_id)

    def on_disabled(self) -> None:
        with self._lock:
            self._epoch += 1
            self._consumed_len.clear()
            self._alerted.clear()
            self._states.clear()
            self._owners.clear()
            self._start_offset.clear()
            self._failed.clear()
            self._backend = None

    def check_all(
        self,
        sampled_token_ids,
        req_ids=None,
        skip_req_ids=None,
        *,
        expected_epoch=None,
        expected_states=None,
    ) -> list[Incident]:
        with self._lock:
            self.refresh_from_config()
            if expected_epoch is not None and expected_epoch != self._epoch:
                return []
            if expected_states is not None:
                req_ids = [
                    rid
                    for rid in req_ids
                    if (expected := expected_states.get(rid)) is not None
                    and (owner := expected()) is not None
                    and RequestGuardStore.get().get_state(rid) is owner
                ]
            return self._check_all_locked(sampled_token_ids, req_ids, skip_req_ids)

    def _check_all_locked(
        self,
        sampled_token_ids: list[list[int]] | None,
        req_ids: list[str] | None = None,
        skip_req_ids: set[str] | None = None,
    ) -> list[Incident]:
        if not self.enabled:
            return []

        req_ids = resolve_batch_req_ids(self._runner, req_ids)
        if not req_ids:
            return []

        io_mgr = RequestIoSnapshotManager.get()
        if sampled_token_ids is not None:
            io_mgr.append_batch(req_ids, sampled_token_ids)

        return self._check_pattern(req_ids, skip_req_ids)

    def _incident(self, rid, idx, fired_at, reason, finish=False):
        self._alerted.add(rid)
        fired_at += self._start_offset.get(rid, 0)
        return Incident(
            incident_type=self.incident_type,
            req_id=rid,
            req_idx=idx,
            is_ill=True,
            ill_type=ILL_TYPE_REPEAT,
            detail={"fired_at": fired_at, "reason": reason, "finish_only": finish},
        )

    def _check_pattern(self, req_ids, skip_req_ids):
        backend = self._backend
        if backend is None:
            return []  # Warmup already logged the missing/invalid resource.
        keys, states, inputs, indices = [], [], [], []
        seen: set[str] = set()
        io = RequestIoSnapshotManager.get()
        for idx, rid in enumerate(req_ids):
            if not rid or rid in seen:
                continue
            seen.add(rid)
            owner = RequestGuardStore.get().get_state(rid)
            if owner is None:
                continue
            if rid in self._owners and self._owners[rid]() is not owner:
                self.clear_finished(rid)
            self._owners[rid] = ref(owner)
            consumed = self._consumed_len.get(rid, 0)
            total, ids = io.cumulative_output_since(rid, consumed)
            if consumed > total:
                self.clear_finished(rid)
                self._owners[rid] = ref(owner)
                total, ids = io.cumulative_output_since(rid, 0)
            if rid in self._failed:
                continue
            if rid in self._alerted or (skip_req_ids and rid in skip_req_ids):
                self._consumed_len[rid] = total
                continue
            if not ids:
                continue
            state = self._states.get(rid)
            if state is None:
                state = self._states[rid] = backend.start()
                self._owners[rid] = ref(owner)
                self._start_offset[rid] = self._consumed_len.get(rid, 0)
            keys.append(rid)
            states.append(state)
            inputs.append(ids)
            indices.append(idx)
        if not states:
            return []
        try:
            events = backend.feed_events(states, inputs)
        except Exception:
            logger.exception("[runtime_guard: token_repeat] pattern batch failed; recovering healthy request cursors")
            alerts = []
            for rid, state, idx, ids in zip(keys, states, indices, inputs):
                try:
                    receipt = backend.progress(state)
                    if receipt is None:
                        self._failed.add(rid)
                        continue
                    received, _, fired_at, reason = receipt
                    cursor = self._start_offset.get(rid, 0) + received
                    previous = self._consumed_len.get(rid, 0)
                    if not previous <= cursor <= previous + len(ids):
                        raise ValueError("pattern receipt outside submitted input")
                    self._consumed_len[rid] = cursor
                    if reason:
                        alerts.append(self._incident(rid, idx, fired_at, reason))
                except Exception:
                    self._failed.add(rid)
                    logger.exception("[runtime_guard: token_repeat] pattern receipt unavailable req=%s", rid)
            return alerts
        for rid, ids in zip(keys, inputs):
            self._consumed_len[rid] = self._consumed_len.get(rid, 0) + len(ids)
        return [self._incident(keys[slot], indices[slot], fired_at, reason) for slot, fired_at, reason in events]

    def finish_requests(self, req_ids):
        """Drain unread output and flush tails BEFORE shared IO is removed."""
        with self._lock:
            if not self.enabled:
                return []
            skipped = RequestGuardStore.get().stopped_req_ids()
            alerts: list[Incident] = []
            pending = list(req_ids)
            # Keep folding until every cursor catches up (or backend is gone).
            while pending and self._backend is not None:
                alerts.extend(self.check_all(None, pending, skipped))
                next_pending = []
                for rid in pending:
                    if rid in skipped or rid in self._alerted or rid in self._failed:
                        continue
                    owner = RequestGuardStore.get().get_state(rid)
                    if owner is None:
                        continue
                    if self._consumed_len.get(rid, 0) < len(owner.output_token_ids):
                        next_pending.append(rid)
                if next_pending == pending:
                    self._failed.update(pending)
                    logger.error(
                        "[runtime_guard: token_repeat] terminal pattern drain could not consume requests: %s",
                        pending,
                    )
                    break
                pending = next_pending
            backend = self._backend
            if backend is None:
                return alerts
            for rid in req_ids:
                state = self._states.get(rid)
                if state is None or rid in skipped or rid in self._alerted or rid in self._failed:
                    continue
                try:
                    result = backend.finish(state)
                    if result.is_ill:
                        alerts.append(self._incident(rid, None, result.fired_at, result.reason, True))
                except Exception:
                    self._failed.add(rid)
                    logger.exception("[runtime_guard: token_repeat] pattern finish failed req=%s", rid)
                    try:
                        receipt = backend.progress(state)
                        if receipt is not None and receipt[3]:
                            alerts.append(self._incident(rid, None, receipt[2], receipt[3], True))
                    except Exception:
                        logger.exception("[runtime_guard: token_repeat] pattern finish receipt unavailable req=%s", rid)
            return alerts
