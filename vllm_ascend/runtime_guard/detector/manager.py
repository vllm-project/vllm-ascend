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

"""Detector manager: stage hooks over a private detector registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.runtime_guard.detector.base import (
    AnomalyDetector,
    DetectorRegistry,
    resolve_batch_req_ids,
)
from vllm_ascend.runtime_guard.detector.logits_finite import LogitsFiniteDetector
from vllm_ascend.runtime_guard.detector.output_substring import OutputSubstringDetector
from vllm_ascend.runtime_guard.detector.spec_acceptance import SpecAcceptanceDetector
from vllm_ascend.runtime_guard.detector.token_repeat import TokenRepeatDetector
from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard.io_snapshot import RequestIoSnapshotManager
from vllm_ascend.runtime_guard.rank_gate import runner_tp_rank
from vllm_ascend.runtime_guard.request_state import RequestGuardStore
from vllm_ascend.runtime_guard.token_utils import freeze_sampled_rows

if TYPE_CHECKING:
    from vllm_ascend.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)


@dataclass
class AfterSampleCpuSnapshot:
    """Host-side after-sample inputs frozen before ActionQueue CPU detect."""

    req_ids: list[str]
    sampled_token_ids: list[list[int]]
    skip_req_ids: set[str] | None


class DetectorManager:
    """Owns detectors; callers use stage hooks / ``get`` for alert routing."""

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig,
        runner: Any,
        tokenizer_provider: Callable[[], Any | None] | None = None,
        detection_gate: Callable[[], bool] | None = None,
        detection_skip_reason: Callable[[], str | None] | None = None,
    ) -> None:
        self._runner = runner
        self._runtime_config = runtime_config
        self._detection_gate = detection_gate
        self._detection_skip_reason = detection_skip_reason
        self._spec_det = SpecAcceptanceDetector(
            runtime_config=runtime_config,
            runner=runner,
        )
        self._output_substring_det = OutputSubstringDetector(
            runtime_config=runtime_config,
            runner=runner,
            tokenizer_provider=tokenizer_provider,
        )
        self._token_repeat_det = TokenRepeatDetector(
            runtime_config=runtime_config,
            runner=runner,
        )
        self._logits_finite_det = LogitsFiniteDetector(
            runtime_config=runtime_config,
            runner=runner,
        )
        self._registry = DetectorRegistry()
        for det in (
            self._spec_det,
            self._output_substring_det,
            self._token_repeat_det,
            self._logits_finite_det,
        ):
            self._registry.register(det)

    def rebind_runner(self, runner: Any) -> None:
        """Point this manager and all detectors at a new runner."""
        self._runner = runner
        for det in self._registry:
            det._runner = runner

    def get(self, incident_type: str) -> AnomalyDetector | None:
        """Resolve a detector for alert routing (``RuntimeGuardProcessor._handle_alert`` only)."""
        return self._registry.get(incident_type)

    def clear_finished(self, req_id: str) -> None:
        """Drop per-request detector state when a request finishes.

        Shared fields (IO / waves) are cleared by
        :meth:`RequestGuardStore.clear`. This also clears ``detection_stopped``
        so direct callers / tests can re-detect without popping the whole state.
        Prefer Store.clear from ``RuntimeGuardProcessor._reap_finished_requests``.
        """
        state = RequestGuardStore.get().get_state(req_id)
        if state is not None:
            state.detection_stopped = False
        for det in self._registry:
            det.clear_finished(req_id)

    def apply_runtime_config(self) -> None:
        """All-rank hook after runtime_config JSON sync — refresh detector deps."""
        for det in self._registry:
            det.refresh_from_config()

    # ---- detection gating -------------------------------------------------

    def _gated(self, stage: str) -> bool:
        """True when anomaly detection is gated off this step; logs skip reason once.

        ``stage`` is a short tag (``after_spec`` / ``before_sample`` / ``after_sample``)
        for the once-per-process skip log. Gate is rank-only (last PP + TP0);
        callers never re-implement it per hook.
        """
        if self._detection_gate is None:
            return False
        if bool(self._detection_gate()):
            return False
        reason = None
        if self._detection_skip_reason is not None:
            reason = self._detection_skip_reason()
        if reason and runner_tp_rank(self._runner) == 0:
            logger.info_once(
                "[runtime_guard: detect short] skip gate (%s): %s (any_detector=%s dump.enabled=%s)",
                stage,
                reason,
                self._runtime_config.any_detector_enabled(),
                self._runtime_config.dump_enabled(),
            )
        return True

    # ---- stage hooks ------------------------------------------------------

    def any_enabled_for_spec(self) -> bool:
        if self._runtime_config is not None and self._runtime_config.hot_reload_enabled:
            self._spec_det.refresh_from_config()
        return bool(self._spec_det.enabled)

    def any_after_sample_cpu_enabled(self) -> bool:
        """output_substring / token_repeat (not logits_finite)."""
        if self._runtime_config is not None and self._runtime_config.hot_reload_enabled:
            self._output_substring_det.refresh_from_config()
            self._token_repeat_det.refresh_from_config()
        return bool(self._output_substring_det.enabled or self._token_repeat_det.enabled)

    def check_after_spec(
        self,
        sampled_tokens: Any,
        accepted_token_nums: Any,
        req_ids: list[str] | None = None,
    ) -> list[Incident]:
        """Run spec-acceptance detect only (no cumulative IO append).

        Accepted tokens are recorded once in :meth:`check_after_sample` from
        the engine's validated sampled ids. Appending here as well doubled
        MTP/Eagle output in reports (same-wave dedupe fails under async
        scheduling when ``clear_wave_cache`` runs before ``get_output``).
        """
        if self._gated("after_spec"):
            return []
        skip = RequestGuardStore.get().stopped_req_ids()
        return self._spec_det.check_all(sampled_tokens, accepted_token_nums, skip_req_ids=skip, req_ids=req_ids)

    def after_sample_hot_path(
        self,
        sampled_token_ids: Any,
        req_ids: list[str] | None = None,
    ) -> tuple[list[Incident], AfterSampleCpuSnapshot | None]:
        """Append IO; drain logits_finite incidents; freeze CPU-detect inputs.

        Must run on ``check_after_sample`` / async ``get_output``. Logits
        ``.item()`` + hit resolve already happened in ``check_before_sample``;
        this only drains queued incidents (dump timing) then snapshots for
        :meth:`run_after_sample_cpu`.

        Skip set comes from ``report.max_per_req`` write-full
        (:meth:`RequestGuardStore.mark_detection_stopped`).
        """
        resolved_ids, need_io, io_owner, skip = self._after_sample_setup(req_ids)
        if self._gated("after_sample"):
            if need_io and io_owner:
                RequestIoSnapshotManager.get().append_batch(resolved_ids, sampled_token_ids)
            return [], None

        if need_io and io_owner:
            RequestIoSnapshotManager.get().append_batch(resolved_ids, sampled_token_ids)

        alerts: list[Incident] = []
        if resolved_ids and all(rid in skip for rid in resolved_ids if rid):
            alerts.extend(self._logits_finite_det.check_deferred(skip_req_ids=skip))
            return alerts, None

        alerts.extend(self._logits_finite_det.check_deferred(skip_req_ids=skip))

        if not resolved_ids or not self.any_after_sample_cpu_enabled():
            return alerts, None

        snap = AfterSampleCpuSnapshot(
            req_ids=list(resolved_ids),
            sampled_token_ids=freeze_sampled_rows(resolved_ids, sampled_token_ids),
            skip_req_ids=skip,
        )
        return alerts, snap

    def run_after_sample_cpu(self, snap: AfterSampleCpuSnapshot) -> list[Incident]:
        """output_substring → token_repeat after hot-path Store append.

        Both detectors read :class:`RequestGuardStore` only. Do **not** re-fold
        from ``snap.sampled_token_ids``: same-wave append dedupe can skip a
        second Store write while a frozen chunk would still be scored twice
        (W1-1 / R-04: ``content_tokens_seen`` > ``output_token_count`` on v2
        when after-sample ran twice with width-1 identical rows).
        """
        skip = snap.skip_req_ids if snap.skip_req_ids is not None else RequestGuardStore.get().stopped_req_ids()
        resolved_ids = snap.req_ids
        alerts: list[Incident] = []
        alerts.extend(
            self._output_substring_det.check_all(
                sampled_token_ids=None,
                req_ids=resolved_ids,
                skip_req_ids=skip,
            )
        )
        alerts.extend(
            self._token_repeat_det.check_all(
                sampled_token_ids=None,
                req_ids=resolved_ids,
                skip_req_ids=skip,
            )
        )
        return alerts

    def check_after_sample(
        self,
        sampled_token_ids: Any,
        req_ids: list[str] | None = None,
    ) -> list[Incident]:
        """Sync convenience: hot path + CPU detect (unit tests / direct callers).

        Production :meth:`RuntimeGuardProcessor.check_after_sample` enqueues
        :meth:`run_after_sample_cpu` instead of waiting.
        """
        alerts, snap = self.after_sample_hot_path(sampled_token_ids, req_ids=req_ids)
        if snap is not None:
            alerts.extend(self.run_after_sample_cpu(snap))
        return alerts

    def _after_sample_setup(self, req_ids: list[str] | None) -> tuple[list[str], bool, bool, set[str]]:
        resolved_ids = resolve_batch_req_ids(self._runner, req_ids)
        need_io = True
        if self._runtime_config is not None:
            need_io = bool(self._runtime_config.needs_cumulative_io())
        io_owner = runner_tp_rank(self._runner) == 0
        skip = RequestGuardStore.get().stopped_req_ids()
        return list(resolved_ids), need_io, io_owner, skip

    def check_before_sample(
        self,
        *,
        logits: Any,
        logits_indices: Any = None,
        input_batch: Any = None,
        **_unused: Any,
    ) -> list[Incident]:
        """Run pre-sample detectors (``logits_finite`` only)."""
        del _unused
        if self._gated("before_sample"):
            return []
        # logits_finite: check_all does .item() gate + hit-only resolve, then
        # enqueues host Incidents; check_deferred drains them on after_sample /
        # get_output so dump still runs before CPU detect. Pass stop-detect skip
        # so stopped reqs avoid enqueue (and whole-batch skip avoids .item()).
        skip = RequestGuardStore.get().stopped_req_ids()
        self._logits_finite_det.check_all(
            logits=logits,
            logits_indices=logits_indices,
            input_batch=input_batch,
            skip_req_ids=skip,
        )
        return []
