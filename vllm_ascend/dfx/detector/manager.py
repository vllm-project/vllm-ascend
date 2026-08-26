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

"""Detector manager facade: stage hooks over a private detector registry.

Concrete detectors are private references held by ``DetectorManager``; callers
(``DfxProcessor`` / model runners) only use the stage hooks, never the detector
instances. The internal ``DetectorRegistry`` keeps iteration / clear-finished
without exposing its public surface to the outside.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.base import AnomalyDetector
from vllm_ascend.dfx.detector.block_kv import BlockKvDetector
from vllm_ascend.dfx.detector.logits_finite import LogitsFiniteDetector
from vllm_ascend.dfx.detector.output_substring import OutputSubstringDetector
from vllm_ascend.dfx.detector.position_alignment import PositionAlignmentDetector
from vllm_ascend.dfx.detector.registry import DetectorRegistry
from vllm_ascend.dfx.detector.spec_acceptance import SpecAcceptanceDetector
from vllm_ascend.dfx.detector.token_logprob import TokenLogprobDetector
from vllm_ascend.dfx.detector.token_repeat import TokenRepeatDetector
from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager
from vllm_ascend.dfx.request_state import RequestDfxStore
from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

logger = init_logger_ascend(__name__)


class DetectorManager:
    """Owns detectors and exposes stage hooks only.

    Callers (``DfxProcessor`` / runners) use ``check_after_spec`` /
    ``check_before_sample`` / ``check_after_sample`` / ``check_kv_block_writes`` /
    ``clear_finished`` (reap path) only.
    Concrete detectors stay private (``_spec_det`` & co.); ``get`` exists solely
    for alert routing in ``DfxProcessor._handle_alert``.
    """

    def __init__(
        self,
        *,
        dfx_config: DfxRuntimeConfig,
        runner: Any,
        is_related_request: Callable[[str, int | None], bool] | None = None,
        tokenizer_provider: Callable[[], Any | None] | None = None,
        detection_gate: Callable[[], bool] | None = None,
        detection_skip_reason: Callable[[], str | None] | None = None,
    ) -> None:
        self._runner = runner
        self._dfx_config = dfx_config
        # Anomaly detection gate (rank / dump / detector-on), owned by the
        # caller (``DfxProcessor`` → ``Dumper``). None = always run.
        self._detection_gate = detection_gate
        self._detection_skip_reason = detection_skip_reason
        # Private concrete references (constructed once; no registry get+assert).
        self._spec_det = SpecAcceptanceDetector(
            dfx_config=dfx_config,
            runner=runner,
            is_related_request=is_related_request,
        )
        self._token_det = TokenLogprobDetector(
            dfx_config=dfx_config,
            runner=runner,
        )
        self._output_substring_det = OutputSubstringDetector(
            dfx_config=dfx_config,
            runner=runner,
            tokenizer_provider=tokenizer_provider,
        )
        self._token_repeat_det = TokenRepeatDetector(
            dfx_config=dfx_config,
            runner=runner,
        )
        self._block_kv_det = BlockKvDetector(
            dfx_config=dfx_config,
            runner=runner,
        )
        self._position_det = PositionAlignmentDetector(
            dfx_config=dfx_config,
            runner=runner,
        )
        self._logits_finite_det = LogitsFiniteDetector(
            dfx_config=dfx_config,
            runner=runner,
        )
        # Internal ordered registry: iterate for clear_finished; not public.
        self._registry = DetectorRegistry()
        for det in (
            self._spec_det,
            self._token_det,
            self._output_substring_det,
            self._token_repeat_det,
            self._block_kv_det,
            self._position_det,
            self._logits_finite_det,
        ):
            self._registry.register(det)
        # stop_after_alert flags live on RequestDfxStore (RequestDfxState).

    def get(self, anomaly_type: str) -> AnomalyDetector | None:
        """Resolve a detector for alert routing (``DfxProcessor._handle_alert`` only)."""
        return self._registry.get(anomaly_type)

    def clear_finished(self, req_id: str) -> None:
        """Drop per-request detector state when a request finishes.

        Shared fields (IO / filter / waves / dump_finish) are cleared by
        :meth:`RequestDfxStore.clear`. This also clears ``stopped_after_alert``
        so direct callers / tests can re-detect without popping the whole state.
        Prefer Store.clear from ``DfxProcessor._reap_finished_requests``.
        """
        state = RequestDfxStore.get().get_state(req_id)
        if state is not None:
            state.stopped_after_alert = False
        for det in self._registry:
            det.clear_finished(req_id)

    def token_logprob_topk_if_enabled(self) -> int | None:
        """Return token-logprob top-k when that detector is enabled; else None.

        With hot-reload off, skip per-sample ``refresh_from_config`` when the
        detector is already known disabled (default service path).
        """
        if self._dfx_config is not None and self._dfx_config.hot_reload_enabled:
            self._token_det.refresh_from_config()
        elif not self._token_det.enabled:
            return None
        if not self._token_det.enabled:
            return None
        topk = int(self._token_det.topk)
        return topk if topk > 0 else None

    def apply_dfx_config(self) -> None:
        """All-rank hook after DFX JSON sync — refresh deps that may force flags off.

        ``token_logprob`` needs msprobe; if missing, force ``enabled=false`` and
        persist on the JSON writer. Must run on every rank (including early PP
        writers that never sample), not only on the detect / sample path.
        Also refresh the newer native detectors so hot-reload flips take effect.
        """
        self._token_det.refresh_from_config()
        self._block_kv_det.refresh_from_config()
        self._position_det.refresh_from_config()
        self._logits_finite_det.refresh_from_config()
        # Spec / substring / repeat also pull knobs from JSON on enable flips.
        self._spec_det.refresh_from_config()
        self._output_substring_det.refresh_from_config()
        self._token_repeat_det.refresh_from_config()

    # ---- detection gating -------------------------------------------------

    def _gated(self, stage: str, *, ignore_dump_busy: bool = False) -> bool:
        """True when anomaly detection is gated off this step; logs skip reason once.

        ``stage`` is a short tag (``after_spec`` / ``after_sample`` / ``kv_block``)
        for the once-per-process skip log. Gate checks live here so callers never
        re-implement them per hook.

        ``ignore_dump_busy``: still run when pending/active dump (same-step
        follow-on detectors such as block_kv after logits/position already armed).
        """
        if self._detection_gate is None:
            return False

        def _call_gate() -> bool:
            try:
                return bool(self._detection_gate(ignore_dump_busy=ignore_dump_busy))  # type: ignore[misc]
            except TypeError:
                if not ignore_dump_busy:
                    return bool(self._detection_gate())
                # Legacy gate without kwarg: treat dump-busy as not gated.
                if self._detection_skip_reason is not None:
                    try:
                        reason = self._detection_skip_reason()
                    except TypeError:
                        reason = None
                    if reason in (
                        "pending_dump already armed",
                        "msprobe dump already active",
                    ):
                        return True
                return bool(self._detection_gate())

        if _call_gate():
            return False
        reason = None
        if self._detection_skip_reason is not None:
            try:
                reason = self._detection_skip_reason(ignore_dump_busy=ignore_dump_busy)  # type: ignore[misc]
            except TypeError:
                reason = self._detection_skip_reason()
        if reason and int(getattr(self._runner, "tp_rank", 0)) == 0:
            logger.info_once(
                "[Anomaly detect short] skip gate (%s): %s (any_detector=%s dump.enabled=%s)",
                stage,
                reason,
                self._dfx_config.any_detector_enabled(),
                self._dfx_config.dump_enabled(),
            )
        return True

    # ---- stage hooks ------------------------------------------------------

    def _stop_after_alert(self) -> bool:
        return bool(self._dfx_config.stop_after_alert())

    def _needs_io_for_dump_finish(self) -> bool:
        """True when dump_finish sidecars may still need cumulative IO on TP0."""
        dfx = getattr(self._runner, "dfx", None)
        dumper = getattr(dfx, "dumper", None) if dfx is not None else None
        if dumper is None:
            return False
        needs = getattr(dumper, "needs_io_for_dump_finish", None)
        return bool(needs()) if callable(needs) else False

    def _mark_alerted(self, alerts: list[AnomalyAlert]) -> None:
        """Stop detecting requests that just produced an anomaly."""
        store = RequestDfxStore.get()
        for alert in alerts:
            if alert.req_id:
                store.mark_stopped_after_alert(alert.req_id)

    def check_after_spec(
        self,
        sampled_tokens: Any,
        accepted_token_nums: Any,
    ) -> list[AnomalyAlert]:
        """Run spec-acceptance detect only (no cumulative IO append).

        Accepted tokens are recorded once in :meth:`check_after_sample` from
        the engine's validated sampled ids. Appending here as well doubled
        MTP/Eagle output in reports (same-wave dedupe fails under async
        scheduling when ``clear_wave_cache`` runs before ``get_output``).
        """
        if self._gated("after_spec"):
            return []
        skip = RequestDfxStore.get().stopped_req_ids() if self._stop_after_alert() else None
        alerts = self._spec_det.check_all(sampled_tokens, accepted_token_nums, skip_req_ids=skip)
        if skip is not None:
            self._mark_alerted(alerts)
        return alerts

    def check_after_sample(
        self,
        sampled_token_ids: Any,
        logprobs_lists: Any,
        req_ids: list[str] | None = None,
    ) -> list[AnomalyAlert]:
        """Append sample tokens to IO buffer; run post-sample detectors.

        Order: ``token_logprob`` → ``output_substring`` → ``token_repeat``.

        ``sampled_token_ids`` is the sole path that appends to cumulative IO
        (including MTP/Eagle accepted tokens). When anomaly detection is gated
        off, TP0 still appends when ``log.print_output_on_finish=true`` (finish
        log for every req) or when a dump finish sidecar may still need
        cumulative output (pending / active dump / already-activated finish
        meta). Appends happen only on steps where those gates are already true
        — there is no backfill of tokens produced while the flag was off;
        mid-request enable may leave finish logs partial or empty. Detection
        then skips ``stop_after_alert`` reqs by id — never by row-subsetting —
        so ``req_idx`` stays aligned with ``input_batch`` (filters / dump related).

        ``OutputSubstringDetector`` and ``TokenRepeatDetector`` are called with
        ``sampled_token_ids=None`` so they read the shared cumulative IO buffer
        instead of re-appending (avoids double count).

        With ``detector.stop_after_alert`` (default true) a request keeps being
        checked on every step until it produces an anomaly; afterwards it is
        skipped entirely so the same anomaly does not write endless reports.
        """
        resolved_ids = req_ids
        if resolved_ids is None:
            input_batch = getattr(self._runner, "input_batch", None)
            resolved_ids = list(getattr(input_batch, "req_ids", None) or [])

        # B10 fix: always append on TP0 even when detect is gated off.
        # Bug #5 half-fix gated append by print_output_on_finish / dump_finish,
        # which left a gap in cumulative IO when those flags flipped mid-flight.
        # output_substring / token_repeat read the same buffer → blinded for
        # the gap. Append is cheap (per-req list extend); only detection is
        # gated.
        if self._gated("after_sample"):
            if int(getattr(self._runner, "tp_rank", 0)) == 0:
                RequestIoSnapshotManager.get().append_batch(resolved_ids, sampled_token_ids)
            return []

        # Detect path: always accumulate once before detectors read cumulative IO.
        RequestIoSnapshotManager.get().append_batch(resolved_ids, sampled_token_ids)

        skip = RequestDfxStore.get().stopped_req_ids() if self._stop_after_alert() else None
        if skip is not None and resolved_ids and all(rid in skip for rid in resolved_ids if rid):
            # Entire batch already alerted: IO updated; no further detect / reports.
            return []

        alerts: list[AnomalyAlert] = []
        alerts.extend(
            self._token_det.check_all(
                sampled_token_ids=sampled_token_ids,
                logprobs_lists=logprobs_lists,
                req_ids=resolved_ids,
                skip_req_ids=skip,
            )
        )
        # Substring + token_repeat share cumulative IO (already appended); pass
        # None to avoid a second append_batch inside each detector.check_all.
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
        if skip is not None:
            self._mark_alerted(alerts)
        return alerts

    def check_before_sample(
        self,
        *,
        scheduler_output: Any,
        logits: Any,
        positions: Any = None,
        total_scheduled_tokens: int = 0,
        logits_indices: Any = None,
        input_batch: Any = None,
    ) -> list[AnomalyAlert]:
        """Run pre-sample detectors (logits finite, then position alignment).

        With ``stop_after_alert``, a req that alerts in logits is marked before
        position runs in the same call, so the same step does not double-report.
        """
        if self._gated("before_sample"):
            return []
        stop = self._stop_after_alert()
        skip = RequestDfxStore.get().stopped_req_ids() if stop else None
        alerts: list[AnomalyAlert] = []
        logits_alerts: list[AnomalyAlert] = []
        for alert in self._logits_finite_det.check_all(
            logits=logits,
            logits_indices=logits_indices,
            input_batch=input_batch,
        ):
            if skip is not None and alert.req_id in skip:
                continue
            logits_alerts.append(alert)
            alerts.append(alert)
        if stop and logits_alerts:
            self._mark_alerted(logits_alerts)
            skip = RequestDfxStore.get().stopped_req_ids()
        for alert in self._position_det.check_all(
            scheduler_output=scheduler_output,
            positions=positions,
            total_scheduled=total_scheduled_tokens,
            input_batch=input_batch,
        ):
            if skip is not None and alert.req_id in skip:
                continue
            alerts.append(alert)
        if stop:
            # Position-only hits (logits already marked above).
            self._mark_alerted(alerts)
        return alerts

    def check_kv_block_writes(
        self,
        req_id: str,
        block_ids: list[int],
        wave: int,
    ) -> list[AnomalyAlert]:
        """Run block KV integrity checks before ``record_writes``.

        Ignores dump-busy gate so same-step pending dump (armed by
        logits/position) does not skip KV checks.
        """
        if self._gated("kv_block", ignore_dump_busy=True):
            return []
        skip = RequestDfxStore.get().stopped_req_ids() if self._stop_after_alert() else None
        if skip is not None and req_id in skip:
            return []
        alerts = self._block_kv_det.check_writes(req_id, block_ids, wave)
        if skip is not None:
            self._mark_alerted(alerts)
        return alerts
