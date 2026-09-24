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

"""Report / alert / manual-trigger mixin for RuntimeGuardProcessor."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_guard.detector.base import AnomalyDetector
from vllm_ascend.observability.runtime_guard.incident import Incident
from vllm_ascend.observability.runtime_guard.io_snapshot import RequestIoSnapshotManager
from vllm_ascend.observability.runtime_guard.kv_block_meta import block_ids_for_request
from vllm_ascend.observability.runtime_guard.manual_trigger import (
    MANUAL_TRIGGER_REQ_ID,
    MANUAL_TRIGGER_TYPE,
    TriggerEvent,
    iter_local_request_rows,
)
from vllm_ascend.observability.runtime_guard.rank_gate import (
    is_action_leader_rank,
    should_dump_kv_on_rank,
)
from vllm_ascend.observability.runtime_guard.token_utils import decode_token_ids, load_model_tokenizer

logger = init_logger_ascend(__name__)


class RuntimeGuardReportMixin:
    """Incident report arming, manual trigger, finish print, tokenizer helpers."""

    # Attributes provided by RuntimeGuardProcessor (mixin composition).
    runner: Any
    runtime_config: Any
    wave_tracker: Any
    action_executor: Any
    _report_tokenizer: Any | None
    _report_tokenizer_failed: bool

    # Defined on RuntimeGuardDumpMixin / RuntimeGuardProcessor.
    _run_kv_dumps: Any

    def _maybe_fire_manual_local(self, *, allow_arm: bool) -> None:
        """Fire ``manual_dump`` locally on each last-PP TP (no job bcast).

        Config update (or already-armed in-memory count) → every dump rank
        enumerates its local requests and D2H's its own shards. Auto single-req
        dumps still use the TP dump lane. All dump ranks decrement in-memory
        ``manual_dump`` together (JSON writer persists at 0).
        """
        if not allow_arm:
            return
        if not should_dump_kv_on_rank():
            return
        cfg = self.runtime_config
        remaining = cfg.manual_trigger_count()
        if remaining <= 0:
            return
        if not cfg.dump_enabled():
            logger.warning(
                "[runtime_guard manual_trigger] dump.manual_dump=%s but dump inactive; not consuming",
                remaining,
            )
            return
        rows = self._batch_request_io_rows()
        if not rows:
            logger.debug(
                "[runtime_guard manual_trigger] dump.manual_dump deferred (no local rows); remaining=%d",
                remaining,
            )
            return

        wave = self.wave_tracker.current_wave()
        arm_id = uuid4().hex
        continuous = cfg.manual_trigger_continuous()
        if continuous:
            logger.info("[runtime_guard manual_trigger] dump.manual_dump event (continuous)")
            remaining_after: bool | int = True
        else:
            remaining_after = max(0, remaining - 1)
            logger.info(
                "[runtime_guard manual_trigger] dump.manual_dump event (remaining_after_arm=%d)",
                remaining_after,
            )

        if is_action_leader_rank(self.runner):
            self._handle_manual_trigger(
                TriggerEvent(
                    trigger_type=MANUAL_TRIGGER_TYPE,
                    req_id=MANUAL_TRIGGER_REQ_ID,
                    detail={
                        "source": "dump.manual_dump",
                        "manual_trigger_remaining_after": remaining_after,
                    },
                    consume_quota=False,
                ),
                write_report=True,
                inject_dump_kv=False,
                consume=False,
            )

        jobs = [
            {
                "req_id": str(req_id),
                "incident_type": MANUAL_TRIGGER_TYPE,
                "wave": wave,
                "arm_id": arm_id,
                "consume_quota": False,
            }
            for req_id, _idx in rows
            if req_id
        ]
        if jobs:
            self._run_kv_dumps(jobs)

        if cfg.consume_manual_trigger():
            logger.info(
                "[runtime_guard manual_trigger] manual_dump consumed, remaining=%d",
                cfg.manual_trigger_count(),
            )

    def _maybe_print_output_on_finish(self, finished_req_ids: Any, io_mgr: RequestIoSnapshotManager) -> None:
        """Log output_token_ids + text for finished reqs (last-PP TP0 only).

        Content comes from runtime_guard cumulative IO accumulated while
        ``log.print_output_on_finish`` was true on sample steps (no historical
        backfill). Mid-request hot-enable may print a partial sequence or
        ``output_token_count=0`` / empty text if nothing was appended after
        enable. See ``RuntimeConfig.log_print_output_on_finish``.
        """
        # Print on the report-leader rank (last-PP TP0): non-last PP ranks
        # never sample and hold no cumulative IO, and other TP ranks would
        # duplicate the print. The gate must read the process groups — v1
        # runners often lack ``tp_rank`` and an attribute fallback would
        # resolve 0 on every TP rank.
        if not is_action_leader_rank(self.runner):
            return
        tokenizer = self._get_detector_tokenizer()
        max_ids = self.runtime_config.report_max_output_token_ids()
        for req_id in finished_req_ids:
            if not req_id:
                continue
            snap = io_mgr.snapshot(self.runner, req_id, None, include_token_ids=True, use_cache=False)
            ids = list(snap.output_token_ids or [])
            truncated = False
            if max_ids > 0 and len(ids) > max_ids:
                ids = ids[:max_ids]
                truncated = True
            text = ""
            if tokenizer is not None and ids:
                try:
                    text = decode_token_ids(tokenizer, ids)
                except Exception as exc:
                    text = f"<decode failed: {exc}>"
            elif tokenizer is None:
                text = "<tokenizer unavailable>"
            logger.info(
                "[runtime_guard print_output] req_id=%s output_token_count=%d truncated=%s "
                "output_token_ids=%s output_text=%r",
                req_id,
                snap.output_token_count,
                truncated,
                ids,
                text,
            )

    def _handle_alert(
        self,
        alert: Incident,
        *,
        detector: AnomalyDetector | None = None,
        write_report: bool = True,
        arm_wave: int | None = None,
        action_override: list[str] | None = None,
    ) -> None:
        if not write_report and action_override is None:
            return
        if alert.block_ids is None or not alert.block_ids:
            alert.block_ids = block_ids_for_request(
                self.runner,
                alert.req_id,
                alert.req_idx,
                input_batch=getattr(self, "_last_input_batch", None),
            )
        if arm_wave is not None:
            alert.wave = arm_wave
        elif alert.wave is None:
            alert.wave = self.wave_tracker.current_wave()
        if detector is not None:
            detector.on_alert_armed(alert)
        detail = alert.to_report_detail()
        include_ids = self.runtime_config.report_save_sensitive_info()
        io_mgr = RequestIoSnapshotManager.get()
        # Never reuse same-wave IO cache for reports (S16): substring may have
        # cached an earlier include_token_ids snapshot in this CPU job.
        snap = io_mgr.snapshot(
            self.runner,
            alert.req_id,
            alert.req_idx,
            include_token_ids=include_ids,
            use_cache=False,
            scheduler_output=getattr(self, "_scheduler_output_for_step", None),
        )
        detail = io_mgr.merge_into_detail(detail, snap)
        detail = self._enrich_detail_with_block_meta(
            detail,
            alert.req_id,
            alert.req_idx,
        )
        self.action_executor.handle(
            alert,
            detail=detail,
            tokenizer=self._get_report_tokenizer(),
            action_override=action_override,
            write_report=write_report,
        )

    def _handle_manual_trigger(
        self,
        trigger: TriggerEvent,
        *,
        write_report: bool = True,
        inject_dump_kv: bool = True,
        consume: bool = True,
    ) -> None:
        """Leader-only: one async report per live req; optional DumpKvAction queue.

        End-of-wave local fire uses ``inject_dump_kv=False`` so every last-PP TP
        D2H's itself via ``_run_kv_dumps``. Each real req gets its own report
        (K-13 / W3-2) via ``ActionExecutor`` (ReportAction commit on the queue).
        """
        if not is_action_leader_rank(self.runner):
            return
        batch_rows = self._batch_request_io_rows()
        include_ids = self.runtime_config.report_save_sensitive_info()
        io_mgr = RequestIoSnapshotManager.get()
        so = getattr(self, "_scheduler_output_for_step", None)
        requests_detail: list[dict[str, Any]] = []
        for req_id, req_idx in batch_rows:
            snap = io_mgr.snapshot(
                self.runner,
                req_id,
                req_idx,
                include_token_ids=include_ids,
                scheduler_output=so,
            )
            entry = {"req_id": req_id, "req_idx": req_idx}
            entry.update(snap.as_detail_fields())
            entry = self._enrich_detail_with_block_meta(entry, req_id, req_idx)
            requests_detail.append(entry)

        wave = self.wave_tracker.current_wave()
        tokenizer = self._get_report_tokenizer()
        base_detail = trigger.to_report_detail()
        n_batch = len(requests_detail)

        if write_report:
            for entry in requests_detail:
                rid = str(entry.get("req_id") or "")
                if not rid or rid == MANUAL_TRIGGER_REQ_ID:
                    continue
                per_detail = dict(base_detail)
                per_detail["trigger_req_id"] = MANUAL_TRIGGER_REQ_ID
                per_detail["num_requests_in_batch"] = n_batch
                per_detail.update(entry)
                bids = entry.get("block_ids")
                block_ids = [int(x) for x in bids] if isinstance(bids, list) else []
                self.action_executor.handle(
                    Incident(
                        incident_type=trigger.trigger_type,
                        req_id=rid,
                        detail=per_detail,
                        consume_quota=False,
                        block_ids=block_ids,
                        wave=wave,
                    ),
                    detail=per_detail,
                    tokenizer=tokenizer,
                    write_report=True,
                    inject_manual_dump_kv=False,
                    action_override=["report"],
                )

        if inject_dump_kv:
            detail = dict(base_detail)
            detail["num_requests"] = n_batch
            detail["requests"] = requests_detail
            dump_block_ids: list[int] = []
            if requests_detail:
                raw = requests_detail[0].get("block_ids")
                if isinstance(raw, list):
                    dump_block_ids = [int(x) for x in raw]
            self.action_executor.handle(
                Incident(
                    incident_type=trigger.trigger_type,
                    req_id=MANUAL_TRIGGER_REQ_ID,
                    detail=detail,
                    consume_quota=False,
                    block_ids=dump_block_ids,
                    wave=wave,
                ),
                detail=detail,
                tokenizer=tokenizer,
                write_report=False,
                inject_manual_dump_kv=True,
            )

        if consume and self.runtime_config.consume_manual_trigger():
            logger.info(
                "[runtime_guard manual_trigger] manual_dump consumed, remaining=%d",
                self.runtime_config.manual_trigger_count(),
            )

    def _enrich_detail_with_block_meta(
        self,
        detail: dict[str, Any],
        req_id: str,
        req_idx: int | None = None,
    ) -> dict[str, Any]:
        """Attach ``block_ids`` when ``report.include_block_ids`` is on."""
        if not self.runtime_config.report_include_block_ids():
            return detail
        out = dict(detail)
        out["block_ids"] = block_ids_for_request(
            self.runner,
            req_id,
            req_idx,
            input_batch=getattr(self, "_last_input_batch", None),
        )
        return out

    def _batch_request_io_rows(self) -> list[tuple[str, int]]:
        """``(req_id, req_idx)`` for every request currently in the local batch."""
        return iter_local_request_rows(
            self.runner,
            getattr(self, "_scheduler_output_for_step", None),
        )

    def _get_detector_tokenizer(self) -> Any | None:
        """Tokenizer for detectors that need encode/decode (not gated on report flags)."""
        if self._report_tokenizer is not None:
            return self._report_tokenizer
        if self._report_tokenizer_failed:
            return None
        runner = getattr(self, "runner", None)
        try:
            tok = load_model_tokenizer(runner)
        except Exception as exc:
            self._report_tokenizer_failed = True
            logger.warning("[runtime_guard] tokenizer load failed error=%s", exc)
            return None
        if tok is None:
            # runner / model_config missing; retry on next call.
            return None
        self._report_tokenizer = tok
        return self._report_tokenizer

    def _get_report_tokenizer(self) -> Any | None:
        """Lazy-load tokenizer for report decode (detect rank only, once)."""
        if not self.runtime_config.report_save_sensitive_info() or not self.runtime_config.report_decode_token_ids():
            return None
        return self._get_detector_tokenizer()
