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

"""Runner-side DFX processing (config sync, detectors, dump/log/report sinks).

Owns construction of ``Dumper`` / detectors / ``DfxReportWriter`` so v1 and v2
model runners only hang a ``DfxProcessor`` and call thin step hooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.distributed.parallel_state import get_pp_group

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.base import AnomalyDetector
from vllm_ascend.dfx.detector.manager import DetectorManager
from vllm_ascend.dfx.dumper import Dumper
from vllm_ascend.dfx.input_filters import InputFilterManager, iter_batch_prompt_token_ids
from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager
from vllm_ascend.dfx.manual_trigger import ManualTriggerManager, TriggerEvent
from vllm_ascend.dfx.report import DfxReportWriter
from vllm_ascend.dfx.tokenizer import load_model_tokenizer
from vllm_ascend.logger import init_logger_ascend

# How many leading prompt token ids to suggest as a filter prefix example.
_PRINT_INPUT_PREFIX_HINT_LEN = 8

if TYPE_CHECKING:
    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

logger = init_logger_ascend(__name__)


class DfxProcessor:
    """Processes DFX step events: config → filter → detect → dump/report."""

    def __init__(self, runner: Any) -> None:
        ascend = runner.ascend_config
        dfx_config: DfxRuntimeConfig = ascend.dfx_config

        self.runner = runner
        self.dfx_config = dfx_config
        # Leader materializes JSON once; path is already logged at AscendConfig /
        # Dumper ready. Non-leaders no-op inside ensure_persisted.
        dfx_config.ensure_persisted()
        # Sole owner of InputFilterManager refresh (init + refresh_config).
        InputFilterManager.get().apply_from_config(dfx_config)
        # Runtime config is solely ``dfx_config`` (JSON).
        self.dumper = Dumper(runner, dfx_config=dfx_config)
        self.report_writer = DfxReportWriter(
            dfx_config.report_dir,
            save_sensitive_info=dfx_config.report_save_sensitive_info(),
            max_prompt_token_ids=dfx_config.report_max_prompt_token_ids(),
            max_output_token_ids=dfx_config.report_max_output_token_ids(),
            decode_token_ids=dfx_config.report_decode_token_ids(),
        )
        self._report_tokenizer: Any | None = None
        self._report_tokenizer_failed = False
        self.manual_triggers = ManualTriggerManager(dfx_config=dfx_config, runner=runner)
        # Stage-hook facade; concrete detectors stay inside the manager.
        self.detectors = DetectorManager(
            dfx_config=dfx_config,
            runner=runner,
            is_related_request=self.dumper.is_related_local_request,
            tokenizer_provider=self._get_detector_tokenizer,
            detection_gate=self.dumper.can_run_anomaly_detection,
            detection_skip_reason=self.dumper.anomaly_check_skip_reason,
        )

    # ---- step entry (all ranks) --------------------------------------------

    def refresh_config(self, *, allow_arm: bool = True) -> bool:
        """All-rank DFX config sync. Must not be skipped on early PP.

        ``allow_arm``: False on idle ``execute_dummy_batch``. Config sync still
        runs. ``dump.manual_trigger`` is **not** consumed on the dummy path — only a real
        ``allow_arm=True`` wave may ``check_all`` / arm (avoids clearing the
        JSON flag with no dump when the service is idle or a peer DP is busy).
        """
        logger.debug("[DFX sync] enter stage=refresh_config allow_arm=%s", allow_arm)
        RequestIoSnapshotManager.get().clear_wave_cache()
        changed = self.dfx_config.sync_dfx_config()
        # Always refresh filter chain from live config (broadcast may update
        # without ``changed``); detectors consult InputFilterManager singleton.
        InputFilterManager.get().apply_from_config(self.dfx_config)
        if changed:
            self.dumper.apply_dfx_config()
            self.report_writer.save_sensitive_info = self.dfx_config.report_save_sensitive_info()
            self.report_writer.max_prompt_token_ids = self.dfx_config.report_max_prompt_token_ids()
            self.report_writer.max_output_token_ids = self.dfx_config.report_max_output_token_ids()
            self.report_writer.decode_token_ids = self.dfx_config.report_decode_token_ids()
        # Dump limits sync only when config changed (via apply_dfx_config).
        trigger = self.manual_triggers.consume_once(allow_arm=allow_arm)
        if trigger is not None:
            self._handle_manual_trigger(trigger)
        self.maybe_print_input_token_ids_once(allow_arm=allow_arm)
        # Detector thresholds / enable flags: pulled lazily in each detector's
        # ``_precheck`` (and ``ensure_logprobs_for_detection``) so we do not
        # refresh every detector twice per step here.
        logger.debug("[DFX sync] leave stage=refresh_config changed=%s", changed)
        return changed

    def maybe_print_input_token_ids_once(self, *, allow_arm: bool = True) -> bool:
        """If ``input_filter.print_input_token_ids_once``, log prompts once then clear.

        Skipped on dummy waves (``allow_arm=False``) and when the batch has no
        printable prompts (flag stays true for the next real request).
        Logging is TP0-only; all ranks consume together when printing succeeds.
        """
        if not allow_arm or not self.dfx_config.print_input_token_ids_once():
            return False
        rows = iter_batch_prompt_token_ids(self.runner)
        if not rows:
            logger.debug(
                "[DFX print_input] print_input_token_ids_once set but no prompt_token_ids in batch; defer consume"
            )
            return False
        log_leader = int(getattr(self.runner, "tp_rank", 0)) == 0
        if log_leader:
            for req_id, req_idx, ids in rows:
                prompt_len = len(ids)
                hint_n = min(_PRINT_INPUT_PREFIX_HINT_LEN, prompt_len)
                prefix_hint = ids[:hint_n]
                logger.info(
                    "[DFX print_input] req_id=%s req_idx=%s length=%d prompt_token_ids=%s",
                    req_id,
                    req_idx,
                    prompt_len,
                    ids,
                )
                if hint_n > 0:
                    logger.info(
                        "[DFX print_input] filter hint req_id=%s length=%d "
                        "input_filter.filters example: "
                        '{"type":"input_token_id_prefix","mode":"include",'
                        '"prefixes":[%s]}',
                        req_id,
                        prompt_len,
                        prefix_hint,
                    )
        self.dfx_config.consume_print_input_token_ids_once()
        return True

    def sync_dump_pending_or(self, *, allow_arm: bool = True) -> bool:
        """Last-PP TP dump OR only (see ``Dumper.sync_dump_pending_or``)."""
        logger.debug("[DFX sync] enter stage=sync_dump_pending_or allow_arm=%s", allow_arm)
        ok = self.dumper.sync_dump_pending_or(allow_arm=allow_arm)
        logger.debug("[DFX sync] leave stage=sync_dump_pending_or ok=%s", ok)
        return ok

    def sync_for_step(self, *, allow_arm: bool = True) -> None:
        """Lockstep DFX sync for one engine wave (real step or idle dummy).

        ``refresh_config`` uses the per-DP sync group (or local file poll) and
        must run on every rank of that EngineCore each wave — including idle
        DP ranks that take ``execute_dummy_batch`` and skip ``execute_model``.
        Do not put this inside ``_dummy_run``: ``execute_model`` may already
        sync then call ``_dummy_run``. Never use a cross-DP full-world
        collective for config hot-reload.
        """
        runner = self.runner
        dp = getattr(runner, "dp_rank", "?")
        tp = getattr(runner, "tp_rank", "?")
        try:
            pp = get_pp_group().rank_in_group
        except Exception:
            pp = "?"
        logger.debug(
            "[DFX sync] enter sync_for_step allow_arm=%s dp=%s tp=%s pp=%s",
            allow_arm,
            dp,
            tp,
            pp,
        )
        try:
            self.refresh_config(allow_arm=allow_arm)
            self.sync_dump_pending_or(allow_arm=allow_arm)
        finally:
            logger.debug(
                "[DFX sync] leave sync_for_step allow_arm=%s dp=%s tp=%s pp=%s",
                allow_arm,
                dp,
                tp,
                pp,
            )

    # ---- sample / get_output hooks ----------------------------------------

    def clear_finished(self, finished_req_ids: Any) -> None:
        if not finished_req_ids:
            return
        io_mgr = RequestIoSnapshotManager.get()
        filt = InputFilterManager.get()
        for req_id in finished_req_ids:
            self.detectors.clear_finished(req_id)
            io_mgr.clear_req(req_id)
            filt.clear_req(req_id)

    def check_after_spec(
        self,
        sampled_tokens: Any,
        accepted_token_nums: Any,
    ) -> None:
        """Speculative step hook: record accepted tokens + run registered spec detectors.

        Detection gating (rank / dump / detector-on) lives in ``DetectorManager``.
        """
        for alert in self.detectors.check_after_spec(sampled_tokens, accepted_token_nums):
            self._handle_alert(alert, detector=self.detectors.get(alert.anomaly_type))

    def check_after_sample(
        self,
        sampled_token_ids: Any,
        logprobs_lists: Any,
        req_ids: list[str] | None = None,
    ) -> None:
        """Sample step hook: run all post-sample detectors (token / substring / …).

        Detection gating (rank / dump / detector-on) lives in ``DetectorManager``.
        """
        for alert in self.detectors.check_after_sample(
            sampled_token_ids=sampled_token_ids,
            logprobs_lists=logprobs_lists,
            req_ids=req_ids,
        ):
            self._handle_alert(alert, detector=self.detectors.get(alert.anomaly_type))

    # ---- dump lifecycle (delegated so callers never touch ``Dumper``) ------

    def start_dump_data(self) -> None:
        """Start dump for this forward (delegated to ``Dumper``)."""
        self.dumper.start_dump_data()

    def finalize_dump_data(self, *, dump: bool = True) -> None:
        """Finalize dump for this forward (delegated to ``Dumper``)."""
        self.dumper.finalize_dump_data(dump=dump)

    def ensure_logprobs_for_detection(self) -> None:
        """Bump per-request top-k logprobs when token-logprob detect needs them.

        Clients need not set ``logprobs`` on the request when
        ``detector.token_logprob.enabled`` is on. Independent of ``dump.enabled``.
        Safe no-op when the check is disabled or the batch has no sampling state.
        """
        topk = self.detectors.token_logprob_topk_if_enabled()
        if topk is None:
            return
        runner = self.runner
        # v1 InputBatch: dict[str, int] num_logprobs + SamplingMetadata
        input_batch = getattr(runner, "input_batch", None)
        num_logprobs = getattr(input_batch, "num_logprobs", None) if input_batch is not None else None
        if isinstance(num_logprobs, dict):
            req_ids = getattr(input_batch, "req_ids", None) or []
            changed = False
            for req_id in req_ids:
                cur = num_logprobs.get(req_id)
                # None / missing: no logprobs. Keep full-vocab requests as-is.
                if cur is None or (0 <= cur < topk):
                    num_logprobs[req_id] = topk
                    changed = True
            if changed and hasattr(input_batch, "_make_sampling_metadata"):
                input_batch.sampling_metadata = input_batch._make_sampling_metadata()
                logger.info_once(
                    "[Anomaly token_logprob] forcing request top-k logprobs=%d for detection",
                    topk,
                )
            return

        # v2 SamplingStates: num_logprobs[req_state_idx], -1 == none
        sampler = getattr(runner, "sampler", None)
        states = getattr(sampler, "sampling_states", None) if sampler is not None else None
        if states is None or input_batch is None:
            return
        idx_mapping_np = getattr(input_batch, "idx_mapping_np", None)
        num_reqs = int(getattr(input_batch, "num_reqs", 0) or 0)
        if idx_mapping_np is None or num_reqs <= 0:
            return
        arr = states.num_logprobs
        changed = False
        for batch_i in range(num_reqs):
            req_state_idx = int(idx_mapping_np[batch_i])
            cur = int(arr[req_state_idx])
            # v2: -1 means no logprobs; positive N means top-N.
            if cur < 0 or cur < topk:
                arr[req_state_idx] = topk
                changed = True
        if changed:
            logger.info_once(
                "[Anomaly token_logprob] forcing request top-k logprobs=%d for detection (v2)",
                topk,
            )

    def _handle_alert(
        self,
        alert: AnomalyAlert,
        *,
        detector: AnomalyDetector | None = None,
        write_report: bool = True,
    ) -> None:
        """Arm dump (optional) and/or write report for an anomaly alert.

        - ``dump.enabled=false`` (detect-only): no dump arm; write report on detect.
        - ``dump.enabled=true``: arm dump. If arm fails (e.g. quota exhausted),
          still write report — detection evidence should not be lost.
        """
        dump_on = self.dfx_config.dump_enabled()
        if dump_on:
            self.dumper.handle_anomaly_alert(alert, detector=detector)
        else:
            if detector is not None:
                detector.on_alert_armed(alert)

        if not write_report:
            return

        if self.dfx_config.report_print_sampling_meta():
            self.save_sample_param(alert.req_id)

        detail = alert.to_report_detail()
        include_ids = self.dfx_config.report_save_sensitive_info()
        io_mgr = RequestIoSnapshotManager.get()
        snap = io_mgr.snapshot(
            self.runner,
            alert.req_id,
            alert.req_idx,
            include_token_ids=include_ids,
        )
        detail = io_mgr.merge_into_detail(detail, snap)

        self.report_writer.write(
            anomaly_type=alert.anomaly_type,
            req_id=alert.req_id,
            detail=detail,
            rank_tag=self.dumper.dump_rank_tag(),
            tokenizer=self._get_report_tokenizer(),
        )

    def _handle_manual_trigger(
        self,
        trigger: TriggerEvent,
        *,
        write_report: bool = True,
    ) -> None:
        """Arm dump (optional) and write report for control-plane manual trigger."""
        self.dumper.handle_manual_trigger(trigger)
        if not write_report:
            return

        batch_rows = self._batch_request_io_rows()
        if self.dfx_config.report_print_sampling_meta():
            for req_id, _req_idx in batch_rows:
                self.save_sample_param(req_id)

        include_ids = self.dfx_config.report_save_sensitive_info()
        io_mgr = RequestIoSnapshotManager.get()
        requests_detail: list[dict[str, Any]] = []
        for req_id, req_idx in batch_rows:
            snap = io_mgr.snapshot(
                self.runner,
                req_id,
                req_idx,
                include_token_ids=include_ids,
            )
            entry = {"req_id": req_id, "req_idx": req_idx}
            entry.update(snap.as_detail_fields())
            requests_detail.append(entry)

        detail = trigger.to_report_detail()
        detail["num_requests"] = len(requests_detail)
        detail["requests"] = requests_detail
        self.report_writer.write(
            anomaly_type=trigger.trigger_type,
            req_id=trigger.req_id,
            detail=detail,
            rank_tag=self.dumper.dump_rank_tag(),
            tokenizer=self._get_report_tokenizer(),
        )

    def _batch_request_io_rows(self) -> list[tuple[str, int]]:
        """``(req_id, req_idx)`` for every request currently in the local batch."""
        runner = self.runner
        input_batch = getattr(runner, "input_batch", None)
        req_ids = getattr(input_batch, "req_ids", None) if input_batch is not None else None
        if req_ids:
            return [(str(req_id), idx) for idx, req_id in enumerate(req_ids) if req_id]
        # Fallback: known requests map (no stable batch index).
        requests = getattr(runner, "requests", None)
        if isinstance(requests, dict) and requests:
            return [(str(req_id), -1) for req_id in requests if req_id]
        return []

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
            logger.warning("[DFX] tokenizer load failed error=%s", exc)
            return None
        if tok is None:
            # runner / model_config missing; retry on next call.
            return None
        self._report_tokenizer = tok
        return self._report_tokenizer

    def _get_report_tokenizer(self) -> Any | None:
        """Lazy-load tokenizer for report decode (detect rank only, once)."""
        if not self.dfx_config.report_save_sensitive_info() or not self.dfx_config.report_decode_token_ids():
            return None
        return self._get_detector_tokenizer()

    def save_sample_param(self, target_req_id: str) -> None:
        """Log sampling metadata for ``target_req_id`` (TP0 + last PP only)."""
        runner = getattr(self, "runner", None)
        if runner is None or not target_req_id:
            return
        try:
            if int(getattr(runner, "tp_rank", 0)) != 0:
                return
            if not get_pp_group().is_last_rank:
                return
        except Exception:
            return

        input_batch = getattr(runner, "input_batch", None)
        if input_batch is None:
            return
        sampling_metadata = getattr(input_batch, "sampling_metadata", None)
        if sampling_metadata is None:
            return
        req_ids = input_batch.req_ids
        for req_idx, req_id in enumerate(req_ids):
            if req_id != target_req_id:
                continue

            temp = sampling_metadata.temperature[req_idx].item() if sampling_metadata.temperature is not None else None
            topk = sampling_metadata.top_k[req_idx].item() if sampling_metadata.top_k is not None else None
            topp = sampling_metadata.top_p[req_idx].item() if sampling_metadata.top_p is not None else None

            freq_pen = sampling_metadata.frequency_penalties[req_idx].item()
            pres_pen = sampling_metadata.presence_penalties[req_idx].item()
            rep_pen = sampling_metadata.repetition_penalties[req_idx].item()

            req_bad_words = sampling_metadata.bad_words_token_ids.get(req_idx, [])
            req_output_tokens = (
                sampling_metadata.output_token_ids[req_idx]
                if sampling_metadata.output_token_ids and req_idx < len(sampling_metadata.output_token_ids)
                else []
            )
            req_spec_tokens = (
                sampling_metadata.spec_token_ids[req_idx]
                if sampling_metadata.spec_token_ids and req_idx < len(sampling_metadata.spec_token_ids)
                else None
            )
            if sampling_metadata.logprob_token_ids:
                req_logprob_tokens = sampling_metadata.logprob_token_ids.get(req_idx, [])
            else:
                req_logprob_tokens = None

            logger.info(
                "[SamplingMeta] req_id=%s req_idx=%d "
                "dp_rank=%d tp_rank=%d "
                "temperature=%.4f top_k=%s top_p=%.4f "
                "freq_pen=%.4f pres_pen=%.4f rep_pen=%.4f "
                "bad_words_group_num=%d output_tokens_len=%d spec_tokens_len=%s logprob_target_tokens_len=%s "
                "all_greedy=%s all_random=%s max_num_logprobs=%s",
                req_id,
                req_idx,
                runner.dp_rank,
                runner.tp_rank,
                temp if temp is not None else -1,
                topk,
                topp if topp is not None else 1.0,
                freq_pen,
                pres_pen,
                rep_pen,
                len(req_bad_words),
                len(req_output_tokens),
                len(req_spec_tokens) if req_spec_tokens else None,
                len(req_logprob_tokens) if req_logprob_tokens else None,
                sampling_metadata.all_greedy,
                sampling_metadata.all_random,
                sampling_metadata.max_num_logprobs,
            )
