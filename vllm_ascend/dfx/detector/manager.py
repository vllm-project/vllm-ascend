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

import torch

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.base import AnomalyDetector
from vllm_ascend.dfx.detector.output_substring import OutputSubstringDetector
from vllm_ascend.dfx.detector.registry import DetectorRegistry
from vllm_ascend.dfx.detector.spec_acceptance import SpecAcceptanceDetector
from vllm_ascend.dfx.detector.token_logprob import TokenLogprobDetector
from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager
from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

logger = init_logger_ascend(__name__)


class DetectorManager:
    """Owns detectors and exposes stage hooks only.

    Callers (``DfxProcessor`` / runners) use ``check_after_spec`` /
    ``check_after_sample`` / ``clear_finished`` only.
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
        # Internal ordered registry: iterate for clear_finished; not public.
        self._registry = DetectorRegistry()
        for det in (
            self._spec_det,
            self._token_det,
            self._output_substring_det,
        ):
            self._registry.register(det)
        # Requests that already produced an anomaly (stop_after_alert): no longer
        # run through detection, so the same anomaly does not write endless reports.
        self._stopped_req_ids: set[str] = set()

    def get(self, anomaly_type: str) -> AnomalyDetector | None:
        """Resolve a detector for alert routing (``DfxProcessor._handle_alert`` only)."""
        return self._registry.get(anomaly_type)

    def clear_finished(self, req_id: str) -> None:
        """Drop per-request state from every detector when a request finishes."""
        self._stopped_req_ids.discard(req_id)
        for det in self._registry:
            det.clear_finished(req_id)

    def token_logprob_topk_if_enabled(self) -> int | None:
        """Return token-logprob top-k when that detector is enabled; else None."""
        self._token_det.refresh_from_config()
        if not self._token_det.enabled:
            return None
        topk = int(self._token_det.topk)
        return topk if topk > 0 else None

    def apply_dfx_config(self) -> None:
        """All-rank hook after DFX JSON sync — refresh deps that may force flags off.

        ``token_logprob`` needs msprobe; if missing, force ``enabled=false`` and
        persist on the JSON writer. Must run on every rank (including early PP
        writers that never sample), not only on the detect / sample path.
        """
        self._token_det.refresh_from_config()

    # ---- detection gating -------------------------------------------------

    def _gated(self, stage: str) -> bool:
        """True when anomaly detection is gated off this step; logs skip reason once.

        ``stage`` is a short tag (``after_spec`` / ``after_sample``) for the
        once-per-process skip log. Gate checks live here so callers never
        re-implement them per hook.
        """
        if self._detection_gate is None or self._detection_gate():
            return False
        reason = self._detection_skip_reason() if self._detection_skip_reason is not None else None
        # Once per process on TP0 only: default-off / rank gate must not 2s-flood,
        # and must not spam every TP rank with the same skip reason.
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

    def _mark_alerted(self, alerts: list[AnomalyAlert]) -> None:
        """Stop detecting requests that just produced an anomaly."""
        for alert in alerts:
            if alert.req_id:
                self._stopped_req_ids.add(alert.req_id)

    def check_after_spec(
        self,
        sampled_tokens: Any,
        accepted_token_nums: Any,
    ) -> list[AnomalyAlert]:
        """Record accepted speculative tokens, then run spec-acceptance detect."""
        if self._gated("after_spec"):
            return []
        self._record_spec_step_outputs(sampled_tokens, accepted_token_nums)
        skip = self._stopped_req_ids if self._stop_after_alert() else None
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
        """Append sample tokens to IO buffer; run token_logprob then output_substring.

        ``sampled_token_ids`` is appended to the cumulative IO buffer once here
        for **all** batch rows (including already-alerted ones) so later
        manual-dump / report snapshots stay complete. Detection then skips
        ``stop_after_alert`` reqs by id — never by row-subsetting — so
        ``req_idx`` stays aligned with ``input_batch`` (filters / dump related).

        ``OutputSubstringDetector.check_all`` is called with ``sampled_token_ids=None``
        so it reads the buffer instead of re-appending (avoids double count).

        With ``detector.stop_after_alert`` (default true) a request keeps being
        checked on every step until it produces an anomaly; afterwards it is
        skipped entirely so the same anomaly does not write endless reports.
        """
        if self._gated("after_sample"):
            return []
        resolved_ids = req_ids
        if resolved_ids is None:
            input_batch = getattr(self._runner, "input_batch", None)
            resolved_ids = list(getattr(input_batch, "req_ids", None) or [])

        # Always accumulate IO first (same as ``check_after_spec``).
        RequestIoSnapshotManager.get().append_batch(resolved_ids, sampled_token_ids)

        skip = self._stopped_req_ids if self._stop_after_alert() else None
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
        # Substring match uses cumulative IO (already appended); pass None to avoid
        # a second append_batch inside OutputSubstringDetector.check_all.
        alerts.extend(
            self._output_substring_det.check_all(
                sampled_token_ids=None,
                req_ids=resolved_ids,
                skip_req_ids=skip,
            )
        )
        if skip is not None:
            self._mark_alerted(alerts)
        return alerts

    # ---- helpers ----------------------------------------------------------

    def _record_spec_step_outputs(
        self,
        sampled_tokens: Any,
        accepted_token_nums: Any,
    ) -> None:
        """Accumulate accepted speculative tokens into DFX cumulative output."""
        if sampled_tokens is None or accepted_token_nums is None:
            return
        input_batch = getattr(self._runner, "input_batch", None)
        req_ids = getattr(input_batch, "req_ids", None) if input_batch is not None else None
        if not req_ids:
            return
        num_reqs = len(req_ids)
        if torch.is_tensor(accepted_token_nums):
            accepted_list = accepted_token_nums[:num_reqs].tolist()
        else:
            accepted_list = [int(x) for x in accepted_token_nums[:num_reqs]]
        io_mgr = RequestIoSnapshotManager.get()
        for batch_idx, req_id in enumerate(req_ids):
            accepted_n = int(accepted_list[batch_idx])
            if accepted_n <= 0:
                continue
            try:
                row = sampled_tokens[batch_idx]
            except (IndexError, TypeError):
                continue
            if torch.is_tensor(row):
                ids = [int(x) for x in row[:accepted_n].tolist()]
            else:
                ids = [int(x) for x in list(row)[:accepted_n]]
            io_mgr.append_output(req_id, ids)
