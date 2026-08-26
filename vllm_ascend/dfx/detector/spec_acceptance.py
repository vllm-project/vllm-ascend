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

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.parallel_state import get_pp_group

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.config_backed import ConfigBackedDetector
from vllm_ascend.dfx.io_snapshot import output_token_count_for_request
from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

logger = init_logger_ascend(__name__)


class SpecAcceptanceDetector(ConfigBackedDetector):
    """Detect abnormal speculative-decoding acceptance rate / length."""

    anomaly_type = "spec_acceptance"
    section_key = "spec_acceptance"

    def __init__(
        self,
        *,
        dfx_config: DfxRuntimeConfig | None = None,
        runner: Any | None = None,
        is_related_request: Callable[[str, int | None], bool] | None = None,
    ) -> None:
        super().__init__(dfx_config=dfx_config, runner=runner, enabled=False)
        self._is_related_request = is_related_request
        # Per-req sliding window: (accepted_draft, draft_len, sampled_ids, accepted_ids)
        self._history: dict[str, deque[tuple[int, int, list[int], list[int]]]] = defaultdict(deque)
        self._window = 10
        self._low_threshold = 0.3
        self._len_low_threshold = 1.4
        self._high_threshold = 0.96
        self._len_high_threshold = 2.8
        # Throttle INFO short logs (per req) so TP0 is not flooded.
        self._short_log_ts: dict[str, float] = {}
        self._short_log_interval_s = 2.0
        # Live knobs from DFX JSON only.
        if dfx_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        self._window = int(getter("window", self._window))
        self._low_threshold = float(getter("low_threshold", self._low_threshold))
        self._len_low_threshold = float(getter("len_low_threshold", self._len_low_threshold))
        self._high_threshold = float(getter("high_threshold", self._high_threshold))
        self._len_high_threshold = float(getter("len_high_threshold", self._len_high_threshold))

    def clear_finished(self, req_id: str) -> None:
        self._history.pop(req_id, None)
        self._short_log_ts.pop(req_id, None)

    def check_all(
        self,
        sampled_tokens: torch.Tensor,
        accepted_token_nums: Any,
        skip_req_ids: set[str] | None = None,
    ) -> list[AnomalyAlert]:
        """Batch entry: return alerts for the model runner to hand to Dumper.

        Note: this detector does **not** accumulate accepted output tokens into
        the DFX cumulative IO buffer. IO is appended once in
        ``DetectorManager.check_after_sample`` from validated sampled ids
        (avoids MTP double-count when both hooks run).

        ``skip_req_ids`` (optional): requests to skip (e.g. already alerted under
        ``stop_after_alert``). Batch index alignment is preserved for the rest.
        """
        if not self._precheck():
            runner = self._runner
            if int(getattr(runner, "tp_rank", 0) if runner is not None else 0) == 0:
                logger.info_once("[Anomaly spec short] skip: detector.spec_acceptance.enabled=false in live DFX config")
            return []
        runner = self._runner
        if runner is None:
            return []
        # Spec check needs speculative decoding, not only MambaSpec
        # (``need_accepted_tokens``). Plain MTP / Eagle also produce accept stats.
        if getattr(runner, "speculative_config", None) is None:
            if int(getattr(runner, "tp_rank", 0)) == 0:
                logger.info_once("[Anomaly spec short] skip: speculative_config is None")
            return []
        input_batch = getattr(runner, "input_batch", None)
        if input_batch is None or not getattr(input_batch, "req_ids", None):
            return []

        num_reqs = len(input_batch.req_ids)
        if torch.is_tensor(accepted_token_nums):
            accepted_list = accepted_token_nums[:num_reqs].tolist()
        else:
            accepted_list = [int(x) for x in accepted_token_nums[:num_reqs]]

        sampled_token_rows = sampled_tokens[:num_reqs]
        requests = getattr(runner, "requests", None)
        draft_lens = getattr(input_batch, "num_draft_tokens_per_req", None)

        alerts: list[AnomalyAlert] = []
        for batch_idx, req_id in enumerate(input_batch.req_ids):
            if skip_req_ids and req_id in skip_req_ids:
                continue
            # B9 fix: short batch edge — accepted_list may be shorter than
            # req_ids under partial-batch / MTP edge cases. IndexError here
            # would abort the whole batch's spec check.
            if batch_idx >= len(accepted_list):
                continue
            accepted_token_num = int(accepted_list[batch_idx])
            if batch_idx >= len(sampled_token_rows):
                continue
            sampled_ids = sampled_token_rows[batch_idx]

            if requests is not None and req_id in requests:
                req_state = requests[req_id]
            else:
                draft_len = int(draft_lens[batch_idx]) if draft_lens is not None else 0
                req_state = SimpleNamespace(
                    prev_num_draft_len=draft_len,
                    prompt_token_ids=None,
                    output_token_ids=None,
                )

            alert = self.check_one(
                req_idx=batch_idx,
                req_id=req_id,
                req_state=req_state,
                accepted_token_num=accepted_token_num,
                sampled_ids=sampled_ids,
            )
            if alert is not None:
                alerts.append(alert)
        return alerts

    def check_one(
        self,
        req_idx: int,
        req_id: str,
        req_state: Any,
        accepted_token_num: int,
        sampled_ids: list[int] | torch.Tensor | None = None,
    ) -> AnomalyAlert | None:
        if not req_id:
            return None
        if not get_pp_group().is_last_rank:
            return None
        prompt_raw = getattr(req_state, "prompt_token_ids", None)
        prompt_ids = self._normalize_token_ids(prompt_raw) if prompt_raw is not None else None
        if not self._passes_input_filter(req_id, req_idx, prompt_token_ids=prompt_ids):
            return None
        runner = self._runner
        log_leader = int(getattr(runner, "tp_rank", 0) if runner is not None else 0) == 0
        draft_len = getattr(req_state, "prev_num_draft_len", 0) or 0
        sampled_norm = self._normalize_token_ids(sampled_ids)
        if draft_len <= 0:
            # Fallback when prev_num_draft_len was not populated (common on
            # non-hybrid MTP): treat last dim of sampled row as draft+bonus.
            draft_len = max(0, len(sampled_norm) - 1)
        if draft_len <= 0:
            if log_leader:
                logger.info_once(
                    "[Anomaly spec short] req_id=%s skip: draft_len=0 sampled_len=%d",
                    req_id,
                    len(sampled_norm),
                )
            return None
        # Related-request filter is for dump arming only; always emit short logs
        # for local batch rows so incorrect filters are visible.
        related_ok = True
        if self._is_related_request is not None and not self._is_related_request(req_id, req_idx):
            related_ok = False
        accepted_draft_tokens = max(0, accepted_token_num - 1)
        accepted_norm = sampled_norm[:accepted_token_num] if accepted_token_num > 0 else []
        history = self._history[req_id]
        prev_hist_len = len(history)
        history.append((accepted_draft_tokens, draft_len, sampled_norm, accepted_norm))
        while len(history) > self._window:
            history.popleft()

        output_token_count = output_token_count_for_request(self._runner, req_id, req_idx)
        prompt_token_count = len(prompt_ids) if prompt_ids is not None else 0
        accepted_sum = sum(accepted for accepted, _, _, _ in history)
        draft_sum = sum(draft for _, draft, _, _ in history)
        acceptance_rate = accepted_sum / draft_sum if draft_sum > 0 else 0.0
        acceptance_len = accepted_sum / len(history) if history else 0.0

        window_ready = len(history) >= self._window
        just_filled = prev_hist_len < self._window <= len(history)
        should_alert = False
        if window_ready:
            should_alert = bool(
                (acceptance_rate < self._low_threshold and acceptance_len < self._len_low_threshold)
                or (acceptance_rate > self._high_threshold and acceptance_len > self._len_high_threshold)
            )

        # INFO only on a real alert candidate; window fill / related miss / routine
        # steps stay DEBUG (report + on_alert_armed still emit INFO on action).
        if log_leader:
            short_msg = (
                "[Anomaly spec short] req_id=%s draft_len=%d "
                "accepted_count=%d accepted_draft_count=%d "
                "accept_rate=%.4f accept_len=%.4f window=%d/%d accepted=%d drafted=%d "
                "prompt_tokens=%d output_tokens=%d "
                "low=(%.2f,%.2f) high=(%.2f,%.2f) related=%s alert=%s"
            )
            short_args = (
                req_id,
                draft_len,
                accepted_token_num,
                accepted_draft_tokens,
                acceptance_rate,
                acceptance_len,
                len(history),
                self._window,
                accepted_sum,
                draft_sum,
                prompt_token_count,
                output_token_count,
                self._low_threshold,
                self._len_low_threshold,
                self._high_threshold,
                self._len_high_threshold,
                related_ok,
                should_alert and related_ok,
            )
            if should_alert and related_ok:
                logger.info(short_msg, *short_args)
            elif just_filled or not related_ok:
                logger.debug(short_msg, *short_args)
            else:
                now = time.time()
                last = self._short_log_ts.get(req_id, 0.0)
                if now - last >= self._short_log_interval_s:
                    self._short_log_ts[req_id] = now
                    logger.debug(short_msg, *short_args)

        if not related_ok or not window_ready or not should_alert:
            return None

        # Detector detail for report: acceptance metrics + per-step sample stats.
        # Full prompt/output ids attached by RequestIoSnapshotManager at report time.
        window_steps = [
            {
                "accepted_draft": int(accepted_draft),
                "draft_len": int(draft),
                "sampled_count": len(step_sampled),
                "accepted_count": len(step_accepted),
                "step_accept_rate": (float(accepted_draft) / float(draft)) if draft > 0 else 0.0,
            }
            for accepted_draft, draft, step_sampled, step_accepted in history
        ]
        detail: dict[str, Any] = {
            "acceptance_rate": acceptance_rate,
            "acceptance_len": acceptance_len,
            "accepted_sum": accepted_sum,
            "draft_sum": draft_sum,
            "window": len(history),
            "window_size": self._window,
            # Current step sample / accept.
            "draft_len": int(draft_len),
            "accepted_count": int(accepted_token_num),
            "accepted_draft_count": int(accepted_draft_tokens),
            "sampled_count": len(sampled_norm),
            "thresholds": {
                "low_rate": self._low_threshold,
                "low_len": self._len_low_threshold,
                "high_rate": self._high_threshold,
                "high_len": self._len_high_threshold,
            },
            "window_steps": window_steps,
            # Detection-window evidence (not full request I/O).
            "window_sampled_token_ids": [list(step_sampled) for _, _, step_sampled, _ in history],
            "window_accepted_token_ids": [list(step_accepted) for _, _, _, step_accepted in history],
            "current_sampled_token_ids": list(sampled_norm),
            "current_accepted_token_ids": list(accepted_norm),
            # prompt/output counts attached at report time by IoSnapshot merge.
        }
        return AnomalyAlert(
            anomaly_type=self.anomaly_type,
            req_id=req_id,
            req_idx=req_idx,
            is_ill=True,
            ill_type=0,
            detail=detail,
            skip_related_check=False,
            log_context={
                "sampled_ids": sampled_norm,
                "accepted_token_num": accepted_token_num,
                "prompt_token_count": prompt_token_count,
                "output_token_count": output_token_count,
                "window_sampled_steps": len(history),
                "window_accepted_steps": len(history),
            },
        )

    def on_alert_armed(self, alert: AnomalyAlert) -> None:
        ctx = alert.log_context
        if not ctx:
            return
        sampled_ids = ctx.get("sampled_ids") or []
        accepted_token_num = int(ctx.get("accepted_token_num") or 0)
        logger.info(
            "[Anomaly spec] req_id=%s sampled_len=%d accepted_len=%d "
            "window_sampled_steps=%d window_accepted_steps=%d "
            "prompt_token_count=%d output_token_count=%d",
            alert.req_id,
            len(sampled_ids),
            accepted_token_num if accepted_token_num > 0 else 0,
            int(ctx.get("window_sampled_steps") or 0),
            int(ctx.get("window_accepted_steps") or 0),
            int(ctx.get("prompt_token_count") or 0),
            int(ctx.get("output_token_count") or 0),
        )
