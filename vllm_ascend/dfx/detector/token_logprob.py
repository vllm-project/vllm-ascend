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

from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.config_backed import ConfigBackedDetector
from vllm_ascend.dfx.io_snapshot import output_token_count_for_request, prompt_token_count_for_request
from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

logger = init_logger_ascend(__name__)


class TokenLogprobDetector(ConfigBackedDetector):
    """Detect ill-formed token/logprob windows via msprobe ILLDetector."""

    anomaly_type = "token_logprob"
    section_key = "token_logprob"

    def __init__(
        self,
        *,
        dfx_config: DfxRuntimeConfig | None = None,
        runner: Any | None = None,
    ) -> None:
        super().__init__(dfx_config=dfx_config, runner=runner, enabled=False)
        self._window = 64
        self._stride = 32
        self._topk = 20
        self._ill_window_thresh = {1: 1, 2: 1, 3: 2, 4: 1}
        self._buf: dict[str, deque[tuple[int, dict[int, float]]]] = {}
        self._since_check: dict[str, int] = defaultdict(int)
        self._checked: set[str] = set()
        self._ill_window_hits: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._ill_detector: Any | None = None
        self._ill_detector_init_failed = False
        # Live knobs from DFX JSON only.
        if dfx_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        self._window = int(getter("window", self._window))
        self._stride = int(getter("stride", self._stride))
        self._topk = int(getter("topk", self._topk))
        self._ill_window_thresh = {
            1: int(getter("ill_rare_window_thresh", self._ill_window_thresh[1])),
            2: int(getter("ill_garbled_window_thresh", self._ill_window_thresh[2])),
            3: int(getter("ill_repet_window_thresh", self._ill_window_thresh[3])),
            4: int(getter("ill_nan_window_thresh", self._ill_window_thresh[4])),
        }

    def refresh_from_config(self) -> None:
        """Pull knobs; if enabled, ensure ILLDetector or force the detector off.

        Clears a prior ``_ill_detector_init_failed`` so a later msprobe install
        + ``enabled=true`` hot-reload can re-init without restarting the worker.
        """
        super().refresh_from_config()
        if not self._enabled:
            return
        if self._ill_detector is not None:
            return
        # Allow retry after a previous ImportError / init failure.
        self._ill_detector_init_failed = False
        if self._get_ill_detector() is not None:
            return
        self._enabled = False
        cfg = self._dfx_config
        if cfg is not None:
            cfg.disable_detector_unavailable(
                "token_logprob",
                reason=(
                    "msprobe response_anomaly / ILLDetector unavailable "
                    "(install msprobe, then set detector.token_logprob.enabled=true again)"
                ),
            )

    @property
    def topk(self) -> int:
        return self._topk

    def clear_finished(self, req_id: str) -> None:
        self._buf.pop(req_id, None)
        self._since_check.pop(req_id, None)
        self._checked.discard(req_id)
        self._ill_window_hits.pop(req_id, None)

    def check_all(
        self,
        sampled_token_ids: list[list[int]] | None,
        logprobs_lists: Any | None,
        req_ids: list[str] | None = None,
        skip_req_ids: set[str] | None = None,
    ) -> list[AnomalyAlert]:
        """Batch entry: return alerts for the model runner to hand to Dumper.

        ``skip_req_ids``: requests to skip (e.g. already alerted under
        ``stop_after_alert``). Batch index alignment is preserved for the rest.
        """
        runner = self._runner
        log_leader = int(getattr(runner, "tp_rank", 0) if runner is not None else 0) == 0
        if not self._precheck():
            if log_leader:
                logger.info_once(
                    "[Anomaly token_logprob short] skip: detector.token_logprob.enabled=false "
                    "in live DFX config (edit JSON + dfx_config_reload_interval>0, or set "
                    "true before start; look for '[DFX runtime_config] updated')"
                )
            return []
        if sampled_token_ids is None:
            if log_leader:
                logger.info_once("[Anomaly token_logprob short] skip: sampled_token_ids is None")
            return []
        if logprobs_lists is None:
            if log_leader:
                logger.info_once(
                    "[Anomaly token_logprob short] skip: no logprobs (enable check should force topk=%d before sample)",
                    self._topk,
                )
            return []

        if req_ids is None:
            input_batch = getattr(runner, "input_batch", None) if runner is not None else None
            req_ids = getattr(input_batch, "req_ids", None) if input_batch is not None else None
        if not req_ids:
            return []

        detector = self._get_ill_detector()
        if detector is None:
            if log_leader:
                logger.info_once("[Anomaly token_logprob short] skip: ILLDetector unavailable")
            return []

        model_config = self._model_config_for_detector()
        alerts: list[AnomalyAlert] = []
        for batch_idx, req_id in enumerate(req_ids):
            if skip_req_ids and req_id in skip_req_ids:
                continue
            if batch_idx >= len(sampled_token_ids):
                break
            if not self._passes_input_filter(req_id, batch_idx):
                continue
            token_ids = sampled_token_ids[batch_idx]
            if not token_ids:
                continue
            topk_rows = self._extract_req_topk_logprobs(logprobs_lists, batch_idx, len(token_ids))
            if topk_rows is None:
                if log_leader:
                    logger.debug(
                        "[Anomaly token_logprob short] req_id=%s skip: extract topk failed num_tokens=%d",
                        req_id,
                        len(token_ids),
                    )
                continue
            alert = self.check_one(
                req_idx=batch_idx,
                req_id=req_id,
                token_ids=token_ids,
                topk_logprobs=topk_rows,
                model_config=model_config,
                detector=detector,
                log_leader=log_leader,
            )
            if alert is not None:
                alerts.append(alert)
        return alerts

    def check_one(
        self,
        req_idx: int,
        req_id: str,
        token_ids: list[int],
        topk_logprobs: list[dict[int, float]],
        model_config: Any,
        detector: Any,
        log_leader: bool = False,
    ) -> AnomalyAlert | None:
        if not token_ids or not topk_logprobs:
            return None
        n = min(len(token_ids), len(topk_logprobs))
        buf = self._buf.get(req_id)
        if buf is None:
            buf = deque(maxlen=self._window)
            self._buf[req_id] = buf

        for i in range(n):
            buf.append((int(token_ids[i]), topk_logprobs[i]))
        self._since_check[req_id] += n

        buf_len = len(buf)
        since = self._since_check[req_id]
        already_checked = req_id in self._checked
        window_ready = buf_len >= self._window
        due = window_ready and (not already_checked or since >= self._stride)

        if not due:
            if log_leader:
                # Filling/stride progress every step stalls TP0 → peer hang.
                logger.debug(
                    "[Anomaly token_logprob short] req_id=%s buf=%d/%d since=%d stride=%d new=%d alert=False reason=%s",
                    req_id,
                    buf_len,
                    self._window,
                    since,
                    self._stride,
                    n,
                    "filling" if not window_ready else "stride",
                )
            return None

        self._since_check[req_id] = 0
        self._checked.add(req_id)
        tokens = [tid for tid, _ in buf]
        topk_dicts = [lp for _, lp in buf]

        try:
            result = detector.detector(topk_dicts, tokens, model_config)
        except Exception as e:
            logger.error(
                "[Anomaly token_logprob] detector failed req_id=%s error=%s",
                req_id,
                e,
            )
            return None

        alert = AnomalyAlert.from_ill_result(
            req_id=req_id,
            result=result,
            req_idx=req_idx,
            skip_related_check=True,
        )
        if alert is None:
            if log_leader:
                logger.debug(
                    "[Anomaly token_logprob short] req_id=%s buf=%d/%d since=0 stride=%d "
                    "new=%d alert=False reason=not_ill",
                    req_id,
                    buf_len,
                    self._window,
                    self._stride,
                    n,
                )
            return None

        thresh = self._ill_window_thresh.get(alert.ill_type)
        if thresh is None:
            logger.warning(
                "[Anomaly token_logprob] unknown ill_type=%d req_id=%s",
                alert.ill_type,
                req_id,
            )
            return None
        hits = self._ill_window_hits[req_id]
        hits[alert.ill_type] += 1
        hit_count = hits[alert.ill_type]
        should_alert = hit_count >= thresh
        if log_leader:
            log_fn = logger.info if should_alert else logger.debug
            log_fn(
                "[Anomaly token_logprob short] req_id=%s buf=%d/%d since=0 stride=%d "
                "new=%d ill_type=%d hits=%d/%d alert=%s",
                req_id,
                buf_len,
                self._window,
                self._stride,
                n,
                alert.ill_type,
                hit_count,
                thresh,
                should_alert,
            )
        if not should_alert:
            return None
        # Counts for logs / on_alert; report I/O counts come from IoSnapshot merge.
        output_token_count = output_token_count_for_request(self._runner, req_id, req_idx)
        prompt_token_count = prompt_token_count_for_request(self._runner, req_id, req_idx)
        alert.detail = {
            "ill_type": alert.ill_type,
            "hits": hit_count,
            "thresh": thresh,
            "window": len(tokens),
            # Detection-window evidence (not full request I/O).
            "window_token_ids": list(tokens),
        }
        alert.log_context = {
            "window_len": len(tokens),
            "prompt_token_count": prompt_token_count,
            "output_token_count": output_token_count,
            "ill_type": alert.ill_type,
            "hits": hit_count,
            "thresh": thresh,
        }
        return alert

    def on_alert_armed(self, alert: AnomalyAlert) -> None:
        ctx = alert.log_context
        if not ctx:
            return
        logger.info(
            "[Anomaly token_logprob] req_id=%s ill_type=%s hits=%s/%s "
            "window_len=%d prompt_token_count=%d output_token_count=%d",
            alert.req_id,
            ctx.get("ill_type"),
            ctx.get("hits"),
            ctx.get("thresh"),
            int(ctx.get("window_len") or 0),
            int(ctx.get("prompt_token_count") or 0),
            int(ctx.get("output_token_count") or 0),
        )

    def _get_ill_detector(self) -> Any | None:
        if self._ill_detector is not None:
            return self._ill_detector
        if self._ill_detector_init_failed:
            return None
        try:
            import msprobe.response_anomaly as response_anomaly
            from msprobe.response_anomaly.detector import ILLDetector

            base = Path(response_anomaly.__file__).resolve().parent
            detector = ILLDetector(
                str(base / "configs" / "config.yaml"),
                str(base / "configs" / "mtype_config.json"),
                str(base / "token2category"),
            )
            detector.window_size = self._window
            detector.stride = self._window
            detector.garbled_window_thresh = 0
            detector.single_window_thresh = 0
            detector.multi_window_thresh = 0
            self._ill_detector = detector
            logger.info_once(
                "[Anomaly token_logprob] ILLDetector ready window=%d stride=%d topk=%d",
                self._window,
                self._stride,
                self._topk,
            )
            return self._ill_detector
        except Exception as e:
            self._ill_detector_init_failed = True
            logger.error("[Anomaly token_logprob] failed to init ILLDetector: %s", e)
            return None

    def _model_config_for_detector(self) -> dict[str, str]:
        runner = self._runner
        model_config = None
        if runner is not None:
            vllm_config = getattr(runner, "vllm_config", None)
            model_config = getattr(vllm_config, "model_config", None)
        raw_name = ""
        if model_config is not None:
            raw_name = str(getattr(model_config, "model", None) or getattr(model_config, "model_id", "") or "")
        return {"model_name": Path(raw_name).name if raw_name else ""}

    def _extract_req_topk_logprobs(
        self,
        logprobs_lists: Any,
        req_idx: int,
        num_tokens: int,
    ) -> list[dict[int, float]] | None:
        try:
            token_ids_arr = logprobs_lists.logprob_token_ids
            logprobs_arr = logprobs_lists.logprobs
            cu = getattr(logprobs_lists, "cu_num_generated_tokens", None)
            if cu is not None:
                start = cu[req_idx]
                end = cu[req_idx + 1] if req_idx + 1 < len(cu) else start + num_tokens
            else:
                if num_tokens == 1:
                    start = req_idx
                    end = req_idx + 1
                else:
                    start = req_idx * num_tokens
                    end = start + num_tokens
            end = min(end, start + num_tokens, len(token_ids_arr))
            if end <= start:
                return None
            rows: list[dict[int, float]] = []
            for row_i in range(start, end):
                rows.append(
                    self._row_to_topk_dict(
                        token_ids_arr[row_i],
                        logprobs_arr[row_i],
                        self._topk,
                    )
                )
            return rows
        except Exception as e:
            logger.error(
                "[Anomaly token_logprob] extract logprobs failed req_idx=%d error=%s",
                req_idx,
                e,
            )
            return None

    @staticmethod
    def _row_to_topk_dict(token_ids_row: Any, logprobs_row: Any, topk: int) -> dict[int, float]:
        tids = token_ids_row.tolist() if hasattr(token_ids_row, "tolist") else list(token_ids_row)
        lps = logprobs_row.tolist() if hasattr(logprobs_row, "tolist") else list(logprobs_row)
        pairs = []
        for tid, lp in zip(tids, lps):
            tid_i = int(tid)
            if tid_i < 0:
                continue
            pairs.append((tid_i, float(lp)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        out: dict[int, float] = {}
        for tid, lp in pairs[:topk]:
            out[tid] = lp
        return out
