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

"""Selected-token logprob detector for garbled / nonsense text.

Computes logprobs on the Guard hot path from stashed pre-sample logits +
sampled token ids. Does **not** require ``SamplingParams.logprobs`` and never
injects logprobs into the user-facing API response.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_config.schema import ConfigField, DetectorSchema
from vllm_ascend.observability.runtime_guard.detector.base import ConfigBackedDetector, resolve_batch_req_ids
from vllm_ascend.observability.runtime_guard.rank_gate import runner_tp_rank
from vllm_ascend.observability.runtime_guard.state import ILL_TYPE_NONE, Incident

if TYPE_CHECKING:
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)


def selected_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Log-prob of each selected token under the row's softmax.

    ``logits``: ``[num_rows, vocab]`` (device or host).
    ``token_ids``: ``[num_rows]`` long (same device preferred).

    Prefers the Ascend fused gather kernel when available; falls back to a
    portable ``log_softmax`` + ``gather`` path for CPU / UT.
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(token_ids, torch.Tensor):
        raise TypeError("logits and token_ids must be torch.Tensor")
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D [rows, vocab], got shape={tuple(logits.shape)}")
    ids = token_ids.reshape(-1).to(dtype=torch.long, device=logits.device)
    if ids.numel() != logits.shape[0]:
        raise ValueError(f"token_ids length {ids.numel()} != logits rows {logits.shape[0]}")

    try:
        from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs

        gathered = compute_token_logprobs(logits, ids.unsqueeze(-1))
        return gathered.squeeze(-1)
    except Exception:
        # CPU UT / missing Triton path: full-row log_softmax is fine at small
        # vocab sizes used in tests; production NPU uses the fused kernel above.
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return log_probs.gather(1, ids.unsqueeze(-1)).squeeze(1)


def _flatten_sampled_token_ids(sampled_token_ids: Any) -> list[list[int]]:
    """Normalize sampler / postprocess sampled ids to ``list[list[int]]``."""
    if sampled_token_ids is None:
        return []
    if isinstance(sampled_token_ids, torch.Tensor):
        if sampled_token_ids.numel() == 0:
            return []
        t = sampled_token_ids.detach()
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        rows = t.tolist()
        out: list[list[int]] = []
        for row in rows:
            if isinstance(row, list):
                out.append([int(x) for x in row if int(x) >= 0])
            else:
                tid = int(row)
                out.append([tid] if tid >= 0 else [])
        return out
    if isinstance(sampled_token_ids, (list, tuple)):
        out = []
        for row in sampled_token_ids:
            if row is None:
                out.append([])
            elif isinstance(row, (list, tuple)):
                out.append([int(x) for x in row if int(x) >= 0])
            elif isinstance(row, torch.Tensor):
                out.append([int(x) for x in row.detach().tolist() if int(x) >= 0])
            else:
                tid = int(row)
                out.append([tid] if tid >= 0 else [])
        return out
    return []


@dataclass
class TokenLogprobState:
    """Per-request consecutive low-logprob streak."""

    consecutive_hits: int = 0
    tokens_seen: int = 0
    last_logprob: float | None = None
    last_token_id: int | None = None
    recent: list[dict[str, float | int]] = field(default_factory=list)


class TokenLogprobDetector(ConfigBackedDetector):
    """Alert when selected-token logprobs stay abnormally low (garbled text).

    Guard computes selected-token logprobs from stashed pre-sample logits even
    when the client did not request ``SamplingParams.logprobs``. Results are
    detection-only and are never attached to the OpenAI response.
    """

    incident_type = "token_logprob"
    section_key = "token_logprob"
    schema = DetectorSchema(
        section_key="token_logprob",
        stage="after_sample",
        help=(
            "Low selected-token logprob (garbled / nonsense). "
            "Guard-side compute; API SamplingParams.logprobs not required."
        ),
        fields=(
            ConfigField("enabled", False, "bool", help="Master switch"),
            ConfigField(
                "min_logprob",
                -15.0,
                "float",
                help="Alert when a selected-token logprob is strictly below this",
            ),
            ConfigField(
                "min_tokens",
                1,
                "int",
                min_value=0,
                help="Minimum selected tokens scored before alerting (0 = no warmup)",
            ),
            ConfigField(
                "consecutive_hits",
                3,
                "int",
                min_value=1,
                help="Required consecutive below-threshold tokens",
            ),
        ),
    )

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig | None = None,
        runner: Any | None = None,
    ) -> None:
        super().__init__(runtime_config=runtime_config, runner=runner, enabled=False)
        defaults = self.schema.defaults_dict()
        self._min_logprob = float(defaults["min_logprob"])
        self._min_tokens = int(defaults["min_tokens"])
        self._consecutive_hits_thresh = int(defaults["consecutive_hits"])
        self._states: dict[str, TokenLogprobState] = {}
        self._alerted: set[str] = set()
        if runtime_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        self._min_logprob = float(getter("min_logprob", self._min_logprob))
        self._min_tokens = max(0, int(getter("min_tokens", self._min_tokens)))
        self._consecutive_hits_thresh = max(1, int(getter("consecutive_hits", self._consecutive_hits_thresh)))

    def clear_finished(self, req_id: str) -> None:
        self._states.pop(req_id, None)
        self._alerted.discard(req_id)

    def check_from_logits(
        self,
        *,
        logits: torch.Tensor,
        sampled_token_ids: Any,
        req_ids: list[str] | None = None,
        skip_req_ids: set[str] | None = None,
    ) -> list[Incident]:
        """Score selected-token logprobs from stashed logits + sampled ids."""
        if not self._precheck():
            return []
        if not isinstance(logits, torch.Tensor) or logits.numel() == 0:
            return []

        runner = self._runner
        log_leader = runner_tp_rank(runner) == 0
        req_ids = resolve_batch_req_ids(runner, req_ids)
        rows = _flatten_sampled_token_ids(sampled_token_ids)
        if not req_ids or not rows:
            return []

        # Align rows to logits: decode is usually 1 logit row per req. Spec /
        # multi-token steps may emit multiple ids per req — score the last id
        # against the matching logits row when shapes allow.
        num_logit_rows = int(logits.shape[0])
        n = min(len(req_ids), len(rows), num_logit_rows)
        if n <= 0:
            return []

        token_ids_for_rows: list[int] = []
        row_to_req: list[int] = []
        for i in range(n):
            ids = rows[i]
            if not ids:
                continue
            token_ids_for_rows.append(int(ids[-1]))
            row_to_req.append(i)
        if not token_ids_for_rows:
            return []

        try:
            row_idx = torch.tensor(row_to_req, dtype=torch.long, device=logits.device)
            logits_rows = logits.index_select(0, row_idx)
            tid_t = torch.tensor(token_ids_for_rows, dtype=torch.long, device=logits.device)
            lp_t = selected_token_logprobs(logits_rows, tid_t)
            # Host list for per-req state machine (small: one value per scored row).
            lp_list = [float(x) for x in lp_t.detach().float().cpu().tolist()]
        except Exception as exc:
            logger.warning("[runtime_guard: token_logprob] compute failed: %s", exc)
            return []

        skip = skip_req_ids or set()
        alerts: list[Incident] = []
        for scored_i, batch_idx in enumerate(row_to_req):
            req_id = req_ids[batch_idx]
            if not req_id or req_id in self._alerted:
                continue
            if req_id in skip:
                continue
            alert = self._fold_one(
                req_id=req_id,
                req_idx=batch_idx,
                token_id=token_ids_for_rows[scored_i],
                logprob=lp_list[scored_i],
                log_leader=log_leader,
            )
            if alert is not None:
                alerts.append(alert)
        return alerts

    def _fold_one(
        self,
        *,
        req_id: str,
        req_idx: int,
        token_id: int,
        logprob: float,
        log_leader: bool,
    ) -> Incident | None:
        state = self._states.get(req_id)
        if state is None:
            state = TokenLogprobState()
            self._states[req_id] = state

        state.tokens_seen += 1
        state.last_logprob = logprob
        state.last_token_id = token_id
        state.recent.append({"token_id": token_id, "logprob": logprob})
        if len(state.recent) > 16:
            state.recent = state.recent[-16:]

        below = logprob < self._min_logprob
        if below:
            state.consecutive_hits += 1
        else:
            state.consecutive_hits = 0

        warmed = state.tokens_seen >= self._min_tokens
        if not (warmed and state.consecutive_hits >= self._consecutive_hits_thresh):
            return None

        self._alerted.add(req_id)
        detail = {
            "violation": "low_selected_logprob",
            "logprob": logprob,
            "min_logprob": self._min_logprob,
            "token_id": token_id,
            "tokens_seen": state.tokens_seen,
            "consecutive_hits": state.consecutive_hits,
            "recent": list(state.recent),
            "source": "guard_selected_logprob",
            "api_logprobs_required": False,
        }
        if log_leader:
            logger.error(
                "[runtime_guard: token_logprob] req=%s logprob=%.4f min_logprob=%.4f "
                "token_id=%s consecutive=%s tokens_seen=%s",
                req_id,
                logprob,
                self._min_logprob,
                token_id,
                state.consecutive_hits,
                state.tokens_seen,
            )
        return Incident(
            incident_type=self.incident_type,
            req_id=req_id,
            req_idx=req_idx,
            is_ill=True,
            # No dedicated msprobe ILL code for low-logprob garbage yet.
            ill_type=ILL_TYPE_NONE,
            detail=detail,
        )

    # Kept for DetectorManager symmetry / unit tests that pass explicit rows.
    def check_all(
        self,
        *,
        logprobs: Sequence[float] | torch.Tensor,
        token_ids: Sequence[int] | torch.Tensor,
        req_ids: list[str] | None = None,
        skip_req_ids: set[str] | None = None,
    ) -> list[Incident]:
        """Host-side check when logprobs are already materialized per req."""
        if not self._precheck():
            return []
        runner = self._runner
        log_leader = runner_tp_rank(runner) == 0
        req_ids = resolve_batch_req_ids(runner, req_ids)
        if isinstance(logprobs, torch.Tensor):
            lp_list = [float(x) for x in logprobs.detach().float().cpu().tolist()]
        else:
            lp_list = [float(x) for x in logprobs]
        if isinstance(token_ids, torch.Tensor):
            tid_list = [int(x) for x in token_ids.detach().cpu().tolist()]
        else:
            tid_list = [int(x) for x in token_ids]
        n = min(len(req_ids), len(lp_list), len(tid_list))
        skip = skip_req_ids or set()
        alerts: list[Incident] = []
        for i in range(n):
            req_id = req_ids[i]
            if not req_id or req_id in self._alerted or req_id in skip:
                continue
            alert = self._fold_one(
                req_id=req_id,
                req_idx=i,
                token_id=tid_list[i],
                logprob=lp_list[i],
                log_leader=log_leader,
            )
            if alert is not None:
                alerts.append(alert)
        return alerts
