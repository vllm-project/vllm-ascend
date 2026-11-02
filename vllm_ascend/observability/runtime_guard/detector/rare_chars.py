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

"""Runtime Guard wrapper for rare_chars rc4 (visible garbled / foreign-script).

Collects ``(top1_logprob, chosen_logprob)`` via sampler-side narrow collect on
Guard-stashed pre-sample logits + sampled ids (v35 Triton
``_v35_partials`` / ``_v35_finish`` — no full logprob table / ranks). Does
**not** require ``SamplingParams.logprobs`` and never injects logprobs into the
API response. CPU rules use the vendored rc4 ``RareDetector`` (see
``detector/rare/``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_config.schema import ConfigField, DetectorSchema
from vllm_ascend.observability.runtime_guard.detector.base import ConfigBackedDetector, resolve_batch_req_ids
from vllm_ascend.observability.runtime_guard.detector.rare import CONFIG_RC4, THINK_END, RareDetector, RareRowBatch
from vllm_ascend.observability.runtime_guard.detector.rare.rare_sched import (
    RARE_CHECK_BUDGET,
    RARE_CHECK_GAP,
    interval_pick,
)
from vllm_ascend.observability.runtime_guard.rank_gate import runner_tp_rank
from vllm_ascend.observability.runtime_guard.state import ILL_TYPE_NONE, Incident

if TYPE_CHECKING:
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)


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


def device_chosen_token_ids(sampled_token_ids: torch.Tensor) -> torch.Tensor:
    """Last valid sampled id per row, kept on the sampled-id device.

    Empty rows stay ``-1``. No host copy.
    """
    ids = sampled_token_ids.detach()
    if ids.numel() == 0:
        return ids.reshape(-1).to(dtype=torch.int64)
    if ids.dim() == 1:
        return ids.to(dtype=torch.int64)
    if ids.dim() != 2:
        raise ValueError(f"sampled token ids must be 1-D or 2-D, got shape={tuple(ids.shape)}")
    if ids.shape[-1] == 1:
        return ids.reshape(-1).to(dtype=torch.int64)
    valid = ids >= 0
    lengths = valid.sum(dim=-1)
    last = (lengths - 1).clamp(min=0)
    chosen = ids.gather(1, last.unsqueeze(1)).reshape(-1).to(dtype=torch.int64)
    return torch.where(lengths > 0, chosen, torch.full_like(chosen, -1))


def pack_rare_step(cpu: Any, sampled: Any, n_req: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Token ids and logprob columns for one step, without a Python list per row.

    ``cpu`` is the pinned ``[rows, 2]`` pair tensor. ``sampled`` is the trimmed
    output ids (tensor, ndarray, or a list of per-request ids). The returned
    arrays are length ``min(n_req, rows)`` and are safe to hand to another thread.
    """
    if isinstance(cpu, torch.Tensor):
        pairs = cpu.detach()
        if pairs.device.type != "cpu":
            pairs = pairs.cpu()
        pairs_np = pairs.numpy()
    else:
        pairs_np = np.asarray(cpu)
    if pairs_np.ndim != 2 or pairs_np.shape[1] < 2:
        raise ValueError(f"rare pair tensor must be [rows, 2], got shape={getattr(pairs_np, 'shape', None)}")
    n = min(int(n_req), int(pairs_np.shape[0]))
    if n <= 0:
        empty_i = np.zeros(0, np.int64)
        empty_f = np.zeros(0, np.float64)
        return empty_i, empty_f, empty_f
    base = np.ascontiguousarray(pairs_np[:n], np.float64)
    token_ids = _last_token_ids(sampled, n)
    return token_ids, np.ascontiguousarray(base[:, 0]), np.ascontiguousarray(base[:, 1])


def _last_token_ids(sampled: Any, n: int) -> np.ndarray:
    """Last non-negative id of each of the first ``n`` rows."""
    if isinstance(sampled, torch.Tensor):
        chosen = device_chosen_token_ids(sampled)
        if chosen.device.type != "cpu":
            chosen = chosen.detach().cpu()
        return np.ascontiguousarray(chosen.numpy()[:n], np.int64)
    if isinstance(sampled, np.ndarray):
        arr = np.asarray(sampled)
        if arr.ndim == 1:
            return np.ascontiguousarray(arr[:n], np.int64)
        out = np.full(n, -1, np.int64)
        width = arr.shape[1]
        block = arr[:n]
        for i in range(n):
            val = -1
            for j in range(width):
                tok = int(block[i, j])
                if tok >= 0:
                    val = tok
            out[i] = val
        return out
    rows = list(sampled) if isinstance(sampled, (list, tuple)) else []
    out = np.full(n, -1, np.int64)
    for i in range(min(n, len(rows))):
        row = rows[i]
        if isinstance(row, (list, tuple, np.ndarray)):
            val = -1
            for tok in row:
                tok_i = int(tok)
                if tok_i >= 0:
                    val = tok_i
            out[i] = val
        else:
            out[i] = int(row)
    return out


def top1_and_chosen_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Natural-log top-1 and selected-token logprobs per row (raw logits).

    Same units as vLLM ``raw_logprobs`` (natural log). rc4 converts to log10
    internally. Prefers the Ascend V2 sampler v35 Triton path
    (``compute_top1_and_chosen_logprobs``: ``_v35_partials`` / ``_v35_finish``)
    — no full ``log_softmax`` table and no ranks. Falls back to portable
    ``log_softmax`` + max/gather for CPU / UT.
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(token_ids, torch.Tensor):
        raise TypeError("logits and token_ids must be torch.Tensor")
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D [rows, vocab], got shape={tuple(logits.shape)}")
    ids = token_ids.reshape(-1).to(dtype=torch.long, device=logits.device)
    if ids.numel() != logits.shape[0]:
        raise ValueError(f"token_ids length {ids.numel()} != logits rows {logits.shape[0]}")

    try:
        from vllm_ascend.worker.v2.sample.logprob import compute_top1_and_chosen_logprobs

        return compute_top1_and_chosen_logprobs(logits, ids)
    except Exception:
        # CPU UT / missing Triton: full-row log_softmax is fine at test vocab.
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        top1 = log_probs.max(dim=-1).values
        chosen = log_probs.gather(1, ids.unsqueeze(-1)).squeeze(1)
        return top1, chosen


def _resolve_table(table: str) -> str:
    """Shipped corpus name or absolute/relative ``*_rare.npz`` path."""
    t = (table or "qwen3").strip()
    if t.endswith(".npz"):
        if not os.path.isfile(t):
            raise FileNotFoundError(f"rare_chars table not found: {t}")
        return t
    return t  # RareDetector load() resolves shipped names under rare/tables/


def _resolve_think_end(table: str, think_end: str) -> int | None | str:
    raw = (think_end or "auto").strip().lower()
    if raw == "auto":
        return "auto"
    if raw in ("none", "null", ""):
        return None
    return int(raw)


@dataclass
class RareCharsReqState:
    """Per-request streaming RareDetector state."""

    det_state: Any = None
    tokens_seen: int = 0
    alerted: bool = False
    last_reason: str = ""
    recent: list[dict[str, float | int | str]] = field(default_factory=list)


class RareCharsDetector(ConfigBackedDetector):
    """rc4 rare_chars: visible garbled / foreign-script / bad UTF-8.

    Shadow-oriented: emits Guard ``Incident`` reports; does not abort the
    request by itself (match upstream V35_RARE=shadow semantics).
    """

    incident_type = "rare_chars"
    section_key = "rare_chars"
    schema = DetectorSchema(
        section_key="rare_chars",
        stage="after_sample",
        help=(
            "Visible garbled text (rc4 rare_chars): foreign-script fragments, "
            "illegal UTF-8, low-confidence rare tokens. Guard uses sampler-side "
            "v35 Triton narrow collect on stashed pre-sample logits "
            "for (top1, chosen) — API SamplingParams.logprobs not required."
        ),
        fields=(
            ConfigField("enabled", False, "bool", help="Master switch"),
            ConfigField(
                "table",
                "qwen3",
                "str",
                help="Shipped table name (qwen3|glm|dsv4) or path to *_rare.npz",
            ),
            ConfigField(
                "think_end",
                "auto",
                "str",
                help="'</think>' token id, 'auto' for shipped tables, or 'none' (chat)",
            ),
            ConfigField(
                "alert_monitor",
                False,
                "bool",
                help="Also alert on rc4 monitor-tier rules (looser false-positive band)",
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
        self._table = str(defaults["table"])
        self._think_end_cfg = str(defaults["think_end"])
        self._alert_monitor = bool(defaults["alert_monitor"])
        self._engine: RareDetector | None = None
        self._engine_key: tuple[str, str] | None = None
        self._states: dict[str, RareCharsReqState] = {}
        # Request-axis rc4 state. One row per live req; a step feeds every row together.
        self._row_batch: RareRowBatch | None = None
        self._row_of: dict[str, int] = {}
        self._interval_pending: dict[str, int] = {}
        self._interval_abnormal: dict[str, bool] = {}
        self._interval_closed: dict[str, bool] = {}
        self._interval_logged = False
        self._interval_capped_logged = False
        if runtime_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        self._table = str(getter("table", self._table))
        self._think_end_cfg = str(getter("think_end", self._think_end_cfg))
        self._alert_monitor = bool(getter("alert_monitor", self._alert_monitor))
        # Drop engine if table / think_end changed.
        key = (self._table, self._think_end_cfg)
        if self._engine_key != key:
            self._engine = None
            self._engine_key = None

    def clear_finished(self, req_id: str) -> None:
        row = self._row_of.pop(req_id, None)
        batch = self._row_batch
        if batch is not None and batch._native is not None and batch._id_map is not None:
            batch._native.map_del(batch._id_map, req_id)
        if row is not None and batch is not None:
            if row < len(batch._req_at):
                batch._req_at[row] = None
            batch.release(row)
        self._states.pop(req_id, None)
        self._interval_pending.pop(req_id, None)
        self._interval_abnormal.pop(req_id, None)
        self._interval_closed.pop(req_id, None)

    def take_interval_batch(
        self,
        req_ids: Sequence[str],
        token_ids: Sequence[int],
        top1: Sequence[float],
        chosen: Sequence[float],
    ) -> tuple[list[str], list[int], list[float], list[float]]:
        """Keep at most ``RARE_CHECK_BUDGET`` requests for this step."""
        picked = interval_pick(
            req_ids,
            self._interval_pending,
            self._interval_abnormal,
            self._interval_closed,
            gap=RARE_CHECK_GAP,
            budget=RARE_CHECK_BUDGET,
        )
        if picked and not self._interval_logged:
            logger.info(
                "[runtime_guard: rare_chars] interval m=%s n=%s picked=%s",
                RARE_CHECK_BUDGET,
                RARE_CHECK_GAP,
                len(picked),
            )
            self._interval_logged = True
        if len(picked) >= RARE_CHECK_BUDGET and not self._interval_capped_logged:
            logger.info(
                "[runtime_guard: rare_chars] interval capped m=%s",
                RARE_CHECK_BUDGET,
            )
            self._interval_capped_logged = True
        return (
            [req_ids[i] for i in picked],
            [int(token_ids[i]) for i in picked],
            [float(top1[i]) for i in picked],
            [float(chosen[i]) for i in picked],
        )

    def note_interval_result(self, req_id: str, abnormal: bool, closed: bool) -> None:
        if not req_id:
            return
        self._interval_abnormal[req_id] = bool(abnormal)
        if closed:
            self._interval_closed[req_id] = True

    def _ensure_engine(self) -> RareDetector:
        key = (self._table, self._think_end_cfg)
        if self._engine is not None and self._engine_key == key:
            return self._engine
        corpus = _resolve_table(self._table)
        think = _resolve_think_end(corpus, self._think_end_cfg)
        if think == "auto" and not (isinstance(corpus, str) and corpus in THINK_END):
            # Path-based table: default to chat mode (no think boundary).
            think = None
        with open(CONFIG_RC4, encoding="utf-8") as f:
            cfg = json.load(f)
        self._engine = RareDetector(corpus, cfg, think_end=think)
        self._engine_key = key
        self._row_batch = RareRowBatch(self._engine)
        self._row_of.clear()
        logger.info(
            "[runtime_guard: rare_chars] loaded rc4 table=%s think_end=%s native=%s",
            corpus,
            think,
            self._row_batch._native is not None,
        )
        return self._engine

    def check_from_logits(
        self,
        *,
        logits: torch.Tensor | None,
        sampled_token_ids: Any,
        req_ids: list[str] | None = None,
        skip_req_ids: set[str] | None = None,
        top1_logprobs: Sequence[float] | torch.Tensor | None = None,
        chosen_logprobs: Sequence[float] | torch.Tensor | None = None,
    ) -> list[Incident]:
        """Score (top1, chosen) from stashed logits (or precomputed) and feed rc4.

        When ``top1_logprobs`` / ``chosen_logprobs`` are provided (sampler-side
        narrow collect already ran), skips the NPU collect and only feeds CPU
        rules. Otherwise runs fused max+LSE+gather on ``logits``.
        """
        if not self._precheck():
            return []

        has_precomputed = top1_logprobs is not None and chosen_logprobs is not None
        if not has_precomputed and (not isinstance(logits, torch.Tensor) or logits.numel() == 0):
            return []

        runner = self._runner
        log_leader = runner_tp_rank(runner) == 0
        req_ids = resolve_batch_req_ids(runner, req_ids)
        rows = _flatten_sampled_token_ids(sampled_token_ids)
        if not req_ids or not rows:
            return []

        num_rows_cap = len(req_ids)
        if isinstance(logits, torch.Tensor) and logits.numel() > 0:
            num_rows_cap = min(num_rows_cap, int(logits.shape[0]))
        n = min(num_rows_cap, len(rows))
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
            if top1_logprobs is not None and chosen_logprobs is not None:
                top1_list, chosen_list = self._coerce_pair_lists(
                    top1_logprobs, chosen_logprobs, len(token_ids_for_rows)
                )
            else:
                if not isinstance(logits, torch.Tensor) or logits.numel() == 0:
                    return []
                row_idx = torch.tensor(row_to_req, dtype=torch.long, device=logits.device)
                logits_rows = logits.index_select(0, row_idx)
                tid_t = torch.tensor(token_ids_for_rows, dtype=torch.long, device=logits.device)
                top1_t, chosen_t = top1_and_chosen_logprobs(logits_rows, tid_t)
                top1_list = [float(x) for x in top1_t.detach().float().cpu().tolist()]
                chosen_list = [float(x) for x in chosen_t.detach().float().cpu().tolist()]
        except Exception as exc:
            logger.warning("[runtime_guard: rare_chars] logprob collect failed: %s", exc)
            return []

        try:
            engine = self._ensure_engine()
        except Exception as exc:
            logger.warning("[runtime_guard: rare_chars] engine init failed: %s", exc)
            return []

        skip = skip_req_ids or set()
        keep_ids: list[str] = []
        keep_idx: list[int] = []
        keep_tok: list[int] = []
        keep_top1: list[float] = []
        keep_chosen: list[float] = []
        for scored_i, batch_idx in enumerate(row_to_req):
            req_id = req_ids[batch_idx]
            if not req_id or req_id in skip:
                continue
            keep_ids.append(req_id)
            keep_idx.append(batch_idx)
            keep_tok.append(token_ids_for_rows[scored_i])
            keep_top1.append(top1_list[scored_i])
            keep_chosen.append(chosen_list[scored_i])
        return self._feed_token_batch(
            engine=engine,
            req_ids=keep_ids,
            req_idxs=keep_idx,
            token_ids=keep_tok,
            top1=keep_top1,
            chosen=keep_chosen,
            log_leader=log_leader,
        )

    def feed_precomputed(
        self,
        *,
        req_ids: list[str],
        token_ids: list[int],
        top1_logprobs: list[float],
        chosen_logprobs: list[float],
    ) -> list[Incident]:
        """Feed rc4 from host pairs. Caller runs this off the sample return path."""
        if not self._precheck():
            return []
        n = min(len(req_ids), len(token_ids), len(top1_logprobs), len(chosen_logprobs))
        if n <= 0:
            return []
        try:
            engine = self._ensure_engine()
        except Exception as exc:
            logger.warning("[runtime_guard: rare_chars] engine init failed: %s", exc)
            return []
        log_leader = runner_tp_rank(self._runner) == 0
        batch = self._ensure_row_batch(engine)
        if batch._native is not None:
            ids = req_ids if len(req_ids) == n else req_ids[:n]
            toks = token_ids if len(token_ids) == n else token_ids[:n]
            t1 = top1_logprobs if len(top1_logprobs) == n else top1_logprobs[:n]
            ch = chosen_logprobs if len(chosen_logprobs) == n else chosen_logprobs[:n]
            return self._feed_native(
                req_ids=ids,
                req_idxs=range(n),
                token_ids=toks,
                top1=t1,
                chosen=ch,
                log_leader=log_leader,
            )
        keep_ids: list[str] = []
        keep_idx: list[int] = []
        keep_tok: list[int] = []
        keep_top1: list[float] = []
        keep_chosen: list[float] = []
        for i in range(n):
            req_id = req_ids[i]
            token_id = int(token_ids[i])
            if not req_id or token_id < 0:
                continue
            keep_ids.append(req_id)
            keep_idx.append(i)
            keep_tok.append(token_id)
            keep_top1.append(float(top1_logprobs[i]))
            keep_chosen.append(float(chosen_logprobs[i]))
        return self._feed_token_batch(
            engine=engine,
            req_ids=keep_ids,
            req_idxs=keep_idx,
            token_ids=keep_tok,
            top1=keep_top1,
            chosen=keep_chosen,
            log_leader=log_leader,
        )

    def _ensure_row_batch(self, engine: RareDetector) -> RareRowBatch:
        if self._row_batch is None or self._row_batch.det is not engine:
            self._row_batch = RareRowBatch(engine)
            self._row_of.clear()
        return self._row_batch

    def _copy_recent(self, st: RareCharsReqState, batch: RareRowBatch, row: int) -> None:
        n = int(batch._recent_len[row])
        st.tokens_seen = int(batch._tokens_seen[row])
        tok = batch._recent_tok[row, :n]
        lp1 = batch._recent_lp1[row, :n]
        lpc = batch._recent_lpc[row, :n]
        st.recent = [{"token_id": int(tok[i]), "top1": float(lp1[i]), "chosen": float(lpc[i])} for i in range(n)]

    def _feed_native(
        self,
        *,
        req_ids: Sequence[str],
        req_idxs: Sequence[int],
        token_ids: Sequence[int],
        top1: Sequence[float],
        chosen: Sequence[float],
        log_leader: bool,
    ) -> list[Incident]:
        """Resolve rows and update recent inside the GIL-free feed. Incidents only for hits."""
        batch = self._row_batch
        if batch is None or batch._native is None:
            return []
        native = batch._native
        n = min(len(req_ids), len(token_ids), len(top1), len(chosen))
        if n <= 0:
            return []
        if batch._id_map is None:
            batch._id_map = native.new_id_map()
        try:
            rows = native.lookup(batch._id_map, req_ids)
        except Exception as exc:
            logger.warning("[runtime_guard: rare_chars] row lookup failed: %s", exc)
            return []
        if int(rows.shape[0]) != n:
            rows = rows[:n]
        missing = np.flatnonzero(rows == -1)
        for raw_i in missing.tolist():
            i = int(raw_i)
            req_id = req_ids[i]
            if not req_id:
                rows[i] = np.int32(-2)
                continue
            row = batch.add()
            self._row_of[req_id] = row
            while len(batch._req_at) <= row:
                batch._req_at.append(None)
            batch._req_at[row] = req_id
            native.map_put(batch._id_map, req_id, row)
            rows[i] = np.int32(row)
            if req_id not in self._states:
                self._states[req_id] = RareCharsReqState(det_state=None)
        try:
            hit_rows = native.feed(
                batch,
                rows.astype(np.int64, copy=False),
                np.ascontiguousarray(token_ids, np.int64),
                np.ascontiguousarray(top1, np.float64),
                np.ascontiguousarray(chosen, np.float64),
            )
        except Exception as exc:
            logger.warning("[runtime_guard: rare_chars] batched feed failed: %s", exc)
            return []
        hit_rows = hit_rows or []
        idx_of: dict[str, int] = {}
        for i in range(n):
            req_id = req_ids[i]
            if req_id:
                idx_of[req_id] = int(req_idxs[i])
        alerts: list[Incident] = []
        for row in hit_rows:
            if row < 0 or row >= len(batch._req_at):
                continue
            req_id = batch._req_at[row]
            if not req_id:
                continue
            st = self._states.get(req_id)
            if st is None or st.alerted:
                continue
            self._copy_recent(st, batch, row)
            last = st.recent[-1] if st.recent else None
            alert = self._incident_from_row(
                st,
                res=batch.strict_at(row),
                mon=batch.monitor_at(row),
                req_id=req_id,
                req_idx=idx_of.get(req_id, 0),
                token_id=None if last is None else int(last["token_id"]),
                top1=None if last is None else float(last["top1"]),
                chosen=None if last is None else float(last["chosen"]),
                log_leader=log_leader,
                source="guard_rare_chars_rc4",
            )
            if alert is not None:
                batch._done[row] = True
                alerts.append(alert)
        return alerts

    def _feed_token_batch(
        self,
        *,
        engine: RareDetector,
        req_ids: list[str],
        req_idxs: list[int],
        token_ids: list[int],
        top1: list[float],
        chosen: list[float],
        log_leader: bool,
    ) -> list[Incident]:
        """One new token per request, scored together on ``RareRowBatch``."""
        batch = self._ensure_row_batch(engine)
        if batch._native is not None:
            return self._feed_native(
                req_ids=req_ids,
                req_idxs=req_idxs,
                token_ids=token_ids,
                top1=top1,
                chosen=chosen,
                log_leader=log_leader,
            )
        rows: list[int] = []
        live_ids: list[str] = []
        live_idx: list[int] = []
        live_tok: list[int] = []
        live_top1: list[float] = []
        live_chosen: list[float] = []
        states: list[RareCharsReqState] = []
        for req_id, req_idx, token_id, lp1, lpc in zip(req_ids, req_idxs, token_ids, top1, chosen):
            st = self._states.get(req_id)
            if st is None:
                st = RareCharsReqState(det_state=None)
                self._states[req_id] = st
            if st.alerted:
                continue
            row = self._row_of.get(req_id)
            if row is None:
                row = batch.add()
                self._row_of[req_id] = row
            rows.append(row)
            live_ids.append(req_id)
            live_idx.append(req_idx)
            live_tok.append(token_id)
            live_top1.append(lp1)
            live_chosen.append(lpc)
            states.append(st)
        if not rows:
            return []
        try:
            batch.feed_rows(
                np.asarray(rows, np.int64),
                np.asarray(live_tok, np.int64),
                np.asarray(live_top1, np.float64),
                np.asarray(live_chosen, np.float64),
            )
        except Exception as exc:
            logger.warning("[runtime_guard: rare_chars] batched feed failed: %s", exc)
            return []
        alerts: list[Incident] = []
        for row, st, req_id, req_idx, token_id, lp1, lpc in zip(
            rows, states, live_ids, live_idx, live_tok, live_top1, live_chosen
        ):
            st.tokens_seen += 1
            st.recent.append({"token_id": int(token_id), "top1": float(lp1), "chosen": float(lpc)})
            if len(st.recent) > 16:
                st.recent = st.recent[-16:]
            alert = self._incident_from_row(
                st,
                res=batch.strict_at(row),
                mon=batch.monitor_at(row),
                req_id=req_id,
                req_idx=req_idx,
                token_id=token_id,
                top1=lp1,
                chosen=lpc,
                log_leader=log_leader,
                source="guard_rare_chars_rc4",
            )
            if alert is not None:
                alerts.append(alert)
        return alerts

    def _incident_from_row(
        self,
        st: RareCharsReqState,
        *,
        res,
        mon,
        req_id: str,
        req_idx: int,
        token_id: int | None,
        top1: float | None,
        chosen: float | None,
        log_leader: bool,
        source: str,
    ) -> Incident | None:
        fired = res is not None and bool(getattr(res, "is_ill", False))
        if not fired and self._alert_monitor and mon is not None and bool(getattr(mon, "is_ill", False)):
            res = mon
            fired = True
        if not fired:
            return None
        st.alerted = True
        st.last_reason = str(getattr(res, "reason", "") or "")
        detail = {
            "violation": "rare_chars_rc4",
            "reason": st.last_reason,
            "fired_at": int(getattr(res, "fired_at", -1)),
            "tokens_seen": st.tokens_seen,
            "table": self._table,
            "recent": list(st.recent),
            "source": source,
            "api_logprobs_required": False,
            "tier": "monitor" if self._alert_monitor and "gb>" in st.last_reason.lower() else "strict",
        }
        if token_id is not None:
            detail["token_id"] = int(token_id)
            detail["top1_logprob"] = float(top1 or 0.0)
            detail["chosen_logprob"] = float(chosen or 0.0)
        if log_leader and token_id is not None:
            logger.error(
                "[runtime_guard: rare_chars] req=%s reason=%s fired_at=%s "
                "token_id=%s top1=%.4f chosen=%.4f tokens_seen=%s",
                req_id,
                st.last_reason,
                detail["fired_at"],
                token_id,
                float(top1 or 0.0),
                float(chosen or 0.0),
                st.tokens_seen,
            )
        elif log_leader:
            logger.error(
                "[runtime_guard: rare_chars] req=%s reason=%s fired_at=%s tokens_seen=%s",
                req_id,
                st.last_reason,
                detail["fired_at"],
                st.tokens_seen,
            )
        return Incident(
            incident_type=self.incident_type,
            req_id=req_id,
            req_idx=req_idx,
            is_ill=True,
            ill_type=ILL_TYPE_NONE,
            detail=detail,
        )

    @staticmethod
    def _coerce_pair_lists(
        top1_logprobs: Sequence[float] | torch.Tensor,
        chosen_logprobs: Sequence[float] | torch.Tensor,
        n: int,
    ) -> tuple[list[float], list[float]]:
        """Normalize precomputed (top1, chosen) to host float lists of length ``n``."""
        if isinstance(top1_logprobs, torch.Tensor):
            top1_list = [float(x) for x in top1_logprobs.detach().float().cpu().reshape(-1).tolist()]
        else:
            top1_list = [float(x) for x in top1_logprobs]
        if isinstance(chosen_logprobs, torch.Tensor):
            chosen_list = [float(x) for x in chosen_logprobs.detach().float().cpu().reshape(-1).tolist()]
        else:
            chosen_list = [float(x) for x in chosen_logprobs]
        if len(top1_list) < n or len(chosen_list) < n:
            raise ValueError(
                f"precomputed logprobs too short: top1={len(top1_list)} chosen={len(chosen_list)} need={n}"
            )
        return top1_list[:n], chosen_list[:n]

    def check_host(
        self,
        *,
        token_ids: Sequence[int],
        top1_logprobs: Sequence[float],
        chosen_logprobs: Sequence[float],
        req_id: str,
        req_idx: int = 0,
    ) -> list[Incident]:
        """Host-side feed for unit tests (precomputed natural-log logprobs)."""
        if not self._precheck():
            return []
        engine = self._ensure_engine()
        alert = self._fold_one(
            engine=engine,
            req_id=req_id,
            req_idx=req_idx,
            token_id=int(token_ids[-1]) if token_ids else -1,
            top1=float(top1_logprobs[-1]) if top1_logprobs else 0.0,
            chosen=float(chosen_logprobs[-1]) if chosen_logprobs else 0.0,
            log_leader=True,
            token_ids=list(token_ids),
            top1_list=list(top1_logprobs),
            chosen_list=list(chosen_logprobs),
        )
        return [alert] if alert is not None else []

    def _fold_one(
        self,
        *,
        engine: RareDetector,
        req_id: str,
        req_idx: int,
        token_id: int,
        top1: float,
        chosen: float,
        log_leader: bool,
        token_ids: list[int] | None = None,
        top1_list: list[float] | None = None,
        chosen_list: list[float] | None = None,
    ) -> Incident | None:
        st = self._states.get(req_id)
        if st is None:
            st = RareCharsReqState(det_state=engine.start())
            self._states[req_id] = st
        if st.alerted:
            return None

        ids = np.asarray(token_ids if token_ids is not None else [token_id], np.int64)
        lp1 = np.asarray(top1_list if top1_list is not None else [top1], np.float64)
        lpc = np.asarray(chosen_list if chosen_list is not None else [chosen], np.float64)
        try:
            res = engine.feed(st.det_state, ids, lp1, lpc)
        except Exception as exc:
            logger.warning("[runtime_guard: rare_chars] feed failed req=%s: %s", req_id, exc)
            return None

        st.tokens_seen += int(ids.size)
        st.recent.append({"token_id": int(token_id), "top1": float(top1), "chosen": float(chosen)})
        if len(st.recent) > 16:
            st.recent = st.recent[-16:]

        fired = res is not None and bool(getattr(res, "is_ill", False))
        if not fired and self._alert_monitor:
            mon = engine.monitor(st.det_state)
            if mon is not None and bool(getattr(mon, "is_ill", False)):
                res = mon
                fired = True

        if not fired:
            return None

        st.alerted = True
        st.last_reason = str(getattr(res, "reason", "") or "")
        detail = {
            "violation": "rare_chars_rc4",
            "reason": st.last_reason,
            "fired_at": int(getattr(res, "fired_at", -1)),
            "token_id": int(token_id),
            "top1_logprob": float(top1),
            "chosen_logprob": float(chosen),
            "tokens_seen": st.tokens_seen,
            "table": self._table,
            "recent": list(st.recent),
            "source": "guard_rare_chars_rc4",
            "api_logprobs_required": False,
            "tier": "monitor" if self._alert_monitor and "gb>" in st.last_reason.lower() else "strict",
        }
        if log_leader:
            logger.error(
                "[runtime_guard: rare_chars] req=%s reason=%s fired_at=%s "
                "token_id=%s top1=%.4f chosen=%.4f tokens_seen=%s",
                req_id,
                st.last_reason,
                detail["fired_at"],
                token_id,
                top1,
                chosen,
                st.tokens_seen,
            )
        return Incident(
            incident_type=self.incident_type,
            req_id=req_id,
            req_idx=req_idx,
            is_ill=True,
            ill_type=ILL_TYPE_NONE,
            detail=detail,
        )

    def finish_request(self, req_id: str) -> Incident | None:
        """Flush held-back token at request end (rc4 look-ahead)."""
        st = self._states.get(req_id)
        if st is None or st.alerted:
            return None
        row = self._row_of.get(req_id)
        if row is not None and self._row_batch is not None:
            try:
                res = self._row_batch.finish_row(row)
                mon = self._row_batch.monitor_at(row)
            except Exception as exc:
                logger.warning("[runtime_guard: rare_chars] finish failed req=%s: %s", req_id, exc)
                return None
            if self._row_batch._native is not None:
                self._copy_recent(st, self._row_batch, row)
            return self._incident_from_row(
                st,
                res=res,
                mon=mon,
                req_id=req_id,
                req_idx=0,
                token_id=None,
                top1=None,
                chosen=None,
                log_leader=runner_tp_rank(self._runner) == 0,
                source="guard_rare_chars_rc4_finish",
            )
        if st.det_state is None:
            return None
        try:
            engine = self._ensure_engine()
            res = engine.finish(st.det_state)
        except Exception as exc:
            logger.warning("[runtime_guard: rare_chars] finish failed req=%s: %s", req_id, exc)
            return None
        if res is None or not bool(getattr(res, "is_ill", False)):
            if self._alert_monitor:
                mon = engine.monitor(st.det_state)
                if mon is None or not bool(getattr(mon, "is_ill", False)):
                    return None
                res = mon
            else:
                return None
        st.alerted = True
        st.last_reason = str(getattr(res, "reason", "") or "")
        return Incident(
            incident_type=self.incident_type,
            req_id=req_id,
            req_idx=0,
            is_ill=True,
            ill_type=ILL_TYPE_NONE,
            detail={
                "violation": "rare_chars_rc4",
                "reason": st.last_reason,
                "fired_at": int(getattr(res, "fired_at", -1)),
                "tokens_seen": st.tokens_seen,
                "table": self._table,
                "source": "guard_rare_chars_rc4_finish",
                "api_logprobs_required": False,
            },
        )
