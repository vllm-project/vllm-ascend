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

"""Runtime Guard wrapper for rare_chars (visible garbled / foreign-script).

Collects ``(top1_logprob, chosen_logprob)`` via sampler-side narrow collect on
Guard-stashed pre-sample logits + sampled ids (v35 Triton). Does **not**
require ``SamplingParams.logprobs``. Per-tokenizer tables are built at warmup
from the served tokenizer. Algorithm knobs (``think_end``, ``alert_monitor``)
live in ``rare.json``; runtime_config only exposes ``enabled``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_config.schema import ConfigField, DetectorSchema
from vllm_ascend.observability.runtime_guard.detector.base import ConfigBackedDetector
from vllm_ascend.observability.runtime_guard.detector.rare import (
    CONFIG_RARE,
    RareDetector,
    RareRowBatch,
    build_from_tokenizer,
    register_rare_table,
    resolve_think_end_id,
)
from vllm_ascend.observability.runtime_guard.rank_gate import runner_tp_rank
from vllm_ascend.observability.runtime_guard.state import ILL_TYPE_NONE, Incident

if TYPE_CHECKING:
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)

_SERVED_TABLE_KEY = "served"


def _algo_knobs() -> tuple[dict[str, Any], str, bool]:
    """Load rare.json once for RareDetector + Guard knobs."""
    with open(CONFIG_RARE, encoding="utf-8") as f:
        cfg = json.load(f)
    think_end = str(cfg.get("think_end", "auto"))
    alert_monitor = bool(cfg.get("alert_monitor", False))
    return cfg, think_end, alert_monitor


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
    """Token ids and logprob columns for one step, without a Python list per row."""
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


@dataclass
class RareCharsReqState:
    """Per-request streaming RareDetector state."""

    tokens_seen: int = 0
    alerted: bool = False
    last_reason: str = ""
    recent: list[dict[str, float | int | str]] = field(default_factory=list)


class RareCharsDetector(ConfigBackedDetector):
    """rare_chars: visible garbled / foreign-script / bad UTF-8.

    Shadow-oriented: emits Guard ``Incident`` reports; does not abort the
    request by itself (match upstream V35_RARE=shadow semantics).
    """

    incident_type = "rare_chars"
    section_key = "rare_chars"
    schema = DetectorSchema(
        section_key="rare_chars",
        stage="after_sample",
        help=(
            "Visible garbled text (rare_chars): foreign-script fragments, "
            "illegal UTF-8, low-confidence rare tokens. Guard uses sampler-side "
            "v35 Triton narrow collect on stashed pre-sample logits "
            "for (top1, chosen) — API SamplingParams.logprobs not required. "
            "Per-tokenizer table is built at warmup; think_end / alert_monitor "
            "come from detector/rare/rare.json."
        ),
        fields=(ConfigField("enabled", False, "bool", help="Master switch"),),
    )

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig | None = None,
        runner: Any | None = None,
    ) -> None:
        super().__init__(runtime_config=runtime_config, runner=runner, enabled=False)
        self._engine: RareDetector | None = None
        self._engine_key: tuple[str, int | None] | None = None
        self._corpus_key: str | None = None
        self._think_end_cfg = "auto"
        self._think_end_id: int | None = None
        self._alert_monitor = False
        self._warmed = False
        self._states: dict[str, RareCharsReqState] = {}
        self._row_batch: RareRowBatch | None = None
        self._row_of: dict[str, int] = {}
        # Serializes feed / finish / clear across rare-pair-wait, async output,
        # and ActionQueue threads (native feed releases the GIL).
        self._lock = threading.RLock()
        if runtime_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter) -> None:
        # Only ``enabled`` is runtime-configurable; algorithm knobs live in rare.json.
        return

    def warmup(self, tokenizer: Any | None = None) -> None:
        """Build the rare table from the served tokenizer (model warmup)."""
        self.refresh_from_config()
        if not self._enabled:
            return
        with self._lock:
            self._warmup_locked(tokenizer)

    def _warmup_locked(self, tokenizer: Any | None) -> None:
        if tokenizer is None:
            from vllm_ascend.observability.runtime_guard.io import load_model_tokenizer

            tokenizer = load_model_tokenizer(self._runner)
        if tokenizer is None:
            raise RuntimeError("rare_chars warmup needs a served tokenizer")
        built = build_from_tokenizer(tokenizer)
        self._corpus_key = register_rare_table(_SERVED_TABLE_KEY, built)
        logger.info(
            "[runtime_guard: rare_chars] built table vocab=%s partial=%s key=%s",
            len(built["rcls"]),
            int(built["u8"].sum()),
            self._corpus_key,
        )
        _cfg, self._think_end_cfg, self._alert_monitor = _algo_knobs()
        self._think_end_id = resolve_think_end_id(tokenizer, self._think_end_cfg)
        self._ensure_engine_locked(_cfg)
        self._warmed = True

    def clear_finished(self, req_id: str) -> None:
        with self._lock:
            row = self._row_of.pop(req_id, None)
            batch = self._row_batch
            if batch is not None and batch._id_map is not None:
                batch._native.map_del(batch._id_map, req_id)
            if row is not None and batch is not None:
                if row < len(batch._req_at):
                    batch._req_at[row] = None
                batch.release(row)
            self._states.pop(req_id, None)

    def _ensure_engine_locked(self, algo_cfg: dict[str, Any] | None = None) -> RareDetector:
        if self._corpus_key is None:
            raise RuntimeError("rare_chars engine used before warmup (table not registered)")
        key = (self._corpus_key, self._think_end_id)
        if self._engine is not None and self._engine_key == key:
            return self._engine
        if algo_cfg is None:
            algo_cfg, self._think_end_cfg, self._alert_monitor = _algo_knobs()
        self._engine = RareDetector(self._corpus_key, algo_cfg, think_end=self._think_end_id)
        self._engine_key = key
        # RareRowBatch requires rare_feed_native (built by setuptools Extension).
        self._row_batch = RareRowBatch(self._engine)
        self._row_of.clear()
        logger.info(
            "[runtime_guard: rare_chars] loaded rare table=%s think_end=%s alert_monitor=%s",
            self._corpus_key,
            self._think_end_id,
            self._alert_monitor,
        )
        return self._engine

    def feed_precomputed(
        self,
        *,
        req_ids: list[str],
        token_ids: Sequence[int],
        top1_logprobs: Sequence[float],
        chosen_logprobs: Sequence[float],
    ) -> list[Incident]:
        """Feed rare_chars from host pairs. Caller runs this off the sample return path."""
        if not self._precheck():
            return []
        n = min(len(req_ids), len(token_ids), len(top1_logprobs), len(chosen_logprobs))
        if n <= 0:
            return []
        with self._lock:
            if not self._warmed:
                try:
                    self._warmup_locked(None)
                except Exception as exc:
                    logger.warning("[runtime_guard: rare_chars] lazy warmup failed: %s", exc)
                    return []
            try:
                self._ensure_engine_locked()
            except Exception as exc:
                logger.warning("[runtime_guard: rare_chars] engine init failed: %s", exc)
                return []
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
            return self._feed_native_locked(
                req_ids=keep_ids,
                req_idxs=keep_idx,
                token_ids=keep_tok,
                top1=keep_top1,
                chosen=keep_chosen,
                log_leader=runner_tp_rank(self._runner) == 0,
            )

    def _copy_recent(self, st: RareCharsReqState, batch: RareRowBatch, row: int) -> None:
        n = int(batch._recent_len[row])
        st.tokens_seen = int(batch._tokens_seen[row])
        tok = batch._recent_tok[row, :n]
        lp1 = batch._recent_lp1[row, :n]
        lpc = batch._recent_lpc[row, :n]
        st.recent = [{"token_id": int(tok[i]), "top1": float(lp1[i]), "chosen": float(lpc[i])} for i in range(n)]

    def _feed_native_locked(
        self,
        *,
        req_ids: Sequence[str],
        req_idxs: Sequence[int],
        token_ids: Sequence[int],
        top1: Sequence[float],
        chosen: Sequence[float],
        log_leader: bool,
    ) -> list[Incident]:
        batch = self._row_batch
        if batch is None or batch._native is None:
            raise RuntimeError("rare_feed_native.so is not built (required for rare_chars)")
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
                self._states[req_id] = RareCharsReqState()
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
                source="guard_rare_chars",
            )
            if alert is not None:
                batch._done[row] = True
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
            "violation": "rare_chars",
            "reason": st.last_reason,
            "fired_at": int(getattr(res, "fired_at", -1)),
            "tokens_seen": st.tokens_seen,
            "table": self._corpus_key or _SERVED_TABLE_KEY,
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

    def finish_request(self, req_id: str) -> Incident | None:
        """Flush held-back token at request end (look-ahead)."""
        with self._lock:
            st = self._states.get(req_id)
            if st is None or st.alerted:
                return None
            row = self._row_of.get(req_id)
            if row is None or self._row_batch is None:
                return None
            try:
                res = self._row_batch.finish_row(row)
                mon = self._row_batch.monitor_at(row)
            except Exception as exc:
                logger.warning("[runtime_guard: rare_chars] finish failed req=%s: %s", req_id, exc)
                return None
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
                source="guard_rare_chars_finish",
            )
