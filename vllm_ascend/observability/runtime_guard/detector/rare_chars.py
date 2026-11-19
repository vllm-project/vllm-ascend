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

"""Runtime Guard wrapper for rare_chars (garbled / foreign-script detection).

Hot path (intentionally multi-stage so D2H+feed can hide under forward / AsyncOutput)::

    pre-sample stash logits
      → after sample: Triton narrow collect → RareLogprobBatch (+ ready_event)
      → AscendAsyncOutput.get_output → RareLogprobFeedQueue
      → wait collect + D2H → native pack/feed (GIL released)

Guard-local state lives on ``RareCharsSession`` (do not grow without documenting)::

    warmup_failed   process latch; skip stash/collect until enabled off→on
    warmed          table + NativeFeed bound
    row_batch       RareRowBatch (native row arena)
    row_by_req      req_id → row index
    states          per-req Python alert bookkeeping
    _lock           feed / finish / clear across FeedQueue + ActionQueue

Debug: ``VLLM_ASCEND_RG_RARE_SYNC_FEED=1`` runs D2H+feed inline (no FeedQueue).
Numeric FT_*/IN_* ids: ``detector/rare/rare_enums.py`` parses ``rare_feed_native.c``.
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


def _load_rare_algo_config() -> tuple[dict[str, Any], str, bool]:
    """Load rare.json once for RareDetector + Guard knobs."""
    with open(CONFIG_RARE, encoding="utf-8") as f:
        cfg = json.load(f)
    think_end = str(cfg.get("think_end", "auto"))
    alert_monitor = bool(cfg.get("alert_monitor", False))
    return cfg, think_end, alert_monitor


def last_sampled_token_ids(sampled_token_ids: torch.Tensor) -> torch.Tensor:
    """Last valid sampled id per row; empty rows → ``-1``."""
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
    token_ids = ids.gather(1, last.unsqueeze(1)).reshape(-1).to(dtype=torch.int64)
    return torch.where(lengths > 0, token_ids, torch.full_like(token_ids, -1))


def _as_cu_num_logits_list(cu_num_logits: Any, num_reqs: int) -> list[int] | None:
    """Normalize ``cu_num_logits`` to a host ``list[int]`` of length ``num_reqs + 1``."""
    if cu_num_logits is None:
        return None
    if isinstance(cu_num_logits, torch.Tensor):
        cu = cu_num_logits.detach()
        if cu.device.type != "cpu":
            cu = cu.cpu()
        vals = [int(x) for x in cu.reshape(-1).tolist()]
    else:
        arr = np.asarray(cu_num_logits).reshape(-1)
        vals = [int(x) for x in arr.tolist()]
    if len(vals) != num_reqs + 1:
        return None
    if vals[0] != 0:
        return None
    for i in range(num_reqs):
        if vals[i + 1] < vals[i]:
            return None
    return vals


def align_sampled_token_ids_to_logits(
    sampled_token_ids: torch.Tensor,
    num_logit_rows: int,
    *,
    cu_num_logits: Any | None = None,
) -> tuple[torch.Tensor, int, list[int] | None] | None:
    """Align sampler ids to pre-sample logit rows for rare_chars collect.

    Returns ``(token_ids [num_logit_rows], slots_per_req, cu_num_logits_or_none)``
    or ``None`` when the layout cannot be matched safely.

    Layouts:
    - Non-spec / prefill: ``logits`` rows == ``num_reqs``; use last valid id/row.
    - Fixed-slot MTP: ``logits`` is ``num_reqs * slots`` and ``sampled`` is
      ``[num_reqs, slots]`` → reshape flatten (``slots_per_req = slots``).
    - Adaptive verification: per-request logit spans come from ``cu_num_logits``;
      ``sampled`` stays ``[num_reqs, max_slots]`` with ``-1`` pads. Scatter
      ``sampled[i, :span]`` into ``flat[cu[i]:cu[i+1]]`` (``slots_per_req = 0``).
    """
    if num_logit_rows <= 0:
        return None
    ids = sampled_token_ids.detach()
    if ids.numel() == 0:
        return None
    if ids.dim() == 1:
        flat = ids.to(dtype=torch.int64)
        if flat.numel() != num_logit_rows:
            return None
        return flat, 1, None
    if ids.dim() != 2:
        return None
    num_reqs, slots = int(ids.shape[0]), int(ids.shape[1])
    if slots <= 0 or num_reqs <= 0:
        return None
    # Spec verify (fixed slots): one logit row per (req, draft/bonus) slot.
    if num_logit_rows == num_reqs * slots:
        return ids.reshape(-1).to(dtype=torch.int64), slots, None
    # One logit row per request (non-spec decode, or first token).
    if num_logit_rows == num_reqs:
        return last_sampled_token_ids(ids), 1, None
    # Adaptive / varlen: scatter through cu_num_logits.
    cu = _as_cu_num_logits_list(cu_num_logits, num_reqs)
    if cu is None or cu[-1] != num_logit_rows:
        return None
    flat = torch.full((num_logit_rows,), -1, dtype=torch.int64, device=ids.device)
    ids64 = ids.to(dtype=torch.int64)
    for i in range(num_reqs):
        start = cu[i]
        end = cu[i + 1]
        span = end - start
        if span <= 0:
            continue
        take = min(span, slots)
        flat[start : start + take] = ids64[i, :take]
    return flat, 0, cu


def expand_req_ids_for_logit_rows(
    req_ids: Sequence[str],
    *,
    n_rows: int,
    slots_per_req: int = 1,
    cu_num_logits: Sequence[int] | None = None,
) -> list[str]:
    """Repeat ``req_ids`` to match logit / rare-pair rows."""
    ids = [str(rid) for rid in req_ids]
    if n_rows <= 0 or not ids:
        return []
    if cu_num_logits is not None:
        cu = _as_cu_num_logits_list(cu_num_logits, len(ids))
        if cu is not None and cu[-1] >= n_rows:
            expanded: list[str] = []
            for i, rid in enumerate(ids):
                expanded.extend([rid] * (cu[i + 1] - cu[i]))
            return expanded[:n_rows]
    slots = max(1, int(slots_per_req))
    if slots > 1:
        expanded = []
        for rid in ids:
            expanded.extend([rid] * slots)
        return expanded[:n_rows]
    return ids[:n_rows]


def build_rare_feed_arrays(
    host_logprobs: Any,
    sampled: Any,
    n_req: int,
    *,
    host_token_ids: Any | None = None,
    slots_per_req: int = 1,
    req_ids: Sequence[str] | None = None,
    cu_num_logits: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    """Host ``(token_ids, top1, sampled_logprobs[, expanded_req_ids])`` for one step.

    When collect stashed MTP / adaptive-expanded pairs, pass aligned
    ``host_token_ids`` plus either fixed ``slots_per_req`` or ``cu_num_logits``
    so every draft/bonus position is fed (``token_id < 0`` rows are filtered
    later in :meth:`RareCharsDetector.feed_precomputed`).
    """
    if isinstance(host_logprobs, torch.Tensor):
        pairs = host_logprobs.detach()
        if pairs.device.type != "cpu":
            pairs = pairs.cpu()
        pairs_np = pairs.numpy()
    else:
        pairs_np = np.asarray(host_logprobs)
    if pairs_np.ndim != 2 or pairs_np.shape[1] < 2:
        raise ValueError(f"rare logprobs must be [rows, 2], got shape={getattr(pairs_np, 'shape', None)}")

    slots = int(slots_per_req)
    n_req_i = int(n_req)
    if n_req_i <= 0 or pairs_np.shape[0] <= 0:
        empty_i = np.zeros(0, np.int64)
        empty_f = np.zeros(0, np.float64)
        return empty_i, empty_f, empty_f, None

    if host_token_ids is not None:
        if isinstance(host_token_ids, torch.Tensor):
            tok_t = host_token_ids.detach()
            if tok_t.device.type != "cpu":
                tok_t = tok_t.cpu()
            tok_np = np.ascontiguousarray(tok_t.numpy(), dtype=np.int64).reshape(-1)
        else:
            tok_np = np.ascontiguousarray(np.asarray(host_token_ids), dtype=np.int64).reshape(-1)
        n = min(int(pairs_np.shape[0]), int(tok_np.shape[0]))
        if cu_num_logits is not None and n_req_i > 0:
            cu = _as_cu_num_logits_list(cu_num_logits, n_req_i)
            if cu is not None:
                n = min(n, int(cu[-1]))
        elif slots > 1 and n_req_i > 0:
            n = min(n, n_req_i * slots)
        if n <= 0:
            empty_i = np.zeros(0, np.int64)
            empty_f = np.zeros(0, np.float64)
            return empty_i, empty_f, empty_f, None
        base = np.ascontiguousarray(pairs_np[:n], dtype=np.float64)
        top1_logprobs = np.empty(n, dtype=np.float64)
        sampled_logprobs = np.empty(n, dtype=np.float64)
        top1_logprobs[:] = base[:, 0]
        sampled_logprobs[:] = base[:, 1]
        expanded: list[str] | None = None
        if req_ids is not None and (slots > 1 or cu_num_logits is not None):
            expanded = expand_req_ids_for_logit_rows(
                list(req_ids)[:n_req_i],
                n_rows=n,
                slots_per_req=max(1, slots),
                cu_num_logits=cu_num_logits,
            )
        elif req_ids is not None:
            expanded = [str(rid) for rid in list(req_ids)[:n]]
        return np.ascontiguousarray(tok_np[:n], dtype=np.int64), top1_logprobs, sampled_logprobs, expanded

    # Legacy 1-row-per-request path (no stashed token ids).
    n = min(n_req_i, int(pairs_np.shape[0]))
    if n <= 0:
        empty_i = np.zeros(0, np.int64)
        empty_f = np.zeros(0, np.float64)
        return empty_i, empty_f, empty_f, None
    base = np.ascontiguousarray(pairs_np[:n], dtype=np.float64)
    top1_logprobs = np.empty(n, dtype=np.float64)
    sampled_logprobs = np.empty(n, dtype=np.float64)
    top1_logprobs[:] = base[:, 0]
    sampled_logprobs[:] = base[:, 1]
    return _last_token_ids(sampled, n), top1_logprobs, sampled_logprobs, None


def _last_token_ids(sampled: Any, n: int) -> np.ndarray:
    if not isinstance(sampled, torch.Tensor):
        sampled = torch.as_tensor(sampled, dtype=torch.int64)
    token_ids = last_sampled_token_ids(sampled)
    if token_ids.device.type != "cpu":
        token_ids = token_ids.detach().cpu()
    return np.ascontiguousarray(token_ids.numpy()[:n], dtype=np.int64)


@dataclass
class RareCharsReqState:
    tokens_seen: int = 0
    alerted: bool = False
    last_reason: str = ""
    recent: list[dict[str, float | int | str]] = field(default_factory=list)


@dataclass
class RareCharsSession:
    """Process-local bind + row-arena lifecycle for ``RareCharsDetector``.

    Transitions::

        idle  --warmup ok-->  ready
        *     --bind/warmup fail-->  failed  (stash/collect/feed skip)
        failed --enabled off (or note_config_disabled)-->  idle  (retry on re-enable)
        ready  --corpus/think_end rebind-->  ready  (row map cleared)
    """

    warmed: bool = False
    warmup_failed: bool = False
    engine: RareDetector | None = None
    engine_key: tuple[str, int | None] | None = None
    corpus_key: str | None = None
    row_batch: RareRowBatch | None = None
    row_by_req: dict[str, int] = field(default_factory=dict)
    states: dict[str, RareCharsReqState] = field(default_factory=dict)

    def clear_fail_latch(self) -> None:
        self.warmup_failed = False

    def mark_failed(self) -> None:
        self.warmup_failed = True
        self.warmed = False

    def mark_ready(self) -> None:
        self.warmed = True
        self.warmup_failed = False

    def can_feed(self) -> bool:
        return not self.warmup_failed

    def drop_request(self, req_id: str) -> None:
        row = self.row_by_req.pop(req_id, None)
        batch = self.row_batch
        if batch is not None and batch._id_map is not None:
            batch._native.map_del(batch._id_map, req_id)
        if row is not None and batch is not None:
            if row < len(batch._req_at):
                batch._req_at[row] = None
            batch.release(row)
        self.states.pop(req_id, None)

    def rebind_rows(self, engine: RareDetector, key: tuple[str, int | None]) -> RareRowBatch:
        self.engine = engine
        self.engine_key = key
        # RareRowBatch requires rare_feed_native (built by setuptools Extension).
        self.row_batch = RareRowBatch(engine)
        self.row_by_req.clear()
        return self.row_batch


class RareCharsDetector(ConfigBackedDetector):
    """Shadow detector: emit ``Incident`` reports; does not abort requests."""

    incident_type = "rare_chars"
    section_key = "rare_chars"
    schema = DetectorSchema(
        section_key="rare_chars",
        stage="after_sample",
        help="Detect garbled / foreign-script / illegal UTF-8 via (top1, chosen) logprobs.",
        fields=(ConfigField("enabled", False, "bool", help="Master switch"),),
    )

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig | None = None,
        runner: Any | None = None,
    ) -> None:
        super().__init__(runtime_config=runtime_config, runner=runner, enabled=False)
        self._session = RareCharsSession()
        self._think_end_cfg = "auto"
        self._think_end_id: int | None = None
        self._alert_monitor = False
        # Serializes feed / finish / clear across rare-pair-wait, async output,
        # and ActionQueue threads (native feed releases the GIL).
        self._lock = threading.RLock()
        if runtime_config is not None:
            self.refresh_from_config()

    @property
    def warmup_failed(self) -> bool:
        return bool(self._session.warmup_failed)

    # Compatibility aliases used by feed/finish helpers and unit tests.
    @property
    def _warmed(self) -> bool:
        return self._session.warmed

    @_warmed.setter
    def _warmed(self, value: bool) -> None:
        self._session.warmed = bool(value)

    @property
    def _warmup_failed(self) -> bool:
        return self._session.warmup_failed

    @_warmup_failed.setter
    def _warmup_failed(self, value: bool) -> None:
        self._session.warmup_failed = bool(value)

    @property
    def _engine(self) -> RareDetector | None:
        return self._session.engine

    @property
    def _engine_key(self) -> tuple[str, int | None] | None:
        return self._session.engine_key

    @property
    def _corpus_key(self) -> str | None:
        return self._session.corpus_key

    @_corpus_key.setter
    def _corpus_key(self, value: str | None) -> None:
        self._session.corpus_key = value

    @property
    def _row_batch(self) -> RareRowBatch | None:
        return self._session.row_batch

    @property
    def _row_by_req(self) -> dict[str, int]:
        return self._session.row_by_req

    # Back-compat for older tests / call sites.
    @property
    def _row_of(self) -> dict[str, int]:
        return self._session.row_by_req

    @property
    def _states(self) -> dict[str, RareCharsReqState]:
        return self._session.states

    def note_config_disabled(self) -> None:
        """Clear the warmup-fail latch when JSON ``enabled`` is false."""
        self._session.clear_fail_latch()

    def _apply_detector_section(self, section: dict[str, Any]) -> None:
        self._enabled = bool(section.get("enabled", self._enabled))
        if not self._enabled:
            self._session.clear_fail_latch()
        self._apply_detector_values(section.get)

    def _apply_detector_values(self, getter) -> None:
        # Only ``enabled`` is runtime-configurable; algorithm knobs live in rare.json.
        return

    def _mark_warmup_failed(self, exc: BaseException) -> None:
        self._session.mark_failed()
        logger.warning(
            "[runtime_guard: rare_chars] warmup failed; latching off stash/collect until "
            "detector.rare_chars.enabled is turned off then on again: %s",
            exc,
        )

    def warmup(self, tokenizer: Any | None = None) -> None:
        """Build the rare table from the served tokenizer (model warmup)."""
        self.refresh_from_config()
        if not self._enabled or self._session.warmup_failed:
            return
        try:
            with self._lock:
                self._warmup_locked(tokenizer)
        except Exception as exc:
            with self._lock:
                self._mark_warmup_failed(exc)
            raise

    def _warmup_locked(self, tokenizer: Any | None) -> None:
        if tokenizer is None:
            from vllm_ascend.observability.runtime_guard.io import load_model_tokenizer

            tokenizer = load_model_tokenizer(self._runner)
        if tokenizer is None:
            raise RuntimeError("rare_chars warmup needs a served tokenizer")
        built = build_from_tokenizer(tokenizer)
        self._session.corpus_key = register_rare_table(_SERVED_TABLE_KEY, built)
        logger.info(
            "[runtime_guard: rare_chars] built table vocab=%s partial=%s key=%s",
            len(built["rcls"]),
            int(built["u8"].sum()),
            self._session.corpus_key,
        )
        _cfg, self._think_end_cfg, self._alert_monitor = _load_rare_algo_config()
        self._think_end_id = resolve_think_end_id(tokenizer, self._think_end_cfg)
        self._ensure_engine_locked(_cfg)
        self._session.mark_ready()

    def clear_finished(self, req_id: str) -> None:
        with self._lock:
            self._session.drop_request(req_id)

    def _ensure_engine_locked(self, algo_cfg: dict[str, Any] | None = None) -> RareDetector:
        if self._session.corpus_key is None:
            raise RuntimeError("rare_chars engine used before warmup (table not registered)")
        key = (self._session.corpus_key, self._think_end_id)
        if self._session.engine is not None and self._session.engine_key == key:
            return self._session.engine
        if algo_cfg is None:
            algo_cfg, self._think_end_cfg, self._alert_monitor = _load_rare_algo_config()
        engine = RareDetector(self._session.corpus_key, algo_cfg, think_end=self._think_end_id)
        self._session.rebind_rows(engine, key)
        logger.info(
            "[runtime_guard: rare_chars] loaded rare table=%s think_end=%s alert_monitor=%s",
            self._session.corpus_key,
            self._think_end_id,
            self._alert_monitor,
        )
        return engine

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
            if not self._session.can_feed():
                return []
            if not self._session.warmed:
                try:
                    self._warmup_locked(None)
                except Exception as exc:
                    self._mark_warmup_failed(exc)
                    return []
            try:
                self._ensure_engine_locked()
            except Exception as exc:
                self._mark_warmup_failed(exc)
                return []
            # Vector filter: one index pass + numpy gather (no per-row Python
            # float()/append into five parallel lists).
            tok = np.asarray(token_ids[:n], dtype=np.int64)
            top1_a = np.asarray(top1_logprobs[:n], dtype=np.float64)
            chosen_a = np.asarray(chosen_logprobs[:n], dtype=np.float64)
            keep_idx = [i for i in range(n) if req_ids[i] and int(tok[i]) >= 0]
            if not keep_idx:
                return []
            keep_ids = [req_ids[i] for i in keep_idx]
            return self._feed_native_locked(
                req_ids=keep_ids,
                req_idxs=keep_idx,
                token_ids=tok[keep_idx],
                top1=top1_a[keep_idx],
                chosen=chosen_a[keep_idx],
                log_leader=runner_tp_rank(self._runner) == 0,
            )

    def _copy_recent(self, req_state: RareCharsReqState, batch: RareRowBatch, row: int) -> None:
        n = int(batch._recent_len[row])
        req_state.tokens_seen = int(batch._tokens_seen[row])
        tok = batch._recent_tok[row, :n]
        top1 = batch._recent_lp1[row, :n]
        chosen = batch._recent_lpc[row, :n]
        req_state.recent = [
            {"token_id": int(tok[i]), "top1": float(top1[i]), "chosen": float(chosen[i])} for i in range(n)
        ]

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
            self._row_by_req[req_id] = row
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
            req_state = self._states.get(req_id)
            if req_state is None or req_state.alerted:
                continue
            self._copy_recent(req_state, batch, row)
            last = req_state.recent[-1] if req_state.recent else None
            alert = self._incident_from_row(
                req_state,
                strict_result=batch.strict_at(row),
                monitor_result=batch.monitor_at(row),
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
        req_state: RareCharsReqState,
        *,
        strict_result,
        monitor_result,
        req_id: str,
        req_idx: int,
        token_id: int | None,
        top1: float | None,
        chosen: float | None,
        log_leader: bool,
        source: str,
    ) -> Incident | None:
        result = strict_result
        fired = result is not None and bool(getattr(result, "is_ill", False))
        if (
            not fired
            and self._alert_monitor
            and monitor_result is not None
            and bool(getattr(monitor_result, "is_ill", False))
        ):
            result = monitor_result
            fired = True
        if not fired:
            return None
        req_state.alerted = True
        req_state.last_reason = str(getattr(result, "reason", "") or "")
        detail = {
            "violation": "rare_chars",
            "reason": req_state.last_reason,
            "fired_at": int(getattr(result, "fired_at", -1)),
            "tokens_seen": req_state.tokens_seen,
            "table": self._corpus_key or _SERVED_TABLE_KEY,
            "recent": list(req_state.recent),
            "source": source,
            "api_logprobs_required": False,
            "tier": ("monitor" if self._alert_monitor and "gb>" in req_state.last_reason.lower() else "strict"),
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
                req_state.last_reason,
                detail["fired_at"],
                token_id,
                float(top1 or 0.0),
                float(chosen or 0.0),
                req_state.tokens_seen,
            )
        elif log_leader:
            logger.error(
                "[runtime_guard: rare_chars] req=%s reason=%s fired_at=%s tokens_seen=%s",
                req_id,
                req_state.last_reason,
                detail["fired_at"],
                req_state.tokens_seen,
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
            req_state = self._states.get(req_id)
            if req_state is None or req_state.alerted:
                return None
            row = self._row_by_req.get(req_id)
            if row is None or self._row_batch is None:
                return None
            try:
                strict_result = self._row_batch.finish_row(row)
                monitor_result = self._row_batch.monitor_at(row)
            except Exception as exc:
                logger.warning("[runtime_guard: rare_chars] finish failed req=%s: %s", req_id, exc)
                return None
            self._copy_recent(req_state, self._row_batch, row)
            return self._incident_from_row(
                req_state,
                strict_result=strict_result,
                monitor_result=monitor_result,
                req_id=req_id,
                req_idx=0,
                token_id=None,
                top1=None,
                chosen=None,
                log_leader=runner_tp_rank(self._runner) == 0,
                source="guard_rare_chars_finish",
            )
