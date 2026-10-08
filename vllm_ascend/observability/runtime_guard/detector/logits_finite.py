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

"""Logits finite (NaN/Inf) detector on sampling rows."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_config._defaults import LOGITS_FINITE_DEFERRED_QUEUE_MAX
from vllm_ascend.observability.runtime_config.schema import ConfigField, DetectorSchema
from vllm_ascend.observability.runtime_guard.detector.base import ConfigBackedDetector
from vllm_ascend.observability.runtime_guard.state import ILL_TYPE_NAN, Incident

logger = init_logger_ascend(__name__)


def query_start_loc_np(
    runner: Any,
    num_reqs: int,
    input_batch: Any = None,
) -> np.ndarray | None:
    """Best-effort host ``query_start_loc`` of length ``num_reqs + 1``."""
    if input_batch is None:
        input_batch = getattr(runner, "input_batch", None)
    if input_batch is not None:
        # V2 AscendInputBatch / community InputBatch often expose numpy directly.
        qsl_np = getattr(input_batch, "query_start_loc_np", None)
        if qsl_np is not None:
            try:
                return np.asarray(qsl_np[: num_reqs + 1], dtype=np.int64)
            except Exception:
                pass
    qsl = getattr(runner, "query_start_loc", None)
    if qsl is None and input_batch is not None:
        qsl = getattr(input_batch, "query_start_loc", None)
    if qsl is None:
        return None
    if hasattr(qsl, "np"):
        arr = qsl.np[: num_reqs + 1]
    elif isinstance(qsl, torch.Tensor):
        arr = qsl[: num_reqs + 1].detach().cpu().numpy()
    else:
        arr = np.asarray(qsl[: num_reqs + 1])
    return np.asarray(arr, dtype=np.int64)


def req_index_for_flat_token(flat_idx: int, qsl: np.ndarray, num_reqs: int) -> int | None:
    for r in range(num_reqs):
        if int(qsl[r]) <= flat_idx < int(qsl[r + 1]):
            return r
    return None


@dataclass
class _PendingAsyncGate:
    """Pre-sample gate launch; ``.item()`` waits in :meth:`check_deferred`."""

    all_finite_host: torch.Tensor
    row_finite: torch.Tensor
    logits: torch.Tensor | None
    req_ids: list[str]
    num_reqs: int
    qsl: np.ndarray | None
    # Optional CPU copy of logits_indices (non-blocking); None when 1:1 decode.
    logits_indices_host: torch.Tensor | None
    skip_req_ids: set[str]


class LogitsFiniteDetector(ConfigBackedDetector):
    """Detect non-finite values in pre-sample logits rows.

    Each sample step:

    1. Device ``isfinite`` → all-finite gate scalar.
    2. **Default (``item_sync=false``):** non-blocking D2H of the gate; wait in
       :meth:`check_deferred`. Between launch and wait, grammar / other code may
       mutate ``logits`` in place (e.g. bitmask → ``-inf``). The gate still uses
       precomputed ``row_finite`` (unaffected), but hit-time ``finite_kind`` may
       observe post-mutation values. Set ``item_sync=true`` for a blocking
       ``.item()`` (+ hit resolve) at pre-sample so kind matches pre-grammar logits.
    3. Clean → nothing queued.
    4. Hit → resolve bad rows / ``finite_kind`` / ``req_id``; enqueue host
       ``Incident``s.
    5. :meth:`check_deferred` drains the queue (and finishes any async gates).
    """

    incident_type = "logits_finite"
    section_key = "logits_finite"
    schema = DetectorSchema(
        section_key="logits_finite",
        stage="before_sample",
        help="Pre-sample logits NaN/Inf (isfinite + all-finite gate).",
        fields=(
            ConfigField("enabled", False, "bool", help="Master switch"),
            ConfigField(
                "item_sync",
                False,
                "bool",
                help=(
                    "false: async .item() (wait in check_deferred; logits may mutate). "
                    "true: blocking .item() at pre-sample for pre-grammar kind."
                ),
            ),
        ),
    )

    def __init__(self, *, runtime_config: Any | None = None, runner: Any | None = None) -> None:
        super().__init__(runtime_config=runtime_config, runner=runner, enabled=False)
        self._deferred: deque[list[Incident]] = deque()
        self._pending_gates: deque[_PendingAsyncGate] = deque()
        self._deferred_queue_max = LOGITS_FINITE_DEFERRED_QUEUE_MAX
        self._item_sync = bool(self.schema.defaults_dict()["item_sync"])
        if runtime_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        self._item_sync = bool(getter("item_sync", self._item_sync))

    def clear_finished(self, req_id: str) -> None:
        del req_id

    def _stage_none(self) -> None:
        """Drop pending alerts / async gates when the detector is disabled."""
        self._deferred.clear()
        self._pending_gates.clear()

    def check_all(
        self,
        *,
        logits: torch.Tensor | None,
        logits_indices: torch.Tensor | None,
        input_batch: Any = None,
        skip_req_ids: set[str] | None = None,
    ) -> list[Incident]:
        """Pre-sample: launch / take the all-finite gate; enqueue on hit.

        Returns ``[]`` always; :meth:`check_deferred` delivers queued incidents
        (and completes async gates).

        ``skip_req_ids`` (``detection_stopped``): whole-batch skip avoids the
        gate sync; mixed batches still scan but do not enqueue stopped reqs.
        """
        if not self._precheck():
            self._stage_none()
            return []
        if logits is None or not isinstance(logits, torch.Tensor):
            return []
        if logits.numel() == 0:
            return []
        runner = self._runner
        if runner is None:
            return []

        skip = set(skip_req_ids or set())
        if input_batch is None:
            input_batch = getattr(runner, "input_batch", None)
        req_ids = list(getattr(input_batch, "req_ids", None) or [])
        # All live reqs already stopped: no host sync / enqueue.
        if skip and req_ids and all((not rid) or rid in skip for rid in req_ids):
            return []

        try:
            row_finite = torch.isfinite(logits).all(dim=-1)
            all_dev = row_finite.all()
        except Exception as exc:
            logger.warning("[runtime_guard: logits_finite] scan failed: %s", exc)
            return []

        num_reqs = len(req_ids)
        qsl = query_start_loc_np(runner, num_reqs, input_batch) if num_reqs > 0 else None
        if logits_indices is None:
            logits_indices = getattr(runner, "logits_indices", None)

        if self._item_sync:
            return self._sync_gate_and_enqueue(
                all_dev=all_dev,
                row_finite=row_finite,
                logits=logits,
                req_ids=req_ids,
                num_reqs=num_reqs,
                qsl=qsl,
                logits_indices=logits_indices,
                skip_req_ids=skip,
            )

        # Async: D2H the gate without waiting; resolve in check_deferred.
        try:
            all_host = all_dev.detach().to("cpu", non_blocking=True)
        except Exception as exc:
            logger.warning(
                "[runtime_guard: logits_finite] async gate D2H failed; falling back to sync .item(): %s",
                exc,
            )
            return self._sync_gate_and_enqueue(
                all_dev=all_dev,
                row_finite=row_finite,
                logits=logits,
                req_ids=req_ids,
                num_reqs=num_reqs,
                qsl=qsl,
                logits_indices=logits_indices,
                skip_req_ids=skip,
            )

        nrows = int(row_finite.shape[0]) if row_finite.dim() > 0 else 1
        idx_host: torch.Tensor | None = None
        # Spec / misaligned rows need indices at resolve time; snapshot now so a
        # later in-place rewrite (or buffer reuse) cannot poison attribution.
        if not (num_reqs > 0 and nrows == num_reqs) and isinstance(logits_indices, torch.Tensor):
            try:
                idx_host = logits_indices.detach().to("cpu", non_blocking=True)
            except Exception:
                idx_host = None

        self._pending_gates.append(
            _PendingAsyncGate(
                all_finite_host=all_host,
                row_finite=row_finite,
                logits=logits,
                req_ids=req_ids,
                num_reqs=num_reqs,
                qsl=qsl,
                logits_indices_host=idx_host,
                skip_req_ids=skip,
            )
        )
        if len(self._pending_gates) > self._deferred_queue_max:
            self._pending_gates.popleft()
            logger.warning_once("[runtime_guard: logits_finite] async gate queue overflow; dropped oldest batch")
        return []

    def _sync_gate_and_enqueue(
        self,
        *,
        all_dev: torch.Tensor,
        row_finite: torch.Tensor,
        logits: torch.Tensor | None,
        req_ids: list[str],
        num_reqs: int,
        qsl: np.ndarray | None,
        logits_indices: torch.Tensor | None,
        skip_req_ids: set[str],
    ) -> list[Incident]:
        """Blocking ``.item()`` on the all-finite gate; enqueue hits. Always ``[]``."""
        try:
            all_finite = bool(all_dev.item())
        except Exception as exc:
            logger.warning("[runtime_guard: logits_finite] sync .item() failed: %s", exc)
            return []
        if all_finite:
            return []
        alerts = self._build_alerts(
            row_finite=row_finite,
            logits=logits,
            req_ids=req_ids,
            num_reqs=num_reqs,
            qsl=qsl,
            logits_indices=logits_indices,
            skip_req_ids=skip_req_ids,
        )
        if alerts:
            self._enqueue_deferred(alerts)
        return []

    def _enqueue_deferred(self, alerts: list[Incident]) -> None:
        self._deferred.append(alerts)
        if len(self._deferred) > self._deferred_queue_max:
            self._deferred.popleft()
            logger.warning_once("[runtime_guard: logits_finite] deferred alert queue overflow; dropped oldest batch")

    def _resolve_pending_gates(self, skip_req_ids: set[str]) -> None:
        """Wait for async all-finite gates and enqueue hits."""
        while self._pending_gates:
            pending = self._pending_gates.popleft()
            try:
                all_finite = bool(pending.all_finite_host.item())
            except Exception as exc:
                logger.warning("[runtime_guard: logits_finite] async .item() wait failed: %s", exc)
                continue
            if all_finite:
                continue
            # Merge late stop-detect skips with the skip set at launch.
            skip = set(pending.skip_req_ids) | skip_req_ids
            alerts = self._build_alerts(
                row_finite=pending.row_finite,
                logits=pending.logits,
                req_ids=pending.req_ids,
                num_reqs=pending.num_reqs,
                qsl=pending.qsl,
                logits_indices=pending.logits_indices_host,
                skip_req_ids=skip,
            )
            if alerts:
                self._enqueue_deferred(alerts)

    def check_deferred(self, *, skip_req_ids: set[str] | None = None) -> list[Incident]:
        """Finish async gates (if any), then drain host-side incidents."""
        skip = set(skip_req_ids or set())
        if not self._enabled:
            self._stage_none()
            return []
        self._resolve_pending_gates(skip)
        if not self._deferred:
            return []
        out: list[Incident] = []
        while self._deferred:
            batch = self._deferred.popleft()
            if not skip:
                out.extend(batch)
                continue
            for alert in batch:
                if alert.req_id and alert.req_id in skip:
                    continue
                out.append(alert)
        return out

    def _build_alerts(
        self,
        *,
        row_finite: torch.Tensor,
        logits: torch.Tensor | None,
        req_ids: list[str],
        num_reqs: int,
        qsl: np.ndarray | None,
        logits_indices: torch.Tensor | None,
        skip_req_ids: set[str] | None,
    ) -> list[Incident]:
        try:
            bad_rows = (~row_finite).nonzero(as_tuple=False).flatten()
        except Exception as exc:
            logger.warning("[runtime_guard: logits_finite] bad-row resolve failed: %s", exc)
            return []
        skip = skip_req_ids or set()
        nrows = int(row_finite.shape[0]) if row_finite.dim() > 0 else 1
        # 1:1 decode rows ↔ reqs: no indices D2H. Spec / misaligned: materialize.
        if num_reqs > 0 and nrows == num_reqs:
            idx_list: list[int] = []
        else:
            idx_list = _materialize_idx_list(logits_indices)

        alerts: list[Incident] = []
        seen_req: set[str] = set()
        unresolved_rows: list[int] = []
        for row_t in bad_rows.tolist():
            row = int(row_t)
            req_id, req_idx = _row_to_req(row, req_ids, num_reqs, qsl, idx_list)
            if not req_id:
                # A1: unattributable rows → warn only (no guessed / null-req Incident).
                unresolved_rows.append(row)
                continue
            if req_id in seen_req or req_id in skip:
                continue
            seen_req.add(req_id)
            finite_kind = _finite_kind_from_logits(logits, row)
            alerts.append(
                Incident(
                    incident_type=self.incident_type,
                    req_id=req_id,
                    req_idx=req_idx,
                    is_ill=True,
                    # msprobe ILL table has no separate Inf code; keep NAN (4).
                    ill_type=ILL_TYPE_NAN,
                    detail={
                        "logits_row": row,
                        "flat_token_index": idx_list[row] if row < len(idx_list) else None,
                        "violation": "non_finite_logits",
                        "finite_kind": finite_kind,
                    },
                )
            )
        if unresolved_rows:
            logger.warning(
                "[runtime_guard: logits_finite] cannot attribute non-finite "
                "logits to any request; skipping those rows. rows=%s "
                "num_unresolved=%d num_reqs=%d nrows=%d",
                unresolved_rows[:16],
                len(unresolved_rows),
                num_reqs,
                nrows,
            )
        return alerts


def _row_to_req(
    row: int,
    req_ids: list[str],
    num_reqs: int,
    qsl: np.ndarray | None,
    idx_list: list[int],
) -> tuple[str | None, int | None]:
    req_id: str | None = None
    req_idx: int | None = None
    if num_reqs > 0 and row < num_reqs and not idx_list:
        req_idx = row
        req_id = req_ids[row] if row < len(req_ids) else None
    elif qsl is not None and idx_list and row < len(idx_list):
        flat = idx_list[row]
        req_idx = req_index_for_flat_token(flat, qsl, num_reqs)
        if req_idx is not None and req_idx < len(req_ids):
            req_id = req_ids[req_idx]
    elif num_reqs > 0 and row < len(req_ids):
        req_idx = row
        req_id = req_ids[row]
    return req_id, req_idx


def _materialize_idx_list(logits_indices: torch.Tensor | None) -> list[int]:
    if not isinstance(logits_indices, torch.Tensor):
        return []
    try:
        return [int(x) for x in logits_indices.detach().cpu().tolist()]
    except Exception:
        return []


def _finite_kind_from_logits(logits: torch.Tensor | None, row: int) -> str:
    """Hit-only kind classification on one logits row (ill_type stays NAN)."""
    if logits is None:
        return "non_finite"
    try:
        row_t = logits[row]
        if bool(torch.isnan(row_t).any().item()):
            return "nan"
        if bool(torch.isposinf(row_t).any().item()):
            return "pos_inf"
        if bool(torch.isneginf(row_t).any().item()):
            return "neg_inf"
    except Exception:
        return "non_finite"
    return "non_finite"
