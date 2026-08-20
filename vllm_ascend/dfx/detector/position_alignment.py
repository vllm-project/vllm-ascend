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

"""Position-id alignment detector for 1-D RoPE text paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.config_backed import ConfigBackedDetector
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


def resolve_positions_tensor(runner: Any, total_scheduled: int) -> torch.Tensor | None:
    """Best-effort positions slice for this step (v1 / v2 runners)."""
    if total_scheduled <= 0:
        return None
    pos = getattr(runner, "positions", None)
    if isinstance(pos, torch.Tensor) and pos.numel() >= total_scheduled:
        return pos[:total_scheduled]
    buffers = getattr(runner, "input_buffers", None)
    if buffers is not None:
        buf_pos = getattr(buffers, "positions", None)
        if isinstance(buf_pos, torch.Tensor) and buf_pos.numel() >= total_scheduled:
            return buf_pos[:total_scheduled]
    return None


def query_start_loc_np(
    runner: Any,
    num_reqs: int,
    input_batch: Any = None,
) -> np.ndarray | None:
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


def _int_from_batch(input_batch: Any, req_idx: int | None) -> int | None:
    if input_batch is None or req_idx is None:
        return None
    for attr in ("num_computed_tokens_np", "num_computed_tokens_cpu"):
        arr = getattr(input_batch, attr, None)
        if arr is None:
            continue
        try:
            return int(arr[int(req_idx)])
        except Exception:
            continue
    return None


def _int_from_requests(runner: Any, req_id: str) -> int | None:
    requests = getattr(runner, "requests", None)
    if requests is None:
        return None
    state = requests.get(req_id)
    if state is None:
        return None
    n = getattr(state, "num_computed_tokens", None)
    if n is None:
        return None
    try:
        return int(n)
    except (TypeError, ValueError):
        return None


def num_computed_before(
    runner: Any,
    req_id: str,
    req_idx: int | None,
    scheduled: int,
    input_batch: Any = None,
) -> int | None:
    """Return wave-before ``num_computed_tokens``, or ``None`` if unknown.

    ``0`` is a valid first-prefill value — never treat it as "missing".
    Prefer ``input_batch`` (same source as positions) over ``runner.requests``.
    At ``check_before_sample`` / prepare-input time, counters are **before**
    this step's scheduled tokens. Do **not** subtract ``scheduled``.
    """
    del scheduled  # kept for call-site clarity; not used for adjustment
    if input_batch is None:
        input_batch = getattr(runner, "input_batch", None)
    from_batch = _int_from_batch(input_batch, req_idx)
    if from_batch is not None:
        return from_batch
    return _int_from_requests(runner, req_id)


class PositionAlignmentDetector(ConfigBackedDetector):
    """Detect non-monotonic or misaligned 1-D ``position_ids`` for new tokens."""

    anomaly_type = "position_alignment"
    section_key = "position_alignment"

    def __init__(self, *, dfx_config: Any | None = None, runner: Any | None = None) -> None:
        super().__init__(dfx_config=dfx_config, runner=runner, enabled=False)
        if dfx_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        return

    def check_all(
        self,
        *,
        scheduler_output: Any,
        positions: torch.Tensor | None,
        total_scheduled: int,
        input_batch: Any = None,
    ) -> list[AnomalyAlert]:
        if not self._precheck():
            return []
        runner = self._runner
        if runner is None or scheduler_output is None or total_scheduled <= 0:
            return []
        if positions is None:
            positions = resolve_positions_tensor(runner, total_scheduled)
        if positions is None:
            logger.debug(
                "[Anomaly position_alignment] skip: no positions tensor (total_scheduled=%s)",
                total_scheduled,
            )
            return []
        # 1-D text RoPE only; M-RoPE / multimodal layouts would false-positive.
        ndim = int(getattr(positions, "ndim", 1) or 1)
        if ndim != 1:
            logger.debug(
                "[Anomaly position_alignment] skip: positions.ndim=%s (1-D text RoPE only; M-RoPE unsupported)",
                ndim,
            )
            return []
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if not isinstance(num_scheduled, dict) or not num_scheduled:
            return []
        if input_batch is None:
            input_batch = getattr(runner, "input_batch", None)
        req_ids = list(getattr(input_batch, "req_ids", None) or [])
        req_id_to_index = getattr(input_batch, "req_id_to_index", None) if input_batch else None
        if not isinstance(req_id_to_index, dict):
            req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}
        num_reqs = len(req_ids)
        qsl = query_start_loc_np(runner, num_reqs, input_batch)
        if qsl is None:
            logger.debug(
                "[Anomaly position_alignment] skip: query_start_loc unresolved (num_reqs=%s)",
                num_reqs,
            )
            return []
        pos = positions[:total_scheduled]
        n_pos = int(pos.shape[0])
        expected_np = np.zeros(n_pos, dtype=np.int64)
        mask_np = np.zeros(n_pos, dtype=bool)
        spans: list[tuple[str, int, int, int, int, int]] = []
        skipped_unknown_computed = 0
        for req_id in req_ids:
            if not req_id:
                continue
            try:
                scheduled = int(num_scheduled.get(req_id, 0))
            except (TypeError, ValueError):
                continue
            if scheduled <= 0:
                continue
            req_idx = None
            if isinstance(req_id_to_index, dict) and req_id in req_id_to_index:
                req_idx = int(req_id_to_index[req_id])
            if not self._passes_input_filter(str(req_id), req_idx, log=False):
                continue
            if req_idx is None or req_idx >= num_reqs:
                continue
            start = int(qsl[req_idx])
            end = int(qsl[req_idx + 1])
            if end <= start or start >= n_pos:
                continue
            end = min(end, n_pos)
            expected_start = num_computed_before(
                runner,
                str(req_id),
                req_idx,
                scheduled,
                input_batch,
            )
            if expected_start is None:
                skipped_unknown_computed += 1
                continue
            width = end - start
            expected_np[start:end] = expected_start + np.arange(width, dtype=np.int64)
            mask_np[start:end] = True
            spans.append((str(req_id), req_idx, start, end, expected_start, scheduled))
        if skipped_unknown_computed:
            logger.debug(
                "[Anomaly position_alignment] skipped %s req(s): num_computed_before unknown",
                skipped_unknown_computed,
            )
        if not spans:
            return []
        try:
            mask_t = torch.as_tensor(mask_np, device=pos.device)
            exp_t = torch.as_tensor(expected_np, device=pos.device, dtype=pos.dtype)
            matched = (~mask_t) | (pos == exp_t)
            if bool(matched.all().item()):
                return []
            pos_np = pos.detach().cpu().numpy().astype(np.int64, copy=False)
        except Exception as exc:
            logger.debug(
                "[Anomaly position_alignment] device compare/D2H failed: %s",
                exc,
            )
            return []
        alerts: list[AnomalyAlert] = []
        for req_id, req_idx, start, end, expected_start, scheduled in spans:
            slice_pos = pos_np[start:end]
            expected = expected_start + np.arange(slice_pos.size, dtype=np.int64)
            if np.array_equal(slice_pos, expected):
                continue
            violation = "non_consecutive" if slice_pos.size > 1 and np.all(np.diff(slice_pos) == 1) else "misaligned"
            if slice_pos.size and slice_pos[0] != expected_start:
                violation = "wrong_start"
            alerts.append(
                AnomalyAlert(
                    anomaly_type=self.anomaly_type,
                    req_id=req_id,
                    req_idx=req_idx,
                    detail={
                        "violation": violation,
                        "expected_start": int(expected_start),
                        "expected_positions": expected.tolist()[:16],
                        "actual_positions": slice_pos.tolist()[:16],
                        "scheduled_tokens": scheduled,
                        "num_computed_before": int(expected_start),
                    },
                )
            )
        return alerts
