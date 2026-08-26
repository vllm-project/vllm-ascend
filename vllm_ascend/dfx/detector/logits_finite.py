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

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.config_backed import ConfigBackedDetector
from vllm_ascend.dfx.detector.position_alignment import query_start_loc_np
from vllm_ascend.dfx.dfx_types import ILL_TYPE_NAN
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


def req_index_for_flat_token(flat_idx: int, qsl: np.ndarray, num_reqs: int) -> int | None:
    for r in range(num_reqs):
        if int(qsl[r]) <= flat_idx < int(qsl[r + 1]):
            return r
    return None


class LogitsFiniteDetector(ConfigBackedDetector):
    """Detect non-finite values in pre-sample logits rows.

    One device reduce + ``.item()`` sync per sample step when enabled.
    Full row / ``logits_indices`` D2H only on a hit.
    """

    anomaly_type = "logits_finite"
    section_key = "logits_finite"

    def __init__(self, *, dfx_config: Any | None = None, runner: Any | None = None) -> None:
        super().__init__(dfx_config=dfx_config, runner=runner, enabled=False)
        if dfx_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        return

    def check_all(
        self,
        *,
        logits: torch.Tensor | None,
        logits_indices: torch.Tensor | None,
        input_batch: Any = None,
    ) -> list[AnomalyAlert]:
        if not self._precheck() or logits is None or not isinstance(logits, torch.Tensor):
            return []
        if logits.numel() == 0:
            return []
        runner = self._runner
        if runner is None:
            return []
        try:
            row_finite = torch.isfinite(logits).all(dim=-1)
            if bool(row_finite.all().item()):
                return []
            bad_rows = (~row_finite).nonzero(as_tuple=False).flatten()
        except Exception as exc:
            logger.warning("[Anomaly logits_finite] check failed: %s", exc)
            return []
        if input_batch is None:
            input_batch = getattr(runner, "input_batch", None)
        req_ids = list(getattr(input_batch, "req_ids", None) or [])
        num_reqs = len(req_ids)
        qsl = query_start_loc_np(runner, num_reqs, input_batch) if num_reqs > 0 else None
        if logits_indices is None:
            logits_indices = getattr(runner, "logits_indices", None)
        idx_list: list[int] = []
        if isinstance(logits_indices, torch.Tensor):
            try:
                idx_list = [int(x) for x in logits_indices.detach().cpu().tolist()]
            except Exception:
                idx_list = []
        alerts: list[AnomalyAlert] = []
        seen_req: set[str] = set()
        for row_t in bad_rows.tolist():
            row = int(row_t)
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
            if not req_id or req_id in seen_req:
                continue
            if not self._passes_input_filter(req_id, req_idx, log=False):
                continue
            seen_req.add(req_id)
            finite_kind = _finite_kind_for_row(logits[row])
            alerts.append(
                AnomalyAlert(
                    anomaly_type=self.anomaly_type,
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
        return alerts


def _finite_kind_for_row(row: torch.Tensor) -> str:
    """Classify a non-finite logits row for report detail (ill_type stays NAN)."""
    try:
        if bool(torch.isnan(row).any().item()):
            return "nan"
        if bool(torch.isposinf(row).any().item()):
            return "pos_inf"
        if bool(torch.isneginf(row).any().item()):
            return "neg_inf"
        if bool((~torch.isfinite(row)).any().item()):
            return "non_finite"
    except Exception:
        return "non_finite"
    return "non_finite"
