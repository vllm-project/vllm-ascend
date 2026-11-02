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

"""When a request is handed to rc4.

``interval_pick`` keeps one new token and drops the rest of the gap.
``RowAccum`` keeps every token. A request is flushed once it has at least
``n`` buffered output tokens and at least ``step_gap`` steps have passed
since its previous flush. Flushed rows may have different lengths; the
native ``feed_spans`` call applies each row's tokens in order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Tokens that must sit between two checks. gap=1 selects a request only after
# two new tokens have arrived, and the token from the skipped step is not fed.
RARE_CHECK_GAP = 1
RARE_CHECK_BUDGET = 64


def interval_pick(
    req_ids: Sequence[str],
    pending: dict[str, int],
    abnormal: dict[str, bool],
    closed: dict[str, bool],
    *,
    gap: int,
    budget: int,
) -> list[int]:
    """Choose which requests to score this step.

    Each call counts one new token. A request is eligible once ``gap + 1`` tokens
    have arrived since its last check. Already-closed requests are skipped.
    Among the eligible, a request that was abnormal on its previous check comes
    first, then the one with more tokens since that check. At most ``budget``
    are returned, and their pending count is reset.
    """
    need = gap + 1
    eligible: list[int] = []
    for i, req_id in enumerate(req_ids):
        if not req_id or closed.get(req_id):
            continue
        seen = pending.get(req_id, 0) + 1
        pending[req_id] = seen
        if seen >= need:
            eligible.append(i)
    eligible.sort(key=lambda i: (not abnormal.get(req_ids[i], False), -pending[req_ids[i]], i))
    picked = eligible[: max(budget, 0)]
    for i in picked:
        pending[req_ids[i]] = 0
    return picked


class RowAccum:
    """Per-row output tokens waiting for one ragged flush.

    ``step_gap=1`` means a row flushed at step ``t`` is not flushed again at
    ``t + 1``. The first flush has no previous step, so it runs once ``n``
    tokens are buffered. A row whose buffer is full is flushed on that step
    even inside the gap, so a token is delayed, not dropped.
    """

    def __init__(self, n_rows: int, *, n: int = 3, step_gap: int = 1, width: int = 64) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if step_gap < 0:
            raise ValueError("step_gap must be >= 0")
        if width < 1:
            raise ValueError("width must be >= 1")
        self.n = int(n)
        self.step_gap = int(step_gap)
        self.width = int(width)
        self.length = np.zeros(n_rows, np.int32)
        self.last_step = np.full(n_rows, -1, np.int32)
        self.tok = np.zeros((n_rows, self.width), np.int64)
        self.top1 = np.zeros((n_rows, self.width), np.float64)
        self.chosen = np.zeros((n_rows, self.width), np.float64)

    def push(self, rows: np.ndarray, token_ids: np.ndarray, top1: np.ndarray, chosen: np.ndarray) -> None:
        """Append one new token on each row. The row must have a free slot."""
        rows = np.asarray(rows, np.int64)
        if rows.size == 0:
            return
        lens = self.length[rows]
        if np.any(lens >= self.width):
            raise RuntimeError("accum buffer is full; flush those rows before push")
        self.tok[rows, lens] = np.asarray(token_ids, np.int64)
        self.top1[rows, lens] = np.asarray(top1, np.float64)
        self.chosen[rows, lens] = np.asarray(chosen, np.float64)
        self.length[rows] = lens + 1

    def ready_rows(self, step: int) -> np.ndarray:
        """Rows to flush at ``step``: enough tokens and the step gap, or a full buffer."""
        gap_ok = (self.last_step < 0) | ((int(step) - self.last_step) >= (self.step_gap + 1))
        ready = (self.length >= self.n) & gap_ok
        full = self.length >= self.width
        return np.flatnonzero(ready | full).astype(np.int64, copy=False)

    def pending_rows(self) -> np.ndarray:
        """Rows that still hold tokens. Used to drain the tail at end of stream."""
        return np.flatnonzero(self.length > 0).astype(np.int64, copy=False)

    def mark_flushed(self, rows: np.ndarray, step: int) -> None:
        rows = np.asarray(rows, np.int64)
        if rows.size == 0:
            return
        self.length[rows] = 0
        self.last_step[rows] = int(step)
