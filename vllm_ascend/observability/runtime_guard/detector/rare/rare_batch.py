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

"""One new token from many requests, updated together via ``rare_feed_native``.

Row buffers match the C extension's ``bind_state`` / ``feed_bound`` layout.
``RareDetector.feed`` remains the scalar Python reference; this batch path
always requires the compiled extension.
"""

from __future__ import annotations

import numpy as np

from .rare import WIN
from .rare_native import bind
from .rare_vocab import CI
from .result import Result

_GROW = 8


class RareRowBatch:
    """Look-ahead rc4 state for many requests, one new token per ``feed_rows``.

    Row ``i`` matches ``RareDetector.feed`` of one token, then ``finish_row``
    matches ``RareDetector.finish``. A strict alert sets the row done; later
    tokens for that row are ignored.
    """

    def __init__(self, detector) -> None:
        if not detector.look:
            raise ValueError("RareRowBatch requires a look-ahead rc4 config")
        self.det = detector
        self._native = bind(detector)
        self._n = 0
        self._cap = 0
        self._free: list[int] = []
        ncls = len(CI)
        self._counts = np.zeros((0, ncls), np.int64)
        self._letters = np.zeros(0, np.int64)
        self._prev_last = np.zeros(0, np.int16)
        self._answer = np.zeros(0, bool)
        self._ring = np.zeros((0, WIN), np.int16)
        self._ring_len = np.zeros(0, np.int16)
        self._dec_n = np.zeros(0, np.uint8)
        self._dec_b = np.zeros((0, 4), np.uint8)
        self._prev = np.zeros(0, np.int64)
        self._has_hold = np.zeros(0, bool)
        self._hold_id = np.zeros(0, np.int64)
        self._hold_lp1 = np.zeros(0, np.float64)
        self._hold_lpc = np.zeros(0, np.float64)
        self._t = np.zeros(0, np.int64)
        self._tokens_seen = np.zeros(0, np.int64)
        self._recent_len = np.zeros(0, np.int16)
        self._recent_tok = np.zeros((0, 16), np.int64)
        self._recent_lp1 = np.zeros((0, 16), np.float64)
        self._recent_lpc = np.zeros((0, 16), np.float64)
        self._done = np.zeros(0, bool)
        self._monitor_set = np.zeros(0, bool)
        self._result: list[Result | None] = []
        self._monitor: list[Result | None] = []
        self._id_map = None
        self._req_at: list[str | None] = []

    def add(self) -> int:
        """Return a row index with fresh per-request state."""
        if self._free:
            row = self._free.pop()
            self.reset_row(row)
            return row
        row = self._n
        self._n += 1
        self._grow(self._n)
        self.reset_row(row)
        return row

    def release(self, row: int) -> None:
        """Return a finished request's row to the free list."""
        self.reset_row(row)
        self._free.append(row)

    def reset_row(self, row: int) -> None:
        self._counts[row] = 0
        self._letters[row] = 0
        self._prev_last[row] = 0
        self._answer[row] = False
        self._ring[row] = 0
        self._ring_len[row] = 0
        self._dec_n[row] = 0
        self._dec_b[row] = 0
        self._prev[row] = -1
        self._has_hold[row] = False
        self._hold_id[row] = -1
        self._hold_lp1[row] = 0.0
        self._hold_lpc[row] = 0.0
        self._t[row] = 0
        self._tokens_seen[row] = 0
        self._recent_len[row] = 0
        self._done[row] = False
        self._monitor_set[row] = False
        self._result[row] = None
        self._monitor[row] = None

    def strict_at(self, row: int) -> Result | None:
        return self._result[row]

    def monitor_at(self, row: int) -> Result:
        mon = self._monitor[row]
        return mon if mon is not None else Result()

    def feed_rows(
        self,
        rows: np.ndarray,
        token_ids: np.ndarray,
        top1: np.ndarray,
        chosen: np.ndarray,
    ) -> None:
        """Feed one new token into each row. Same effect as ``feed`` of length 1."""
        self._native.feed(self, rows, token_ids, top1, chosen)

    def finish_row(self, row: int) -> Result:
        """Decide the held token with no right neighbour. Matches ``finish``."""
        if self._done[row]:
            got = self._result[row]
            return got if got is not None else Result()
        self._native.finish(self, row)
        got = self._result[row]
        return got if got is not None else Result()

    def _grow(self, need: int) -> None:
        if need <= self._cap:
            return
        cap = max(need, self._cap * 2, _GROW)
        ncls = len(CI)

        def _vec(old: np.ndarray, shape: tuple[int, ...], dtype) -> np.ndarray:
            out = np.zeros(shape, dtype)
            if self._cap:
                out[: self._cap] = old
            return out

        self._counts = _vec(self._counts, (cap, ncls), np.int64)
        self._letters = _vec(self._letters, (cap,), np.int64)
        self._prev_last = _vec(self._prev_last, (cap,), np.int16)
        self._answer = _vec(self._answer, (cap,), bool)
        self._ring = _vec(self._ring, (cap, WIN), np.int16)
        self._ring_len = _vec(self._ring_len, (cap,), np.int16)
        self._dec_n = _vec(self._dec_n, (cap,), np.uint8)
        self._dec_b = _vec(self._dec_b, (cap, 4), np.uint8)
        self._prev = _vec(self._prev, (cap,), np.int64)
        self._has_hold = _vec(self._has_hold, (cap,), bool)
        self._hold_id = _vec(self._hold_id, (cap,), np.int64)
        self._hold_lp1 = _vec(self._hold_lp1, (cap,), np.float64)
        self._hold_lpc = _vec(self._hold_lpc, (cap,), np.float64)
        self._t = _vec(self._t, (cap,), np.int64)
        self._tokens_seen = _vec(self._tokens_seen, (cap,), np.int64)
        self._recent_len = _vec(self._recent_len, (cap,), np.int16)
        self._recent_tok = _vec(self._recent_tok, (cap, 16), np.int64)
        self._recent_lp1 = _vec(self._recent_lp1, (cap, 16), np.float64)
        self._recent_lpc = _vec(self._recent_lpc, (cap, 16), np.float64)
        self._done = _vec(self._done, (cap,), bool)
        self._monitor_set = _vec(self._monitor_set, (cap,), bool)
        self._result.extend([None] * (cap - len(self._result)))
        self._monitor.extend([None] * (cap - len(self._monitor)))
        self._cap = cap
