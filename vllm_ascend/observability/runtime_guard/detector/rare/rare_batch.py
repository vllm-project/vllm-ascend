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

"""One new token from many requests, updated together.

``RareDetector.feed`` vectorizes along one request's token axis. This batch
vectorizes along the request axis: each row keeps its own counts, UTF-8
decoder, look-ahead hold and letter ring, and one step scores every row's
held token with ``holds`` / ``GB.score`` on arrays of length ``rows``.
"""

from __future__ import annotations

import codecs

import numpy as np

from .rare import LN10, WIN, Lazy, _LET, _bad_utf8, _ctx_features, describe, holds
from .rare_vocab import CI
from .result import Result

_GROW = 8


def _eager(values: np.ndarray) -> Lazy:
    """Lazy view of an already-computed per-row array."""
    return Lazy(lambda idx, values=values: values[idx], len(values))


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
        self._pending = np.zeros(0, bool)
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
        self._decs: list = []
        self._result: list[Result | None] = []
        self._monitor: list[Result | None] = []
        self._native = None
        self._id_map = None
        self._req_at: list[str | None] = []
        try:
            from .rare_native import bind

            self._native = bind(detector)
        except Exception:
            self._native = None

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
        self._pending[row] = False
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
        self._decs[row] = codecs.getincrementaldecoder("utf-8")(errors="replace")
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
        if self._native is not None:
            self._native.feed(self, rows, token_ids, top1, chosen)
            return
        rows = np.asarray(rows, np.int64)
        token_ids = np.asarray(token_ids, np.int64)
        top1 = np.asarray(top1, np.float64)
        chosen = np.asarray(chosen, np.float64)
        if len(rows) == 0:
            return
        if not (len(rows) == len(token_ids) == len(top1) == len(chosen)):
            raise ValueError("feed_rows arguments must have the same length")
        alive = ~self._done[rows]
        if not alive.any():
            return
        rows = rows[alive]
        token_ids = token_ids[alive]
        top1 = top1[alive]
        chosen = chosen[alive]
        has = self._has_hold[rows]
        if has.any():
            deciding = rows[has]
            # Decide the held token; the new id is only its right neighbour.
            self._decide(deciding, token_ids[has], lag=1)
            self._prev[deciding] = self._hold_id[deciding]
            self._hold_id[deciding] = token_ids[has]
            self._hold_lp1[deciding] = top1[has]
            self._hold_lpc[deciding] = chosen[has]
        fresh = ~has
        if fresh.any():
            first = rows[fresh]
            self._hold_id[first] = token_ids[fresh]
            self._hold_lp1[first] = top1[fresh]
            self._hold_lpc[first] = chosen[fresh]
            self._has_hold[first] = True

    def finish_row(self, row: int) -> Result:
        """Decide the held token with no right neighbour. Matches ``finish``."""
        if self._done[row]:
            got = self._result[row]
            return got if got is not None else Result()
        if self._native is not None:
            self._native.finish(self, row)
            got = self._result[row]
            return got if got is not None else Result()
        if self._has_hold[row]:
            self._decide(np.array([row], np.int64), np.array([-1], np.int64), lag=0)
            self._has_hold[row] = False
        self._done[row] = True
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
                # Row axis grows; class / window axes stay the same width.
                out[: self._cap] = old
            return out

        self._counts = _vec(self._counts, (cap, ncls), np.int64)
        self._letters = _vec(self._letters, (cap,), np.int64)
        self._prev_last = _vec(self._prev_last, (cap,), np.int16)
        self._answer = _vec(self._answer, (cap,), bool)
        self._ring = _vec(self._ring, (cap, WIN), np.int16)
        self._ring_len = _vec(self._ring_len, (cap,), np.int16)
        self._pending = _vec(self._pending, (cap,), bool)
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
        self._decs.extend([None] * (cap - len(self._decs)))
        self._result.extend([None] * (cap - len(self._result)))
        self._monitor.extend([None] * (cap - len(self._monitor)))
        self._cap = cap

    def _decide(self, rows: np.ndarray, nxt: np.ndarray, lag: int) -> None:
        """Score each row's held token. ``nxt`` is that token's right neighbour."""
        det = self.det
        table = det.T
        tok = self._hold_id[rows]
        top1 = self._hold_lp1[rows]
        chosen = self._hold_lpc[rows]
        prv = self._prev[rows]
        m = len(rows)
        cls = table["rcls"][tok]
        isl = _LET[cls]
        fs = np.ones(m, np.float64)
        if isl.any():
            picked = rows[isl]
            cls_i = cls[isl].astype(np.int64, copy=False)
            prev_c = self._counts[picked, cls_i]
            fs[isl] = (prev_c + 1.0) / (self._letters[picked] + 1.0)
            np.add.at(self._counts, (picked, cls_i), 1)
            self._letters[picked] += 1
        first = table["first"][tok]
        last = table["last"][tok]
        lead = np.asarray(table["lead"][tok], dtype=bool)
        prev_last = self._prev_last[rows]
        let = (prev_last == 1) | (prev_last == 2)
        caseless = (first == 5) & ~lead & let
        self._prev_last[rows] = last
        answer = self._answer[rows].copy()
        think_end = det.think_end
        if think_end is None:
            answer[:] = True
            self._answer[rows] = True
        else:
            hit = ~answer & (tok == think_end)
            if hit.any():
                answer[hit] = True
                self._answer[rows[hit]] = True
        bad = self._bad_utf8_rows(rows, tok)
        fsw = self._fsw_and_push(rows, cls, isl)
        lp1 = top1 / LN10
        lpc = chosen / LN10
        features = {
            "lp1": lp1,
            "lpc": lpc,
            "fs": fs,
            "bad_utf8": bad,
            "caseless_after_latin": caseless,
            "answer": answer,
            "gap": lp1 - lpc,
            "fsw": _eager(fsw),
        }
        _ctx_features(features, tok, prv, nxt, table, det.ref, det.gbm, det.need)
        self._apply_rules(rows, features, lag)

    def _bad_utf8_rows(self, rows: np.ndarray, tok: np.ndarray) -> np.ndarray:
        """``_bad_utf8`` per row. Complete UTF-8 with an empty decoder stays False."""
        u8 = self.det.T["u8"][tok]
        need = np.flatnonzero(u8 | self._pending[rows])
        bad = np.zeros(len(rows), dtype=bool)
        if len(need) == 0:
            return bad
        table = self.det.T
        for local in need.tolist():
            row = int(rows[local])
            one = tok[local : local + 1]
            bad[local] = bool(_bad_utf8(self._decs[row], one, table)[0])
            self._pending[row] = bool(self._decs[row].getstate()[0])
        return bad

    def _fsw_and_push(self, rows: np.ndarray, cls: np.ndarray, isl: np.ndarray) -> np.ndarray:
        """Letter-class share over the last ``WIN`` letters, then append this letter.

        The share is the ring from before this token, matching ``_window_share``.
        """
        m = len(rows)
        fsw = np.ones(m, np.float64)
        if not isl.any():
            return fsw
        ring_len = self._ring_len[rows]
        hist = self._ring[rows]
        valid = np.arange(WIN)[None, :] < ring_len[:, None]
        cnt = ((hist == cls.astype(np.int16)[:, None]) & valid).sum(axis=1)
        fsw[isl] = (cnt[isl] + 1.0) / (ring_len[isl] + 1.0)
        sub = hist.copy()
        full = isl & (ring_len == WIN)
        if full.any():
            sub[full, :-1] = sub[full, 1:]
            sub[full, -1] = cls[full]
        room = isl & ~full
        if room.any():
            sub[room, ring_len[room]] = cls[room]
        self._ring[rows] = sub
        self._ring_len[rows[isl]] = np.minimum(ring_len[isl] + 1, WIN)
        return fsw

    def _apply_rules(self, rows: np.ndarray, features: dict, lag: int) -> None:
        """Length-1 ``_step`` for every row: earliest rule wins, then strict rules."""
        det = self.det
        m = len(rows)
        memo: dict = {}
        mon_rule = np.full(m, -1, np.int16)
        strict_rule = np.full(m, -1, np.int16)
        already = self._monitor_set[rows]
        for i, rule in enumerate(det.compiled):
            if det.has_gb[i] and det.gbm is not None and features["gb"].nothing():
                continue
            hit = holds(rule, features, memo)
            if not det.strict[i]:
                hit = hit & ~already
            if not hit.any():
                continue
            # Rules run in order, so the first hit is the earliest rule index.
            take_mon = hit & (mon_rule < 0) & ~already
            mon_rule[take_mon] = i
            if det.strict[i]:
                take_strict = hit & (strict_rule < 0)
                strict_rule[take_strict] = i
        fired_at = self._t[rows] + 1 + lag
        self._t[rows] += 1
        for local in np.flatnonzero(mon_rule >= 0).tolist():
            row = int(rows[local])
            if self._monitor_set[row]:
                continue
            rule_i = int(mon_rule[local])
            self._monitor[row] = Result(True, int(fired_at[local]), describe(det.rules[rule_i]))
            self._monitor_set[row] = True
        for local in np.flatnonzero(strict_rule >= 0).tolist():
            row = int(rows[local])
            rule_i = int(strict_rule[local])
            self._result[row] = Result(True, int(fired_at[local]), describe(det.rules[rule_i]))
            self._done[row] = True
