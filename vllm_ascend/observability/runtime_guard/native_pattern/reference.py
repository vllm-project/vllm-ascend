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
# This file is a part of the vllm-ascend project.
#

"""Pure-Python gold for ``_pattern_native`` parity tests.

Not a production backend: never imported by ``PatternDetector``,
``token_repeat``, or package ``__init__``. Used only by
``tests/ut/runtime_guard/test_token_repeat_pattern.py``.
"""

from __future__ import annotations

import math
import operator
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from vllm_ascend.observability.runtime_guard.native_pattern.config import Result
from vllm_ascend.observability.runtime_guard.native_pattern.rules import load_pattern_rules

BLOCK = 64
H = 4096
RAW_KEEP = 6064
MASK = ~((1 << 24) - 1) & ((1 << 64) - 1)
HASH_BASE = 0x100000001B3
OP_GE = 0
OP_LE = 1
NUM_FEATURES = 9
MAX_ATOMS = 4
SEEDS = (0x51ED27, 0x9E3779, 0x7F4A7C, 0x2545F4)


def _u64(x: int) -> int:
    return x & ((1 << 64) - 1)


def mix(x: int) -> int:
    x = int(x)
    x = _u64(x + 0x9E3779B97F4A7C15)
    x = _u64((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9)
    x = _u64((x ^ (x >> 27)) * 0x94D049BB133111EB)
    return _u64(x ^ (x >> 31))


class Ring:
    __slots__ = ("a", "end")

    def __init__(self, cap: int):
        self.a = [0] * cap
        self.end = 0

    def push(self, x: int) -> None:
        self.a[self.end % len(self.a)] = x
        self.end += 1

    def at(self, i: int) -> int:
        base = max(0, self.end - len(self.a))
        if i < base or i >= self.end:
            raise IndexError("ring history")
        return self.a[i % len(self.a)]

    def add(self, x: int) -> None:
        self.push(self.at(self.end - 1) + x)

    def share(self, n: int, w: int) -> float:
        if n < w:
            return float("nan")
        return (self.at(n) - self.at(n - w)) / w


class Ngrams:
    __slots__ = (
        "ng",
        "m",
        "context",
        "cursor",
        "rolling",
        "outgoing_weight",
        "positions",
        "history",
        "counts",
    )

    def __init__(self, size: int):
        self.ng = size
        self.m = 0
        self.context = [0] * 8
        self.cursor = 0
        self.rolling = 0
        self.outgoing_weight = 1
        for _ in range(1, size):
            self.outgoing_weight = _u64(self.outgoing_weight * HASH_BASE)
        self.positions: dict[int, deque[int]] = {}
        self.history: deque[tuple[int, int]] = deque()
        self.counts = [Ring(1089) for _ in range(4)]
        for r in self.counts:
            r.push(0)

    def push(self, id_: int, answer: int) -> None:
        if self.m >= (1 << 24):
            raise OverflowError("pattern position limit 2^24")
        incoming = mix(id_)
        self.rolling = _u64(
            (self.rolling - _u64(self.context[self.cursor] * self.outgoing_weight)) * HASH_BASE + incoming
        )
        self.context[self.cursor] = incoming
        self.cursor = 0 if self.cursor + 1 == self.ng else self.cursor + 1
        count = d1 = d2 = 0
        if self.m >= self.ng - 1:
            while self.history and self.history[0][0] < self.m - H:
                pos, key = self.history.popleft()
                q = self.positions[key]
                if q[0] != pos:
                    raise RuntimeError("ngram eviction mismatch")
                q.popleft()
                if not q:
                    del self.positions[key]
            key = self.rolling & MASK
            q = self.positions.setdefault(key, deque())
            floor = max(self.m - H, 0)
            count = len(q)
            if answer > floor and self.m >= answer:
                count = len(q) - bisect_left(q, answer)
            if count >= 1:
                d1 = self.m - q[-1]
            if count >= 2:
                d2 = self.m - q[-2]
            q.append(self.m)
            self.history.append((self.m, key))
        self.counts[0].add(int(count >= 3))
        self.counts[1].add(int(count >= 5))
        self.counts[2].add(int(count >= 2 and abs(d2 - 2 * d1) <= (d1 >> 3)))
        self.counts[3].add(int(count == 0))
        self.m += 1


def exact(raw: Ring, nnew: int, pmin: int, pmax: int, copies: int) -> bool:
    end = raw.end
    base = max(0, end - len(raw.a))
    b0 = end - nnew
    hi = min(pmax, (end - base) // copies)
    if nnew <= 0 or hi < pmin:
        return False
    for p in range(pmin, hi + 1):
        need = (copies - 1) * p
        if (copies - 1) * pmin >= nnew:
            candidate = True
            for k in range(6):
                x = b0 - ((need - nnew) * k) // 5
                if x - p < base or raw.at(x) != raw.at(x - p):
                    candidate = False
                    break
            if not candidate:
                continue
        lo = max(b0 - need + 1, base + p)
        run = 0
        for q in range(lo, end):
            run = run + 1 if raw.at(q) == raw.at(q - p) else 0
            if q >= b0 and run >= need:
                return True
    return False


@dataclass
class _RefState:
    tables: object
    pending: list[int] = field(default_factory=list)
    t: int = 0
    nx: int = 0
    answer: int = -1
    fired_at: int = -1
    rule_idx: int = -1
    done: bool = False
    failed: bool = False
    n4: Ngrams = field(default_factory=lambda: Ngrams(4))
    n8: Ngrams = field(default_factory=lambda: Ngrams(8))
    raw: Ring = field(default_factory=lambda: Ring(RAW_KEEP))
    sep: Ring = field(default_factory=lambda: Ring(321))
    chars: Ring = field(default_factory=lambda: Ring(321))
    ws: Ring = field(default_factory=lambda: Ring(321))
    repeat: Ring = field(default_factory=lambda: Ring(2113))
    marker: Ring = field(default_factory=lambda: Ring(1089))
    duplicate: Ring = field(default_factory=lambda: Ring(81))
    seen: bytearray = field(default_factory=bytearray)
    run_char: int = -1
    run_len: int = 0
    run_max: int = 0
    sentences: int = 0
    sentence_len: int = 0
    signature: list[int] = field(default_factory=lambda: [((1 << 64) - 1)] * 4)
    sentence_history: list[list[int]] = field(default_factory=lambda: [[0] * 4 for _ in range(512)])
    streak: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.seen = bytearray(self.tables.norm_id_span // 8 + 1)
        self.streak = [0] * self.tables.num_rules
        for r in (self.sep, self.chars, self.ws, self.repeat, self.marker, self.duplicate):
            r.push(0)

    @property
    def received(self) -> int:
        if self.failed:
            raise RuntimeError("failed detector state; accepted prefix unknown")
        return self.t + len(self.pending)

    def close_sentence(self) -> None:
        if self.sentence_len >= 4:
            dup = False
            for i in range(min(self.sentences, 512)):
                agree = sum(self.sentence_history[i][k] == self.signature[k] for k in range(4))
                if agree >= 3:
                    dup = True
                    break
            self.sentence_history[self.sentences % 512] = list(self.signature)
            self.sentences += 1
            self.duplicate.add(int(dup))
        self.sentence_len = 0
        self.signature = [((1 << 64) - 1)] * 4

    def features(self) -> list[float]:
        if self.failed:
            raise RuntimeError("failed detector state; discard it")
        w = self.tables.feature_windows
        sep_w = w[8]
        return [
            self.n4.counts[0].share(self.nx, w[0]),
            self.n4.counts[1].share(self.nx, w[1]),
            self.n4.counts[3].share(self.nx, w[2]),
            self.n8.counts[2].share(self.nx, w[3]),
            self.repeat.share(self.nx, w[4]),
            self.repeat.share(self.nx, w[5]),
            self.marker.share(self.t, w[6]),
            self.duplicate.share(self.sentences, w[7]),
            float(self.sep.at(self.t) - self.sep.at(max(self.t - sep_w, 0))) / max(min(self.t, sep_w), 1),
        ]

    def block(self, n: int) -> None:
        tables = self.tables
        if self.answer < 0 and tables.think_end >= 0:
            before = 0
            for i in range(n):
                id_ = self.pending[i]
                if id_ == tables.think_end:
                    self.answer = self.nx + before
                    break
                before += tables.norm[id_] != -1
        for i in range(n):
            id_ = self.pending[i]
            norm = int(tables.norm[id_])
            ch = int(tables.chars[id_])
            rc = int(tables.run_char[id_])
            if norm != -1:
                self.n4.push(norm, self.answer)
                self.n8.push(norm, self.answer)
                mask = 1 << (norm & 7)
                self.repeat.add(int((self.seen[norm >> 3] & mask) != 0))
                self.seen[norm >> 3] |= mask
                self.nx += 1
                self.sentence_len += 1
                h = mix(norm)
                for k in range(4):
                    self.signature[k] = min(self.signature[k], _u64(h * (SEEDS[k] | 1) + SEEDS[k]))
                if self.sentence_len == 256:
                    self.close_sentence()
            if int(tables.sentence_end[id_]):
                self.close_sentence()
            self.raw.push(id_)
            self.sep.add(int(norm == -1))
            self.chars.add(ch)
            self.ws.add(int(tables.ws[id_]))
            self.marker.add(int(tables.marker[id_]))
            if rc < 0:
                self.run_char = -1
                self.run_len = 0
            elif rc == self.run_char:
                self.run_len += ch
            else:
                self.run_char = rc
                self.run_len = ch
            self.run_max = max(self.run_max, self.run_len)
        self.t += n
        f = [float(np.float32(v)) if not math.isnan(v) else float(np.float32(np.nan)) for v in self.features()]
        rules = tables
        n_rules = rules.num_rules
        ok = [False] * n_rules
        ok[0] = exact(self.raw, n, *rules.exact0)
        if not ok[0]:
            ok[1] = exact(self.raw, n, *rules.exact1)
        ok[2] = self.run_max >= rules.run_max_threshold
        for r, fr in enumerate(rules.feature):
            hold = fr["n_atoms"] > 0
            for a in range(fr["n_atoms"]):
                atom = fr["atoms"][a]
                v = f[atom["f"]]
                hold = (v >= atom["thr"]) if atom["op"] == OP_GE else (v <= atom["thr"])
                if not hold:
                    break
            ok[3 + r] = hold
        for i in range(n_rules):
            self.streak[i] = self.streak[i] + 1 if ok[i] else 0
            if self.streak[i] >= rules.persist[i]:
                self.rule_idx = i
                self.fired_at = self.t
                self.done = True
                break

    def compute_pending(self) -> None:
        try:
            self.block(len(self.pending))
            self.pending.clear()
        except Exception:
            self.failed = True
            raise

    def push(self, token: int) -> None:
        if self.failed:
            raise RuntimeError("failed detector state; discard it")
        if self.done:
            return
        if token < 0 or token > np.iinfo(np.int32).max or token >= len(self.tables.norm):
            raise IndexError("token id")
        self.pending.append(int(token))
        if len(self.pending) == BLOCK:
            self.compute_pending()


class _Tables:
    __slots__ = (
        "norm",
        "chars",
        "ws",
        "run_char",
        "marker",
        "sentence_end",
        "think_end",
        "norm_id_span",
        "exact0",
        "exact1",
        "run_max_threshold",
        "feature_windows",
        "feature",
        "persist",
        "num_rules",
        "reasons",
        "rules",
    )

    def __init__(self, tables, think_end, rules):
        self.norm = np.asarray(tables["norm"], dtype=np.int32)
        self.chars = np.asarray(tables["chars"], dtype=np.int32)
        self.ws = np.asarray(tables["ws"], dtype=np.int32)
        self.run_char = np.asarray(tables["run_char"], dtype=np.int32)
        self.marker = np.asarray(tables["marker"], dtype=np.int32)
        self.sentence_end = np.asarray(tables["sentence_end"], dtype=np.int32)
        self.think_end = -1 if think_end is None else int(think_end)
        self.norm_id_span = int(self.norm.max()) + 1
        ep = rules["exact_params"]
        self.exact0 = (int(ep[0, 0]), int(ep[0, 1]), int(ep[0, 2]))
        self.exact1 = (int(ep[1, 0]), int(ep[1, 1]), int(ep[1, 2]))
        self.run_max_threshold = int(rules["run_max_threshold"])
        self.feature_windows = [int(x) for x in rules["feature_windows"]]
        meta = rules["atom_meta"]
        thr = rules["atom_thr"]
        self.feature = []
        for r in range(meta.shape[0]):
            n_atoms = int(meta[r, 0])
            atoms = []
            for a in range(n_atoms):
                atoms.append(
                    {
                        "f": int(meta[r, 1 + a * 2]),
                        "op": int(meta[r, 1 + a * 2 + 1]),
                        "thr": float(thr[r, a]),
                    }
                )
            self.feature.append({"n_atoms": n_atoms, "atoms": atoms})
        self.persist = [int(x) for x in rules["persist"]]
        self.num_rules = len(self.persist)
        self.reasons = rules["reasons"]
        self.rules = rules["rules"]


class ReferenceDetector:
    """Python gold with the same public surface as ``PatternDetector``."""

    def __init__(self, tables, think_end=None, *, rules_path: str | None = None):
        if think_end is not None:
            if isinstance(think_end, (bool, np.bool_)):
                raise TypeError("think_end must be an integer token ID or None")
            think_end = operator.index(think_end)
        compiled = load_pattern_rules(rules_path)
        self.reasons = compiled["reasons"]
        self.rules = compiled["rules"]
        self._tables = _Tables(tables, think_end, compiled)

    def start(self):
        return _RefState(self._tables)

    def result(self, state):
        if state.failed:
            raise RuntimeError("failed detector state; discard it")
        if state.rule_idx < 0:
            return None
        return Result(True, state.fired_at, self.reasons[state.rule_idx], self.rules[state.rule_idx])

    def progress(self, state):
        if state.failed:
            return None
        reason = self.reasons[state.rule_idx] if state.rule_idx >= 0 else ""
        rule = self.rules[state.rule_idx] if state.rule_idx >= 0 else ""
        return (state.received, state.t, state.fired_at, reason, rule)

    def feed_events(self, states, token_lists):
        if len(states) != len(token_lists):
            raise ValueError("batch sizes")
        if len(set(id(s) for s in states)) != len(states):
            raise ValueError("duplicate state in feed_events")
        for state in states:
            if state is None:
                raise ValueError("state must not be None")
            if state.tables is not self._tables:
                raise ValueError("state belongs to another detector")
            if state.failed:
                raise RuntimeError("failed detector state; discard it")
        events = []
        for slot, (state, tokens) in enumerate(zip(states, token_lists)):
            was_done = state.done
            if not state.done:
                try:
                    for token in tokens:
                        state.push(token)
                        if state.done:
                            break
                except Exception:
                    state.failed = True
                    raise
            if not was_done and state.fired_at >= 0:
                events.append((slot, state.fired_at, self.reasons[state.rule_idx], self.rules[state.rule_idx]))
        return events

    def finish(self, state):
        if state.tables is not self._tables:
            raise ValueError("state belongs to another detector")
        if state.failed:
            raise RuntimeError("failed detector state; discard it")
        if not state.done and state.pending:
            state.compute_pending()
        state.pending.clear()
        state.done = True
        return self.result(state) or Result()
