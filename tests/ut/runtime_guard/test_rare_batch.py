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

"""RareRowBatch matches RareDetector.feed one token at a time."""

from __future__ import annotations

import numpy as np

from vllm_ascend.observability.runtime_guard.detector.rare import CONFIG_RC4, RareDetector, RareRowBatch


def _snapshot(det: RareDetector, st) -> tuple:
    res = st.result
    mon = det.monitor(st)
    return (
        st.done,
        int(st.t),
        None if res is None else (res.is_ill, res.fired_at, res.reason),
        (mon.is_ill, mon.fired_at, mon.reason),
    )


def _batch_snapshot(det: RareDetector, batch: RareRowBatch, row: int) -> tuple:
    res = batch.strict_at(row)
    mon = batch.monitor_at(row)
    return (
        bool(batch._done[row]),
        int(batch._t[row]),
        None if res is None else (res.is_ill, res.fired_at, res.reason),
        (mon.is_ill, mon.fired_at, mon.reason),
    )


def _compare_streams(token_rows: list[np.ndarray], top1_rows: list[np.ndarray], chosen_rows: list[np.ndarray]) -> None:
    det = RareDetector("qwen3", CONFIG_RC4, think_end="auto")
    batch = RareRowBatch(det)
    scalars = [det.start() for _ in token_rows]
    rows = [batch.add() for _ in token_rows]
    steps = max(len(tok) for tok in token_rows)
    for step in range(steps):
        live_rows = []
        live_tok = []
        live_lp = []
        live_lc = []
        live_i = []
        for i, tok in enumerate(token_rows):
            if step >= len(tok) or scalars[i].done:
                continue
            one = tok[step : step + 1]
            det.feed(scalars[i], one, top1_rows[i][step : step + 1], chosen_rows[i][step : step + 1])
            live_i.append(i)
            live_rows.append(rows[i])
            live_tok.append(int(one[0]))
            live_lp.append(float(top1_rows[i][step]))
            live_lc.append(float(chosen_rows[i][step]))
        if live_rows:
            batch.feed_rows(
                np.asarray(live_rows, np.int64),
                np.asarray(live_tok, np.int64),
                np.asarray(live_lp, np.float64),
                np.asarray(live_lc, np.float64),
            )
        for i in live_i:
            assert _batch_snapshot(det, batch, rows[i]) == _snapshot(det, scalars[i]), (
                f"step {step} row {i}: batch {_batch_snapshot(det, batch, rows[i])} scalar {_snapshot(det, scalars[i])}"
            )
    for i, st in enumerate(scalars):
        det.finish(st)
        batch.finish_row(rows[i])
        assert _batch_snapshot(det, batch, rows[i]) == _snapshot(det, st), (
            f"finish row {i}: batch {_batch_snapshot(det, batch, rows[i])} scalar {_snapshot(det, st)}"
        )


def test_batch_matches_scalar_feed_on_mixed_streams():
    det = RareDetector("qwen3", CONFIG_RC4, think_end="auto")
    table = det.T
    cls = table["rcls"]
    u8 = table["u8"]
    latin = np.flatnonzero((cls == 1) & ~u8)[:80]
    others = [np.flatnonzero((cls == c) & ~u8)[:8] for c in range(len(set(cls.tolist()))) if c != 1]
    others = [pool for pool in others if len(pool)]
    partial = np.flatnonzero(u8)[:4]
    think = det.think_end
    streams = []
    top1s = []
    chosens = []
    # Same-class, mild logprobs: rules stay quiet, state still advances.
    streams.append(latin[:40])
    top1s.append(np.full(40, -0.2))
    chosens.append(np.full(40, -0.4))
    # Class changes every token so the learned-score gate stays open.
    mix = np.array([others[i % len(others)][i % len(others[i % len(others)])] for i in range(36)], np.int64)
    streams.append(mix)
    top1s.append(np.full(36, -0.1))
    chosens.append(np.full(36, -1.2))
    # Low chosen logprob, partial UTF-8 in the middle, then the think-end token.
    rough = np.concatenate([latin[:8], partial[:2], np.array([think], np.int64), latin[8:20]])
    streams.append(rough)
    top1s.append(np.full(len(rough), -0.05))
    chosens.append(np.linspace(-0.2, -8.0, len(rough)))
    _compare_streams(streams, top1s, chosens)


def test_released_row_matches_a_fresh_request():
    det = RareDetector("qwen3", CONFIG_RC4, think_end="auto")
    latin = np.flatnonzero((det.T["rcls"] == 1) & ~det.T["u8"])[:10]
    batch = RareRowBatch(det)
    row = batch.add()
    batch.feed_rows(
        np.array([row]),
        latin[:1],
        np.array([-0.2]),
        np.array([-0.4]),
    )
    batch.release(row)
    reused = batch.add()
    assert reused == row
    fresh = det.start()
    again = RareRowBatch(det)
    other = again.add()
    for i in range(6):
        tok = latin[i : i + 1]
        lp = np.array([-0.2])
        lc = np.array([-0.3])
        det.feed(fresh, tok, lp, lc)
        batch.feed_rows(np.array([reused]), tok, lp, lc)
        again.feed_rows(np.array([other]), tok, lp, lc)
        assert _batch_snapshot(det, batch, reused) == _snapshot(det, fresh)
        assert _batch_snapshot(det, batch, reused) == _batch_snapshot(det, again, other)
