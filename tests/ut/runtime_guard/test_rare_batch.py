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

from vllm_ascend.observability.runtime_guard.detector.rare import (
    CONFIG_RC4,
    RareDetector,
    RareRowBatch,
    build_from_token_bytes,
    register_rare_table,
)


def _table_key() -> str:
    chunks = [b"a", b"\x80", b"hi", b"ok", b" "]
    return register_rare_table("ut_batch", build_from_token_bytes(chunks))


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


def test_batch_matches_scalar_streams():
    key = _table_key()
    token_rows = [
        np.array([1, 1, 2], np.int64),
        np.array([0, 2], np.int64),
    ]
    top1_rows = [np.zeros(len(t), np.float64) for t in token_rows]
    chosen_rows = [np.zeros(len(t), np.float64) for t in token_rows]

    det = RareDetector(key, CONFIG_RC4, think_end=None)
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
            assert _batch_snapshot(det, batch, rows[i]) == _snapshot(det, scalars[i])
    for i, st in enumerate(scalars):
        det.finish(st)
        batch.finish_row(rows[i])
        assert _batch_snapshot(det, batch, rows[i]) == _snapshot(det, st)
