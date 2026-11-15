#
# Copyright (c) 2026 SeaGaruda
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
#
"""CPU-side contract tests for the TopKPerRow aclnn operator.

These pin down the semantics that ``csrc/attention/top_k_per_row`` implements, so
the kernel and its callers can be validated without an NPU:

    top_k_per_row(logits, row_starts, row_ends, topk) -> indices

For row ``r``: ``s = clamp(row_starts[r], 0, width)``,
``e = clamp(row_ends[r], 0, width)``, ``L = max(0, e - s)``:

* ``L >= topk``: ``topk`` column indices from ``[s, e)``, descending by value;
* ``L <  topk``: the sorted valid ones first, the rest ``-1``;
* ties resolve to the smaller column index;
* ``NaN`` ranks below every finite value and below ``-inf``.

The reference is implemented twice (vectorised numpy + plain loop) and the two
are cross-checked before being used as ground truth.
"""

import numpy as np
import pytest

TOPK_MAX = 2048


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else (hi if v > hi else v)


def top_k_per_row_numpy(logits, row_starts, row_ends, topk):
    logits = np.asarray(logits, dtype=np.float32)
    num_rows, width = logits.shape
    out = np.full((num_rows, topk), -1, dtype=np.int32)
    starts = np.clip(np.asarray(row_starts, dtype=np.int64), 0, width)
    ends = np.clip(np.asarray(row_ends, dtype=np.int64), 0, width)

    for r in range(num_rows):
        s, e = int(starts[r]), int(ends[r])
        if e <= s:
            continue
        row = logits[r, s:e].astype(np.float64)
        nan_mask = np.isnan(row)
        key = np.where(nan_mask, -np.inf, row)
        idx = np.arange(e - s, dtype=np.int64)
        order = np.lexsort((idx, -key))
        if nan_mask.any():
            order = np.concatenate([order[~nan_mask[order]], order[nan_mask[order]]])
        take = min(topk, order.size)
        out[r, :take] = (order[:take] + s).astype(np.int32)
    return out


def top_k_per_row_loop(logits, row_starts, row_ends, topk):
    logits = np.asarray(logits, dtype=np.float32)
    num_rows, width = logits.shape
    out = np.full((num_rows, topk), -1, dtype=np.int32)
    for r in range(num_rows):
        s = _clamp(int(row_starts[r]), 0, width)
        e = _clamp(int(row_ends[r]), 0, width)
        cand = []
        for c in range(s, e):
            v = float(logits[r, c])
            rank = (1, 0.0, c) if v != v else (0, -v, c)
            cand.append((rank, c))
        cand.sort()
        for i in range(min(topk, len(cand))):
            out[r, i] = cand[i][1]
    return out


def check_invariants(logits, row_starts, row_ends, topk, indices):
    logits = np.asarray(logits, dtype=np.float32)
    num_rows, width = logits.shape
    problems = []
    assert indices.shape == (num_rows, topk), f"{indices.shape} != {(num_rows, topk)}"
    assert indices.dtype == np.int32

    for r in range(num_rows):
        s = _clamp(int(row_starts[r]), 0, width)
        e = _clamp(int(row_ends[r]), 0, width)
        length = max(0, e - s)
        row = indices[r]
        n_valid = int((row >= 0).sum())
        if n_valid != min(topk, length):
            problems.append(f"row {r}: {n_valid} valid slots, expected {min(topk, length)}")
        if len(set(int(x) for x in row[:n_valid])) != n_valid:
            problems.append(f"row {r}: duplicate indices")
        for x in row[:n_valid]:
            if not (s <= int(x) < e):
                problems.append(f"row {r}: index {x} outside window [{s},{e})")
                break
        if n_valid and not np.all(row[n_valid:] == -1):
            problems.append(f"row {r}: padding is not -1")
        vals = [float(logits[r, int(x)]) for x in row[:n_valid]]
        for a in range(n_valid - 1):
            va, vb = vals[a], vals[a + 1]
            ia, ib = int(row[a]), int(row[a + 1])
            if va != va:
                continue
            if vb == vb and va < vb:
                problems.append(f"row {r}: not descending at {a}")
                break
            if vb == vb and va == vb and ia > ib:
                problems.append(f"row {r}: tie not index-ascending at {a}")
                break
        if 0 < n_valid == topk < length:
            sel = set(int(x) - s for x in row[:n_valid])
            rest = [logits[r, s + i] for i in range(length) if i not in sel]
            rest = np.asarray(rest, dtype=np.float32)
            rest = rest[~np.isnan(rest)]
            if rest.size and np.nanmax(rest) > vals[-1]:
                problems.append(f"row {r}: missed a larger value")
    return problems


def _ragged(num_rows, width, rng, empty_rows=0):
    starts = np.zeros(num_rows, dtype=np.int32)
    ends = np.zeros(num_rows, dtype=np.int32)
    for r in range(num_rows):
        s = int(rng.integers(0, width))
        e = int(rng.integers(s, width + 1))
        starts[r] = s
        ends[r] = e
    for r in rng.choice(num_rows, size=min(empty_rows, num_rows), replace=False):
        starts[r] = ends[r] = int(rng.integers(0, width + 1))
    return starts, ends


def test_reference_implementations_agree():
    rng = np.random.default_rng(0)
    for trial in range(30):
        rows = int(rng.integers(1, 9))
        width = int(rng.integers(1, 300))
        topk = int(rng.integers(1, min(TOPK_MAX, width) + 1))
        logits = rng.normal(size=(rows, width)).astype(np.float32)
        starts, ends = _ragged(rows, width, rng, empty_rows=1)
        a = top_k_per_row_numpy(logits, starts, ends, topk)
        b = top_k_per_row_loop(logits, starts, ends, topk)
        np.testing.assert_array_equal(a, b, err_msg=f"trial {trial}")
        assert check_invariants(logits, starts, ends, topk, a) == []


@pytest.mark.parametrize("topk", [1, 7, 32, 64, 199, 200])
def test_heavy_ties_resolve_by_index(topk):
    rng = np.random.default_rng(1)
    logits = rng.integers(0, 4, size=(6, 200)).astype(np.float32)
    starts = np.zeros(6, dtype=np.int32)
    ends = np.full(6, 200, dtype=np.int32)
    got = top_k_per_row_numpy(logits, starts, ends, topk)
    np.testing.assert_array_equal(got, top_k_per_row_loop(logits, starts, ends, topk))
    assert check_invariants(logits, starts, ends, topk, got) == []
    for r in range(6):
        vals = logits[r, got[r]]
        assert np.all(np.diff(vals) <= 0)
        for i in range(topk - 1):
            if vals[i] == vals[i + 1]:
                assert got[r, i] < got[r, i + 1]


def test_special_values():
    logits = np.array(
        [
            [np.nan, 1.0, -np.inf, np.inf, 0.0, -0.0],
            [-0.0, 0.0, np.nan, np.nan, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    starts = np.zeros(2, dtype=np.int32)
    ends = np.full(2, 6, dtype=np.int32)
    for topk in (1, 3, 6):
        np.testing.assert_array_equal(
            top_k_per_row_numpy(logits, starts, ends, topk),
            top_k_per_row_loop(logits, starts, ends, topk),
        )
    out = top_k_per_row_numpy(logits, starts, ends, 6)
    assert out[0, 0] == 3  # +inf first
    assert out[0, -1] == 0  # NaN last
    assert out[1, 0] == 4


def test_short_window_pads_minus_one():
    logits = np.arange(24, dtype=np.float32).reshape(3, 8)
    starts = np.array([0, 2, 5], dtype=np.int32)
    ends = np.array([3, 4, 5], dtype=np.int32)  # lengths 3, 2, 0
    out = top_k_per_row_numpy(logits, starts, ends, 4)
    np.testing.assert_array_equal(out[0], [2, 1, 0, -1])
    np.testing.assert_array_equal(out[1], [3, 2, -1, -1])
    np.testing.assert_array_equal(out[2], [-1, -1, -1, -1])
    assert check_invariants(logits, starts, ends, 4, out) == []


def test_window_is_clamped():
    logits = np.arange(10, dtype=np.float32).reshape(1, 10)
    out = top_k_per_row_numpy(logits, np.array([-5], np.int32), np.array([99], np.int32), 3)
    np.testing.assert_array_equal(out[0], [9, 8, 7])


def test_full_width_selection_is_reversed_range():
    logits = np.arange(64, dtype=np.float32).reshape(1, 64)
    out = top_k_per_row_numpy(logits, np.zeros(1, np.int32), np.full(1, 64, np.int32), 64)
    np.testing.assert_array_equal(out[0], np.arange(63, -1, -1))


def test_single_column_rows():
    logits = np.full((17, 1), 2.5, dtype=np.float32)
    out = top_k_per_row_numpy(logits, np.zeros(17, np.int32), np.ones(17, np.int32), 5)
    assert np.all(out[:, 0] == 0)
    assert np.all(out[:, 1:] == -1)
