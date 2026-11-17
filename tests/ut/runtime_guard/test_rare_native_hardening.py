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

# mypy: ignore-errors
"""Hardening: corrupt npz/json, OOB tokens, empty feed, GB.load validation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from vllm_ascend.observability.runtime_guard.detector.rare import (
    CONFIG_RARE,
    RareDetector,
    RareRowBatch,
    build_from_token_bytes,
    rare_native,
    register_rare_table,
)
from vllm_ascend.observability.runtime_guard.detector.rare.rare_gb import GB


def _require_native() -> None:
    rare_native._MOD = None
    rare_native._MOD_TRIED = False
    if rare_native._module() is None:
        pytest.skip("rare_feed_native not built")


def _det() -> RareDetector:
    key = register_rare_table("ut_hard", build_from_token_bytes([b"a", b"\x80", b"hi", b" "]))
    return RareDetector(key, CONFIG_RARE, think_end=None)


def test_native_matches_numpy_bad_utf8_stream():
    _require_native()
    det = _det()
    toks = np.array([1, 1, 1], np.int64)
    lp = np.zeros(3, np.float64)
    st = det.start()
    batch = RareRowBatch(det)
    row = batch.add()
    for i in range(3):
        det.feed(st, toks[i : i + 1], lp[i : i + 1], lp[i : i + 1])
        batch.feed_rows(np.array([row]), toks[i : i + 1], lp[i : i + 1], lp[i : i + 1])
        br, sr = batch.strict_at(row), st.result
        assert (br is None) == (sr is None)
        if br is not None:
            assert (br.is_ill, br.fired_at, br.reason) == (sr.is_ill, sr.fired_at, sr.reason)
    det.finish(st)
    batch.finish_row(row)
    br, sr = batch.strict_at(row), st.result
    assert (br is None) == (sr is None)
    if br is not None:
        assert (br.is_ill, br.fired_at, br.reason) == (sr.is_ill, sr.fired_at, sr.reason)


def test_oob_token_id_no_crash():
    _require_native()
    det = _det()
    batch = RareRowBatch(det)
    row = batch.add()
    before = batch._native.oob_tokens
    batch.feed_rows(np.array([row]), np.array([10**9], np.int64), np.zeros(1), np.zeros(1))
    batch.feed_rows(np.array([row]), np.array([-5], np.int64), np.zeros(1), np.zeros(1))
    assert batch.strict_at(row) is None
    assert batch._native.oob_tokens == before + 2


def test_empty_feed_ok():
    _require_native()
    det = _det()
    batch = RareRowBatch(det)
    batch.feed_rows(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros(0))


def test_gb_load_missing_file():
    with pytest.raises(FileNotFoundError):
        GB.load("/nonexistent/rare_gb.npz")


def test_gb_load_corrupt_keys(tmp_path: Path):
    p = tmp_path / "bad.npz"
    np.savez(p, depth=np.array(3))
    with pytest.raises(ValueError, match="missing keys"):
        GB.load(str(p))


def test_json_missing_rules():
    key = register_rare_table("ut_json", build_from_token_bytes([b"a"]))
    cfg = {"gate": "fs", "ref": "cats", "gb": "rare_gb.npz"}
    with pytest.raises(ValueError, match="rules"):
        RareDetector(key, cfg, think_end=None)


def test_json_bad_atom():
    key = register_rare_table("ut_json_atom", build_from_token_bytes([b"a"]))
    cfg = {
        "gate": "fs",
        "ref": "cats",
        "gb": "rare_gb.npz",
        "rules": [{"and": [{"lt": -1.0}]}],
    }
    with pytest.raises(ValueError, match='needs "f"'):
        RareDetector(key, cfg, think_end=None)


def test_json_thr_must_be_number():
    key = register_rare_table("ut_json_thr", build_from_token_bytes([b"a"]))
    cfg = {
        "gate": "fs",
        "ref": "cats",
        "gb": "rare_gb.npz",
        "rules": [{"and": [{"f": "lpc", "lt": "nope"}]}],
    }
    with pytest.raises(ValueError, match="must be a number"):
        RareDetector(key, cfg, think_end=None)


def test_json_bool_thr_rejected():
    key = register_rare_table("ut_json_bool_thr", build_from_token_bytes([b"a"]))
    cfg = {
        "gate": "fs",
        "ref": "cats",
        "gb": "rare_gb.npz",
        "rules": [{"and": [{"f": "lpc", "lt": True}]}],
    }
    with pytest.raises(ValueError, match="must be a number"):
        RareDetector(key, cfg, think_end=None)


def test_gb_load_bad_shape(tmp_path: Path):
    p = tmp_path / "shape.npz"
    np.savez(
        p,
        depth=np.array(2),
        feat=np.zeros((1, 2), np.int64),  # depth=2 needs 3 internal nodes
        thr=np.zeros((1, 2), np.float64),
        cat=np.zeros((1, 2), bool),
        mask=np.zeros((1, 2), np.int64),
        leaf=np.zeros((1, 4), np.float64),
        base=np.array(0.0),
        inputs=np.array(["lp1"]),
    )
    with pytest.raises(ValueError, match="feat shape"):
        GB.load(str(p))


def test_package_data_ships_rare_gb_npz():
    """Wheel package_data must match tables/*.npz (not *.npy)."""
    from pathlib import Path

    setup = Path(__file__).resolve().parents[3] / "setup.py"
    text = setup.read_text(encoding="utf-8")
    assert '"tables/*.npz"' in text
    assert '"tables/*.npy"' not in text
    tables = Path(__file__).resolve().parents[3] / ("vllm_ascend/observability/runtime_guard/detector/rare/tables")
    assert list(tables.glob("*.npz")), "rare_gb.npz missing from source tree"


def test_repo_path_strips_parent_dirs(tmp_path: Path, monkeypatch):
    """Relative gb paths are basename-only under package tables/ (no traversal)."""
    from vllm_ascend.observability.runtime_guard.detector.rare import rare as rare_mod

    # Traversal-looking relative path must resolve under tables/<basename>.
    got = rare_mod._repo_path("../../etc/passwd")
    assert got.endswith(os.path.join("tables", "passwd")) or got.endswith("tables/passwd")
    assert "etc" not in Path(got).parts[-2:]


def test_native_gb_score_aligns_numpy_gate():
    """Native batch vs numpy RareDetector on a short stream (exact reason/fired_at)."""
    _require_native()
    det = _det()
    # Mix: normal ascii, bad utf-8 lead, short word.
    toks = np.array([0, 1, 2, 3, 0], np.int64)
    top1 = np.array([-1.0, -3.0, -0.1, -5.0, -2.0], np.float64)
    chosen = np.array([-1.2, -4.0, -0.2, -6.0, -2.5], np.float64)
    st = det.start()
    batch = RareRowBatch(det)
    row = batch.add()
    for i in range(len(toks)):
        det.feed(st, toks[i : i + 1], top1[i : i + 1], chosen[i : i + 1])
        batch.feed_rows(np.array([row]), toks[i : i + 1], top1[i : i + 1], chosen[i : i + 1])
        br, sr = batch.strict_at(row), st.result
        assert (br is None) == (sr is None)
        if br is not None:
            assert (br.is_ill, br.fired_at, br.reason) == (sr.is_ill, sr.fired_at, sr.reason)
        bm, sm = batch.monitor_at(row), det.monitor(st)
        assert (bm.is_ill, bm.fired_at, bm.reason) == (sm.is_ill, sm.fired_at, sm.reason)
    det.finish(st)
    batch.finish_row(row)
    br, sr = batch.strict_at(row), st.result
    assert (br is None) == (sr is None)
    if br is not None:
        assert (br.is_ill, br.fired_at, br.reason) == (sr.is_ill, sr.fired_at, sr.reason)
