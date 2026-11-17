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
    batch.feed_rows(np.array([row]), np.array([10**9], np.int64), np.zeros(1), np.zeros(1))
    batch.feed_rows(np.array([row]), np.array([-5], np.int64), np.zeros(1), np.zeros(1))
    assert batch.strict_at(row) is None


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
