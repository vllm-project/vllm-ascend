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

"""Load the GIL-releasing rc4 feed and adapt ``RareDetector`` tables to it.

Feature ids and GB input kinds are the enums in ``rare_feed_native.c``.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np

from .rare import LN10, WIN, describe
from .rare_vocab import CI
from .result import Result

# Rule features. Must match FT_* in rare_feed_native.c.
_FEAT = {
    "lp1": 0,
    "lpc": 1,
    "gap": 2,
    "fs": 3,
    "fsw": 4,
    "caseless_after_latin": 5,
    "bad_utf8": 6,
    "gb": 7,
    "answer": 8,
}
# GB input columns, in the order stored on the model. Must match IN_* .
_INPUT = {
    "lp1": 0,
    "lpc": 1,
    "gap": 2,
    "cls": 3,
    "pcls": 4,
    "ncls": 5,
    "first": 6,
    "last": 7,
    "lead": 8,
    "pcat": 9,
    "ncat": 10,
    "fs": 11,
    "fsw": 12,
    "answer": 13,
    "bad_utf8": 14,
}

_MOD = None
_MOD_TRIED = False


def _module():
    """Load the compiled extension built by setup.py / CMake.

    Prefer the installed package module (setuptools Extension or cmake
    ``rare_feed_native`` target). Fall back to a sidecar ``rare_feed_native.so``
    for hand-built / editable copies next to this file.
    """
    global _MOD, _MOD_TRIED
    if _MOD_TRIED:
        return _MOD
    _MOD_TRIED = True

    pkg_name = "vllm_ascend.observability.runtime_guard.detector.rare.rare_feed_native"
    try:
        _MOD = importlib.import_module(pkg_name)
        return _MOD
    except Exception:
        pass

    here = Path(__file__).resolve().parent
    candidates = [here / "rare_feed_native.so", *sorted(here.glob("rare_feed_native*.so"))]
    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("rare_feed_native", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            continue
        _MOD = module
        return module
    return None


def _log(msg: str) -> None:
    try:
        from vllm_ascend.logger import init_logger_ascend

        init_logger_ascend(__name__).info("%s", msg)
    except Exception:
        return


def _literal_fffd(boff: np.ndarray, bflat: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, np.uint8)
    for i in range(n):
        raw = bflat[int(boff[i]) : int(boff[i + 1])].tobytes()
        out[i] = raw.decode("utf-8", "ignore").count("\ufffd")
    return out


class NativeFeed:
    """Immutable tables plus the compiled extension."""

    def __init__(self, detector) -> None:
        """Copy vocab, rule, and GB tables into buffers the extension can hold."""
        module = _module()
        if module is None:
            raise RuntimeError("rare_feed_native.so is not built")
        gbm = detector.gbm
        if gbm is None:
            raise RuntimeError("native feed requires the rc4 gb model")
        table = detector.T
        ref = detector.ref
        if ref is None or "lcat_s" not in ref:
            raise RuntimeError("native feed requires category tables")
        rcls = np.ascontiguousarray(table["rcls"], np.int8)
        n = int(rcls.shape[0])
        boff = np.ascontiguousarray(table["boff"], np.int64)
        bflat = np.ascontiguousarray(table["bflat"], np.uint8)
        if boff.shape[0] != n + 1:
            raise RuntimeError("boff must have length vocab + 1")
        lcat = np.ascontiguousarray(ref["lcat_s"], np.int64)
        fcat = np.ascontiguousarray(ref["fcat_s"], np.int64)
        if lcat.shape[0] != n + 1 or fcat.shape[0] != n + 1:
            raise RuntimeError("category sentinel must have length vocab + 1")
        is_letter = np.zeros(len(CI), np.uint8)
        from .rare import _LET

        is_letter[np.asarray(_LET, bool)] = 1
        in_kind = np.array([_INPUT[name] for name in gbm.inputs], np.int8)
        atoms_f: list[int] = []
        atoms_k: list[int] = []
        atoms_t: list[float] = []
        begins: list[int] = []
        ends: list[int] = []
        strict: list[int] = []
        for rule, is_strict in zip(detector.compiled, detector.strict):
            begins.append(len(atoms_f))
            for _key, feat, kind, arg in rule:
                if feat not in _FEAT:
                    raise RuntimeError(f"native feed has no feature {feat}")
                if kind == "lt":
                    k, thr = 0, float(arg)
                elif kind == "gt":
                    k, thr = 1, float(arg)
                elif kind == "bool":
                    k, thr = 2, 0.0
                else:
                    raise RuntimeError(f"native feed has no atom kind {kind}")
                atoms_f.append(_FEAT[feat])
                atoms_k.append(k)
                atoms_t.append(thr)
            ends.append(len(atoms_f))
            strict.append(1 if is_strict else 0)
        if not atoms_f:
            raise RuntimeError("native feed requires at least one rule atom")
        qcat = np.zeros(len(gbm._qfeat), np.uint8)
        qcmask = np.zeros(len(gbm._qfeat), np.int64)
        qcat[np.asarray(gbm._qcat, np.int64)] = 1
        qcmask[np.asarray(gbm._qcat, np.int64)] = np.asarray(gbm._qcmask, np.int64)
        qstart = np.concatenate([np.asarray(gbm._qstart, np.int64), np.array([len(gbm._qfeat)], np.int64)])
        self._keep = {
            "rcls": rcls,
            "first": np.ascontiguousarray(table["first"], np.int8),
            "last": np.ascontiguousarray(table["last"], np.int8),
            "lead": np.ascontiguousarray(table["lead"], np.uint8),
            "u8": np.ascontiguousarray(table["u8"], np.uint8),
            "boff": boff,
            "bflat": bflat,
            "lit": _literal_fffd(boff, bflat, n),
            "lcat": lcat,
            "fcat": fcat,
            "is_letter": is_letter,
            "in_kind": in_kind,
            "qfeat": np.ascontiguousarray(gbm._qfeat, np.int32),
            "qthr": np.ascontiguousarray(gbm._qthr, np.float64),
            "qcat": qcat,
            "qcmask": qcmask,
            "qrule": np.ascontiguousarray(gbm._qrule, np.int64),
            "qstart": np.ascontiguousarray(qstart, np.int32),
            "qleaf": np.ascontiguousarray(gbm._qleaf, np.float64),
            "qleaf0": np.ascontiguousarray(gbm._qleaf0, np.int32),
            "atom_feat": np.asarray(atoms_f, np.int16),
            "atom_kind": np.asarray(atoms_k, np.int8),
            "atom_thr": np.asarray(atoms_t, np.float64),
            "rule_begin": np.asarray(begins, np.int32),
            "rule_end": np.asarray(ends, np.int32),
            "rule_strict": np.asarray(strict, np.uint8),
        }
        think = detector.think_end
        self._ctx = module.make_ctx(
            n,
            len(CI),
            int(in_kind.shape[0]),
            int(self._keep["qfeat"].shape[0]),
            int(gbm._qleaf0.shape[0]),
            int(self._keep["atom_feat"].shape[0]),
            int(self._keep["rule_begin"].shape[0]),
            1 if think is None else 0,
            0 if think is None else int(think),
            float(LN10),
            float(gbm.quant),
            float(gbm.base),
            0.05,
            self._keep["rcls"],
            self._keep["first"],
            self._keep["last"],
            self._keep["lead"],
            self._keep["u8"],
            self._keep["boff"],
            self._keep["bflat"],
            self._keep["lit"],
            self._keep["lcat"],
            self._keep["fcat"],
            self._keep["is_letter"],
            self._keep["in_kind"],
            self._keep["qfeat"],
            self._keep["qthr"],
            self._keep["qcat"],
            self._keep["qcmask"],
            self._keep["qrule"],
            self._keep["qstart"],
            self._keep["qleaf"],
            self._keep["qleaf0"],
            self._keep["atom_feat"],
            self._keep["atom_kind"],
            self._keep["atom_thr"],
            self._keep["rule_begin"],
            self._keep["rule_end"],
            self._keep["rule_strict"],
        )
        self._mod = module
        self._rules = detector.rules
        self._bound = None
        self._bound_key: tuple[int, int] | None = None
        self._hit_cap = 0
        self._hit_row = np.empty(0, np.int32)
        self._hit_mon = np.empty(0, np.int16)
        self._hit_strict = np.empty(0, np.int16)
        self._hit_fired = np.empty(0, np.int64)
        if WIN != 64:
            raise RuntimeError("native feed window is 64")

    def _ensure_hits(self, n: int) -> None:
        if self._hit_cap >= n:
            return
        cap = max(n, self._hit_cap * 2, 8)
        self._hit_row = np.empty(cap, np.int32)
        self._hit_mon = np.empty(cap, np.int16)
        self._hit_strict = np.empty(cap, np.int16)
        self._hit_fired = np.empty(cap, np.int64)
        self._hit_cap = cap

    def _ensure_bound(self, batch):
        """Pin row-state buffers until the batch storage grows."""
        key = (int(batch._letters.shape[0]), id(batch._counts))
        if self._bound is not None and self._bound_key == key:
            return self._bound
        bind = getattr(self._mod, "bind_state", None)
        if bind is None:
            return None
        self._bound = bind(
            self._ctx,
            key[0],
            batch._counts,
            batch._letters,
            batch._prev_last,
            batch._answer,
            batch._ring,
            batch._ring_len,
            batch._dec_n,
            batch._dec_b,
            batch._prev,
            batch._has_hold,
            batch._hold_id,
            batch._hold_lp1,
            batch._hold_lpc,
            batch._t,
            batch._done,
            batch._monitor_set,
            batch._tokens_seen,
            batch._recent_len,
            batch._recent_tok,
            batch._recent_lp1,
            batch._recent_lpc,
        )
        self._bound_key = key
        return self._bound

    def feed(self, batch, rows: np.ndarray, token_ids: np.ndarray, top1: np.ndarray, chosen: np.ndarray) -> list[int]:
        return self._call(batch, rows, token_ids, top1, chosen, finish=0)

    def finish(self, batch, row: int) -> None:
        self._call(
            batch,
            np.asarray([row], np.int64),
            np.asarray([0], np.int64),
            np.asarray([0.0], np.float64),
            np.asarray([0.0], np.float64),
            finish=1,
        )

    def _call(self, batch, rows, token_ids, top1, chosen, finish: int) -> list[int]:
        rows = np.ascontiguousarray(rows, np.int64)
        token_ids = np.ascontiguousarray(token_ids, np.int64)
        top1 = np.ascontiguousarray(top1, np.float64)
        chosen = np.ascontiguousarray(chosen, np.float64)
        n = int(rows.shape[0])
        if n == 0:
            return []
        self._ensure_hits(n)
        bound = self._ensure_bound(batch)
        if bound is not None:
            n_hits = int(
                self._mod.feed_bound(
                    bound,
                    rows,
                    token_ids,
                    top1,
                    chosen,
                    self._hit_row,
                    self._hit_mon,
                    self._hit_strict,
                    self._hit_fired,
                    int(finish),
                )
            )
        else:
            n_hits = int(
                self._mod.feed(
                    self._ctx,
                    int(batch._letters.shape[0]),
                    batch._counts,
                    batch._letters,
                    batch._prev_last,
                    batch._answer,
                    batch._ring,
                    batch._ring_len,
                    batch._dec_n,
                    batch._dec_b,
                    batch._prev,
                    batch._has_hold,
                    batch._hold_id,
                    batch._hold_lp1,
                    batch._hold_lpc,
                    batch._t,
                    batch._done,
                    batch._monitor_set,
                    batch._tokens_seen,
                    batch._recent_len,
                    batch._recent_tok,
                    batch._recent_lp1,
                    batch._recent_lpc,
                    rows,
                    token_ids,
                    top1,
                    chosen,
                    int(finish),
                    self._hit_row,
                    self._hit_mon,
                    self._hit_strict,
                    self._hit_fired,
                )
            )
        hit_row = self._hit_row
        hit_mon = self._hit_mon
        hit_strict = self._hit_strict
        hit_fired = self._hit_fired
        for i in range(n_hits):
            row = int(hit_row[i])
            fired = int(hit_fired[i])
            mon = int(hit_mon[i])
            strict = int(hit_strict[i])
            if mon >= 0:
                batch._monitor[row] = Result(True, fired, describe(self._rules[mon]))
            if strict >= 0:
                batch._result[row] = Result(True, fired, describe(self._rules[strict]))
        return [int(hit_row[i]) for i in range(n_hits)]

    def new_id_map(self):
        return self._mod.map_new()

    def map_put(self, id_map, req_id: str, row: int) -> None:
        self._mod.map_put(id_map, req_id, int(row))

    def map_del(self, id_map, req_id: str) -> None:
        self._mod.map_del(id_map, req_id)

    def lookup(self, id_map, req_ids) -> np.ndarray:
        raw = self._mod.map_lookup(id_map, req_ids)
        if not raw:
            return np.zeros(0, np.int32)
        return np.frombuffer(raw, dtype=np.int32).copy()


def bind(detector) -> NativeFeed | None:
    """Return a native feed for ``detector``, or None when the extension is absent."""
    try:
        native = NativeFeed(detector)
    except Exception as exc:
        _log(f"[runtime_guard: rare_chars] native feed unavailable: {exc}")
        return None
    _log("[runtime_guard: rare_chars] native feed enabled")
    return native
