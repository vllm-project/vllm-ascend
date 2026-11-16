# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023-2025 The vLLM team.
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
"""Engram SSD offload: reliable libaio row-read engine (native ABI v2).

Python wrapper of ``native/ssd_fill.c``. Contract (phase-3 docs/04 §8):
``fill`` returns only after every requested row's 256B codes + 32B scales
have been copied into the payload planes; any short read / IO error fails
the whole batch without writing the failed rows, and in-flight requests
are drained before returning. Completions are validated by batch
generation so stale events from a previous batch can never be applied.

The native library ships in the wheel via the ``engram_ssd_c`` CMake
target; no runtime compilation is performed.
"""

from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_NATIVE_DIR = _HERE / "native"
_LIB_NAME = "libengram_ssd.so"

_LIB_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None


class SSDError(Exception):
    """Base error for the SSD engine (code + native detail string)."""

    def __init__(self, code: int, detail: str):
        super().__init__(f"[code={code}] {detail}")
        self.code = code
        self.detail = detail


class SSDEngineError(SSDError):
    """open/close/lifecycle error."""


class SSDReadError(SSDError):
    """fill-time IO/short-read/range error."""


class SSDInputError(SSDError):
    """Caller-side input validation error (never enters native code)."""


class _CError(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int32), ("detail", ctypes.c_char * 256)]


class CExtent(ctypes.Structure):
    """Prefill merged-read extent (must mirror ssdf_extent in native)."""

    _fields_ = [("offset", ctypes.c_int64), ("file_index", ctypes.c_int32),
                ("length", ctypes.c_int32), ("scatter_begin", ctypes.c_int32),
                ("scatter_count", ctypes.c_int32)]


class CScatter(ctypes.Structure):
    """Row placement inside one extent (must mirror ssdf_scatter)."""

    _fields_ = [("rel_offset", ctypes.c_int32), ("target_slot", ctypes.c_int32)]


def plan_row_extents(rows, slots, *, base1: int = 0, dual: bool = False,
                     bmax: int = 16384):
    """Sort unique miss rows and greedily merge them into 4K-aligned extents.

    Returns ``(extents, scatters, order)`` where ``order`` sorts ``rows`` and
    ``slots[order]`` aligns with the scatter entries. Cross-block rows force
    their extent to cover the full span; merged extents never exceed ``bmax``.
    """
    order = np.argsort(rows, kind="stable")
    rows_s = rows[order]
    slots_s = slots[order]
    if dual:
        is_b = rows_s >= np.uint64(base1)
        local = np.where(is_b, rows_s - np.uint64(base1), rows_s)
    else:
        local = rows_s
    offs = local.astype(np.int64) * 288
    starts = (offs // 4096 * 4096).tolist()
    ends = ((offs + 288 + 4095) // 4096 * 4096).tolist()
    offs = offs.tolist()
    file_ids = (rows_s >= np.uint64(base1)).astype(np.int32).tolist() if dual \
        else [0] * len(rows_s)

    ext_list, sc_list = [], []
    cur_start = cur_end = cur_file = None
    cur_begin = 0
    for i in range(len(rows_s)):
        st, en, f = starts[i], ends[i], file_ids[i]
        openable = (cur_start is not None and cur_end >= st and cur_file == f
                    and (max(cur_end, en) - cur_start) <= bmax)
        if not openable:
            if cur_start is not None:
                ext_list.append((cur_start, cur_file, cur_end - cur_start,
                                 cur_begin, len(sc_list) - cur_begin))
            cur_start, cur_end, cur_file, cur_begin = st, en, f, len(sc_list)
        else:
            cur_end = max(cur_end, en)
        sc_list.append((offs[i] - cur_start, int(slots_s[i])))
    if cur_start is not None:
        ext_list.append((cur_start, cur_file, cur_end - cur_start,
                         cur_begin, len(sc_list) - cur_begin))

    ext_arr = (CExtent * len(ext_list))()
    for i, (o, f, ln, b, c) in enumerate(ext_list):
        ext_arr[i] = CExtent(o, f, ln, b, c)
    sct_arr = (CScatter * len(sc_list))()
    for i, (r, sp) in enumerate(sc_list):
        sct_arr[i] = CScatter(r, sp)
    return ext_arr, sct_arr, order


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.ssdf_open.restype = ctypes.c_void_p
    lib.ssdf_open.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS"), ctypes.c_int,
        ctypes.POINTER(_CError),
    ]
    lib.ssdf_open_dual.restype = ctypes.c_void_p
    lib.ssdf_open_dual.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(_CError),
    ]
    lib.ssdf_fill.restype = ctypes.c_int
    lib.ssdf_fill.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
        ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(_CError),
    ]
    lib.ssdf_read_extents.restype = ctypes.c_int
    lib.ssdf_read_extents.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(CExtent), ctypes.c_int64,
        ctypes.POINTER(CScatter), ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(_CError),
    ]
    lib.ssdf_close.restype = None
    lib.ssdf_close.argtypes = [ctypes.c_void_p]
    lib.ssdf_stats.restype = None
    lib.ssdf_stats.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS"), ctypes.c_int64,
    ]
    lib.ssdf_stats_ms.restype = None
    lib.ssdf_stats_ms.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), ctypes.c_int64,
    ]
    lib.ssdf_test_hook.restype = None
    lib.ssdf_test_hook.argtypes = [ctypes.c_int, ctypes.c_long]
    return lib


def load_native() -> ctypes.CDLL:
    """Locate and bind ``libengram_ssd.so``.

    Search order: vllm_ascend package root (wheel/develop convention shared
    with vllm_ascend_C), then the in-tree native/ directory beside this file
    (checkout convenience). No runtime compilation ever happens.
    """
    global _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        # engine.py lives at vllm_ascend/models/deepseek_v41/engram/ssd/.
        package_root = _HERE.parents[3]
        candidates = [
            package_root / _LIB_NAME,        # wheel/develop install convention
            _NATIVE_DIR / _LIB_NAME,         # source-tree fallback
        ]
        for path in candidates:
            if path.exists():
                _LIB = _bind(ctypes.CDLL(str(path)))
                return _LIB
        raise SSDEngineError(
            6, f"{_LIB_NAME} not found (looked in {package_root} and {_NATIVE_DIR}); "
               f"build vllm_ascend with COMPILE_CUSTOM_KERNELS=1 so the "
               f"engram_ssd_c target is packaged")


_HOOKS = frozenset({
    "submit_short", "submit_eagain", "getevents_delay_ms",
    "forge_stale", "io_setup_fail_idx", "clear",
})
_HOOK_CODES = {"clear": 0, "submit_short": 1, "submit_eagain": 2,
               "getevents_delay_ms": 3, "forge_stale": 4, "io_setup_fail_idx": 5}


def set_test_hook(kind: str, arg: int) -> None:
    """Fault-injection hook for tests. Never call in production."""
    if kind not in _HOOKS:
        raise SSDInputError(1, f"unknown hook {kind}")
    load_native().ssdf_test_hook(_HOOK_CODES[kind], int(arg))


class SSDFillEngine:
    """Persistent libaio engine reading 288B rows (256B codes + 32B scales).

    Args:
        table_paths: one full packed file, or two files where the second is
            an exact layer-14 slice.
        layer_rows: logical row count per layer.
        layer_bases: packed global row base per layer.
        threads_per_disk: persistent worker threads per disk (16 calibrated).
        qd: queue depth per worker. Calibrated value 1024; do not change it
            together with io_nr in one experiment.
        io_nr: io_setup request events per worker (default 2*qd). Verified
            independently from qd in phase-3 runs (1280 sufficient at
            qd=1024 under saturation, saving aio-nr quota vs 2048).
        cpu_base: first pinned CPU for workers (cpu_base+i); negative = no
            pinning.
    """

    def __init__(self, table_paths: list[str], *, layer_rows: list[int],
                 layer_bases: list[int], threads_per_disk: int = 16,
                 qd: int = 1024, io_nr: int | None = None,
                 cpu_base: int = -1, cpu_base_second: int | None = None):
        if len(layer_rows) != len(layer_bases) or not layer_rows:
            raise SSDInputError(1, f"layer_rows/bases mismatch: {layer_rows} vs {layer_bases}")
        for p in table_paths:
            if not Path(p).exists():
                raise SSDEngineError(6, f"table path not found: {p}")
        lib = load_native()
        io_nr = 2 * qd if io_nr is None else io_nr
        self.layer_rows = list(layer_rows)
        self.layer_bases = list(layer_bases)
        self.total_rows = layer_bases[-1] + layer_rows[-1]
        self._e = _CError()
        self._lib = lib
        self.dual = len(table_paths) == 2
        if self.dual and len(layer_rows) != 2:
            raise SSDInputError(1, "dual-disk mode requires exactly 2 layers")
        cb2 = cpu_base if cpu_base_second is None else cpu_base_second
        if self.dual:
            iv = np.array([0, layer_rows[0],
                           layer_bases[1], layer_bases[1] + layer_rows[1]], np.uint64)
            self._h = lib.ssdf_open_dual(
                str(table_paths[0]).encode(), str(table_paths[1]).encode(),
                layer_bases[1], layer_rows[0], layer_rows[1],
                threads_per_disk, qd, io_nr, cpu_base, cb2,
                ctypes.byref(self._e))
        else:
            iv = np.array([v for b, r in zip(layer_bases, layer_rows) for v in (b, b + r)],
                          np.uint64)
            self._h = lib.ssdf_open(str(table_paths[0]).encode(), threads_per_disk, qd,
                                    io_nr, cpu_base, iv, len(layer_rows),
                                    ctypes.byref(self._e))
        if not self._h:
            raise SSDEngineError(6, self._e.detail.decode(errors="replace"))

    def fill(self, ids: np.ndarray, slots: np.ndarray,
             codes: np.ndarray, scales: np.ndarray) -> dict[str, Any]:
        """Read miss rows and scatter into payload planes.

        ``ids`` are packed global row ids (uint64); ``slots`` the target DDR
        slot per id (int32, same length); ``codes``/``scales`` writable
        C-contiguous uint8 planes of shape [capacity, 256] / [capacity, 32].
        """
        if not self._h:
            raise SSDEngineError(5, "engine is closed")
        ids = np.asarray(ids)
        slots = np.asarray(slots)
        if ids.ndim != 1 or slots.ndim != 1:
            raise SSDInputError(1, "ids/slots must be 1-D")
        if ids.dtype != np.uint64 or slots.dtype != np.int32:
            raise SSDInputError(1, f"dtype must be uint64/int32, got {ids.dtype}/{slots.dtype}")
        if not ids.flags["C_CONTIGUOUS"] or not slots.flags["C_CONTIGUOUS"]:
            raise SSDInputError(1, "ids/slots must be C-contiguous")
        if len(ids) != len(slots):
            raise SSDInputError(1, f"ids/slots length mismatch: {len(ids)} vs {len(slots)}")
        if (codes.ndim != 2 or codes.dtype != np.uint8 or codes.shape[1] != 256
                or not codes.flags["C_CONTIGUOUS"] or not codes.flags["WRITEABLE"]):
            raise SSDInputError(1, f"codes must be writable uint8[cap,256], got {codes.shape} {codes.dtype}")
        if (scales.ndim != 2 or scales.dtype != np.uint8 or scales.shape[1] != 32
                or not scales.flags["C_CONTIGUOUS"] or not scales.flags["WRITEABLE"]):
            raise SSDInputError(1, f"scales must be writable uint8[cap,32], got {scales.shape} {scales.dtype}")
        if codes.shape[0] != scales.shape[0]:
            raise SSDInputError(1, f"plane capacity mismatch: {codes.shape[0]} vs {scales.shape[0]}")
        if len(ids) == 0:
            return {"rows": 0, "wall_ms": 0.0}
        wall = ctypes.c_double()
        rc = self._lib.ssdf_fill(self._h, ids, slots, len(ids),
                                 codes.ctypes.data, scales.ctypes.data, codes.shape[0],
                                 ctypes.byref(wall), ctypes.byref(self._e))
        if rc != 0:
            raise SSDReadError(rc, self._e.detail.decode(errors="replace"))
        return {"rows": len(ids), "wall_ms": wall.value}

    def read_extents(self, extents, scatters, codes, scales) -> dict[str, Any]:
        """Prefill merged reads: one aligned read per extent, scatter rows."""
        if not self._h:
            raise SSDEngineError(5, "engine is closed")
        n_ext, n_sct = len(extents), len(scatters)
        if n_ext == 0:
            return {"extents": 0, "rows": 0, "wall_ms": 0.0}
        wall = ctypes.c_double()
        rc = self._lib.ssdf_read_extents(self._h, extents, n_ext, scatters, n_sct,
                                         codes.ctypes.data, scales.ctypes.data,
                                         codes.shape[0], ctypes.byref(wall),
                                         ctypes.byref(self._e))
        if rc != 0:
            raise SSDReadError(rc, self._e.detail.decode(errors="replace"))
        return {"extents": n_ext, "rows": n_sct, "wall_ms": wall.value}

    def stats(self) -> dict[str, Any]:
        if not self._h:
            return {}
        out = np.zeros(10, np.int64)
        ms = np.zeros(2, np.float64)
        self._lib.ssdf_stats(self._h, out, 10)
        self._lib.ssdf_stats_ms(self._h, ms, 2)
        names = ("rows", "ge_calls", "ge_events", "fails", "stale_events",
                 "partial_submits", "eagain", "eintr", "poisoned", "rsv")
        d: dict[str, Any] = dict(zip(names, (int(v) for v in out)))
        d["extract_ms"] = float(ms[0])
        d["submit_ms"] = float(ms[1])
        return d

    def close(self) -> None:
        h = getattr(self, "_h", None)
        if h:
            self._lib.ssdf_close(h)
            self._h = None

    def __enter__(self) -> SSDFillEngine:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
