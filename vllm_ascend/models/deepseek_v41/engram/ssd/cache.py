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
"""Engram SSD row cache: DDR slot directory with strict lease semantics.

State machine (docs/04 §7)::

    IDLE --acquire--> RESERVED --publish--> PUBLISHED --release--> IDLE
    RESERVED --abort--> IDLE        (fill failed; pending rows never visible)
    any --consumer state unknown--> POISONED

One active lease at a time; a second acquire is rejected. Hits are pinned
before miss victims are chosen, so eviction can never displace a row the
current batch still needs. Directory visibility switches only at publish,
so a partially-filled batch never becomes visible. Keys are packed global
row ids validated against the table's logical row limit.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vllm_ascend.models.deepseek_v41.engram.ssd.engine import (
    SSDError,
    SSDEngineError,
    SSDInputError,
    load_native,
)


class SSDCacheError(SSDError):
    """Cache state-machine / capacity error."""


class _PhaseStats(ctypes.Structure):
    _fields_ = [
        ("validate_ms", ctypes.c_double),
        ("lookup_ms", ctypes.c_double),
        ("sort_ms", ctypes.c_double),
        ("assign_ms", ctypes.c_double),
        ("wall_ms", ctypes.c_double),
        ("maintain_ms", ctypes.c_double),
        ("hit_rows", ctypes.c_uint64),
        ("miss_rows", ctypes.c_uint64),
        ("unique_misses", ctypes.c_uint64),
        ("evictions", ctypes.c_uint64),
        ("probe_total", ctypes.c_uint64),
        ("probe_max", ctypes.c_uint64),
        ("tombstones_after", ctypes.c_uint64),
        ("directory_buckets", ctypes.c_uint64),
        ("rebuilt", ctypes.c_int),
    ]


def _bind(lib: ctypes.CDLL) -> None:
    lib.rc_create.restype = ctypes.c_void_p
    lib.rc_create.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint, ctypes.c_uint64]
    lib.rc_destroy.restype = None
    lib.rc_destroy.argtypes = [ctypes.c_void_p]
    lib.rc_plan.restype = ctypes.c_int
    lib.rc_plan.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS"),
        ctypes.POINTER(_PhaseStats),
    ]
    lib.rc_publish.restype = ctypes.c_int
    lib.rc_publish.argtypes = [ctypes.c_void_p]
    lib.rc_release.restype = ctypes.c_int
    lib.rc_release.argtypes = [ctypes.c_void_p]
    lib.rc_abort.restype = ctypes.c_int
    lib.rc_abort.argtypes = [ctypes.c_void_p]
    lib.rc_poison.restype = None
    lib.rc_poison.argtypes = [ctypes.c_void_p]
    lib.rc_lease_id.restype = ctypes.c_uint64
    lib.rc_lease_id.argtypes = [ctypes.c_void_p]


_BIND_DONE = False


def _ensure_bind() -> ctypes.CDLL:
    global _BIND_DONE
    lib = load_native()
    if not _BIND_DONE:
        _bind(lib)
        _BIND_DONE = True
    return lib


@dataclass
class CacheLease:
    """Result of one acquire: slots aligned with the request order.

    ``slots[i]`` is the DDR slot for ``rows[i]`` (-1 only transiently inside
    acquire; the published lease always carries a valid slot per request).
    ``miss_rows``/``miss_slots`` are deduplicated rows that must be filled
    before publish.
    """

    lease_id: int
    rows: np.ndarray
    slots: np.ndarray
    miss_rows: np.ndarray
    miss_slots: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)
    published: bool = False
    released: bool = False
    aborted: bool = False


class SSDRowCache:
    """Directory over a fixed-capacity DDR payload plane.

    Args:
        capacity: number of 288B payload slots (two layers share one budget).
        row_limit: total packed global rows the ids may reference (manifest).
        threads: reserved for parallel plan segments (currently 1).
        rebuild_threshold: tombstone count that triggers a batch-boundary
            directory rebuild (default capacity/4 — a starting value to be
            tuned against probe-length/maintenance measurements, not a
            settled threshold).
    """

    def __init__(self, capacity: int, row_limit: int, threads: int = 1,
                 rebuild_threshold: int | None = None):
        if capacity <= 0 or row_limit <= 0:
            raise SSDInputError(1, f"capacity/row_limit must be positive: {capacity}/{row_limit}")
        if rebuild_threshold is None:
            rebuild_threshold = max(capacity // 4, 1)
        self.capacity = int(capacity)
        self.row_limit = int(row_limit)
        lib = _ensure_bind()
        self._lib = lib
        self._h = lib.rc_create(ctypes.c_uint32(capacity), ctypes.c_uint64(row_limit),
                                ctypes.c_uint(threads), ctypes.c_uint64(rebuild_threshold))
        if not self._h:
            raise SSDEngineError(6, "row cache allocation failed")
        self._active: CacheLease | None = None
        self._poisoned = False
        self._closed = False
        self._miss_buf_rows = np.empty(0, np.uint64)
        self._miss_buf_slots = np.empty(0, np.uint32)

    # ---------- state machine ----------

    def acquire(self, rows: np.ndarray) -> CacheLease:
        """Protect hits, pick miss victims, return the lease (RESERVED)."""
        if self._closed:
            raise SSDCacheError(5, "cache is closed")
        if self._poisoned:
            raise SSDCacheError(7, "cache is poisoned; consumer state unknown")
        if self._active is not None:
            raise SSDCacheError(3, "second acquire while lease active is not allowed")
        rows = np.asarray(rows)
        if rows.ndim != 1 or rows.dtype != np.uint64 or not rows.flags["C_CONTIGUOUS"]:
            raise SSDInputError(1, "rows must be 1-D contiguous uint64 packed global ids")
        if len(rows) > 0xFFFFFFFF:
            raise SSDInputError(1, "batch too large")
        n = len(rows)
        slots = np.full(n, -1, np.int32)
        if n > len(self._miss_buf_rows):
            self._miss_buf_rows = np.empty(n, np.uint64)
            self._miss_buf_slots = np.empty(n, np.uint32)
        st = _PhaseStats()
        rc = self._lib.rc_plan(self._h, rows, n, slots,
                               self._miss_buf_rows, self._miss_buf_slots,
                               ctypes.byref(st))
        if rc != 0:
            if rc == -2:
                raise SSDCacheError(2, "batch exceeds cache capacity (protected+miss)")
            if rc == -1:
                raise SSDInputError(1, "row id out of logical range")
            if rc == -5:
                raise SSDCacheError(7, "cache is poisoned")
            raise SSDCacheError(10, f"plan failed rc={rc}")
        miss_count = int(st.unique_misses)
        lease = CacheLease(
            lease_id=int(self._lib.rc_lease_id(self._h)),
            rows=rows.copy(),
            slots=slots,
            miss_rows=self._miss_buf_rows[:miss_count].copy(),
            miss_slots=self._miss_buf_slots[:miss_count].astype(np.int32),
            stats={
                "hit_rows": int(st.hit_rows), "miss_rows": int(st.miss_rows),
                "unique_misses": miss_count, "evictions": int(st.evictions),
                "wall_ms": st.wall_ms, "lookup_ms": st.lookup_ms,
                "sort_ms": st.sort_ms, "assign_ms": st.assign_ms,
                "maintain_ms": st.maintain_ms, "probe_max": int(st.probe_max),
                "tombstones": int(st.tombstones_after),
                "directory_buckets": int(st.directory_buckets),
                "rebuilt": int(st.rebuilt),
            },
        )
        self._active = lease
        return lease

    def _require(self, lease: CacheLease, *, active: bool = True) -> None:
        if lease is not self._active:
            raise SSDCacheError(3, "operation on expired/foreign lease")
        if active and lease.published:
            raise SSDCacheError(3, "lease already published")

    def publish(self, lease: CacheLease) -> None:
        """Full-batch success: make new rows directory-visible."""
        self._require(lease)
        rc = self._lib.rc_publish(self._h)
        if rc != 0:
            raise SSDCacheError(3, f"publish rejected rc={rc}")
        lease.published = True

    def release(self, lease: CacheLease) -> None:
        """All consumers finished: drop protection (payload stays cached)."""
        if lease is not self._active:
            raise SSDCacheError(3, "release on expired/foreign lease")
        if not lease.published:
            raise SSDCacheError(3, "release before publish is not allowed (use abort)")
        rc = self._lib.rc_release(self._h)
        if rc != 0:
            raise SSDCacheError(3, f"release rejected rc={rc}")
        lease.released = True
        self._active = None

    def abort(self, lease: CacheLease) -> None:
        """Fill failed: remove all pending rows, keep victims' loss."""
        self._require(lease)
        rc = self._lib.rc_abort(self._h)
        if rc != 0:
            raise SSDCacheError(3, f"abort rejected rc={rc}")
        lease.aborted = True
        self._active = None

    def poison(self, reason: str) -> None:
        """Consumer completion unknown: freeze the cache permanently."""
        self._poisoned = True
        self._lib.rc_poison(self._h)

    # ---------- lifecycle ----------

    def close(self) -> None:
        if self._h and not self._closed:
            if self._active is not None:
                raise SSDCacheError(3, "close with active lease (drain first)")
            self._lib.rc_destroy(self._h)
            self._h = None
            self._closed = True

    @property
    def is_active(self) -> bool:
        return self._active is not None

    @property
    def poisoned(self) -> bool:
        return self._poisoned
