#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Lightweight HCCL lifecycle for sleep mode.

The historical sleep path destroys every HCCL process group and recreates it
on wake-up.  Recreating a communicator costs seconds on multi-rank jobs and
can fail when ranks wake up at different times.  CANN exposes
``HcclCommSuspend`` / ``HcclCommResume`` which keep the communicator identity
and only pause/resume its transport state.

Verified on Atlas A3 (910_93) with CANN 9.2 and torch_npu 2.10:

* ``HcclCommSuspend`` returns ``HCCL_E_SUSPENDING`` (22) for a healthy
  communicator.  The suspension is effective: ``resume_hccl_comm`` restores
  the ProcessGroup and a following all-reduce is exact.
* ``HcclCommResume`` (raw) returns 0 in ~1 ms, while suspend drains in-flight
  work and takes ~0.5 s for a 2-rank group.
* A suspended communicator must not be used for any collective (barriers
  included).  Callers therefore have to synchronize through a non-HCCL
  control plane (CPU/gloo or file) while the group is suspended.
"""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.utils import find_hccl_library

HCCL_OK = 0
# Returned by HcclCommSuspend for a healthy communicator.  The suspension is
# initiated asynchronously, so this is not an error for the sleep path.
HCCL_E_SUSPENDING = 22

SLEEP_HCCL_POLICIES = ("auto", "lightweight", "destroy", "skip")
_DISABLE_ENV = "VLLM_ASCEND_SLEEP_HCCL_DISABLE"


@dataclass
class HcclSuspendStats:
    """Counters exposed in logs and sleep-mode metrics."""

    suspend_calls: int = 0
    suspend_ok: int = 0
    suspend_suspending: int = 0
    suspend_failed: int = 0
    resume_calls: int = 0
    resume_ok: int = 0
    resume_failed: int = 0
    fallbacks: int = 0
    last_suspend_ms: float = 0.0
    last_resume_ms: float = 0.0
    suspended_handles: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "suspend_calls": self.suspend_calls,
            "suspend_ok": self.suspend_ok,
            "suspend_suspending": self.suspend_suspending,
            "suspend_failed": self.suspend_failed,
            "resume_calls": self.resume_calls,
            "resume_ok": self.resume_ok,
            "resume_failed": self.resume_failed,
            "fallbacks": self.fallbacks,
            "last_suspend_ms": round(self.last_suspend_ms, 3),
            "last_resume_ms": round(self.last_resume_ms, 3),
            "suspended_handles": list(self.suspended_handles),
        }


def comm_to_int(comm_handle: Any) -> int:
    """Normalize torch_npu's HcclComm handle (int or capsule-like object)."""
    if comm_handle is None:
        return 0
    if isinstance(comm_handle, int):
        return comm_handle
    for attr in ("value", "handle"):
        candidate = getattr(comm_handle, attr, None)
        if isinstance(candidate, int):
            return candidate
    try:
        return int(comm_handle)
    except (TypeError, ValueError):
        return 0


def _group_rank(device_group: Any) -> int:
    for attr in ("rank", "rank_in_group"):
        value = getattr(device_group, attr, None)
        if callable(value):
            try:
                return int(value())
            except Exception:  # noqa: BLE001
                continue
        elif isinstance(value, int):
            return value
    for env in ("LOCAL_RANK", "RANK"):
        raw = os.environ.get(env)
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                continue
    return 0


def get_group_backend(device_group: Any) -> Any | None:
    if device_group is None:
        return None
    try:
        return device_group._get_backend(torch.device("npu"))
    except Exception:  # noqa: BLE001
        return None


def get_group_comm_handles(device_group: Any) -> list[int]:
    """Best-effort extraction of the raw HcclComm(s) owned by a ProcessGroup."""
    backend = get_group_backend(device_group)
    getter = getattr(backend, "get_hccl_comm", None)
    if not callable(getter):
        return []
    rank = _group_rank(device_group)
    for args in ((rank,), (rank, False), (rank, True), ()):
        try:
            handle = comm_to_int(getter(*args))
        except TypeError:
            continue
        except Exception:  # noqa: BLE001
            return []
        return [handle] if handle else []
    return []


class HcclSuspendBackend:
    """Thin ctypes binding for HCCL suspend/resume with capability probing."""

    def __init__(self, stats: HcclSuspendStats | None = None) -> None:
        self.stats = stats or HcclSuspendStats()
        self._loaded = False
        self._suspend_fn: Any = None
        self._resume_fn: Any = None
        self._support_resume_fn: Any = None
        self._load_error: str | None = None

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if os.environ.get(_DISABLE_ENV) == "1":
            self._load_error = f"{_DISABLE_ENV}=1"
            return
        try:
            lib = ctypes.CDLL(find_hccl_library())
            suspend = lib.HcclCommSuspend
            suspend.argtypes = [ctypes.c_void_p]
            suspend.restype = ctypes.c_int
            resume = lib.HcclCommResume
            resume.argtypes = [ctypes.c_void_p]
            resume.restype = ctypes.c_int
            self._suspend_fn = suspend
            self._resume_fn = resume
            support = getattr(lib, "HcommIsSupportHcclCommResume", None)
            if support is not None:
                support.argtypes = []
                support.restype = ctypes.c_int
                self._support_resume_fn = support
            logger.info("Loaded HCCL suspend/resume symbols from %s", find_hccl_library())
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            logger.info("HCCL suspend/resume unavailable: %s", exc)

    def available(self) -> bool:
        self._load()
        if self._suspend_fn is None or self._resume_fn is None:
            return False
        if self._support_resume_fn is not None:
            try:
                if int(self._support_resume_fn()) == 0:
                    return False
            except Exception:  # noqa: BLE001
                return False
        return True

    def suspend(self, handle: int) -> bool:
        if not self.available() or not handle:
            return False
        self.stats.suspend_calls += 1
        started = time.perf_counter()
        try:
            rc = int(self._suspend_fn(ctypes.c_void_p(handle)))
        except Exception as exc:  # noqa: BLE001
            self.stats.suspend_failed += 1
            logger.warning("HcclCommSuspend raised for handle=%s: %s", handle, exc)
            return False
        self.stats.last_suspend_ms = (time.perf_counter() - started) * 1000.0
        if rc == HCCL_E_SUSPENDING:
            self.stats.suspend_suspending += 1
            self.stats.suspend_ok += 1
            return True
        if rc != HCCL_OK:
            self.stats.suspend_failed += 1
            logger.warning("HcclCommSuspend failed with code %s for handle %s", rc, handle)
            return False
        self.stats.suspend_ok += 1
        return True

    def resume(self, handle: int) -> bool:
        if not self.available() or not handle:
            return False
        self.stats.resume_calls += 1
        started = time.perf_counter()
        try:
            rc = int(self._resume_fn(ctypes.c_void_p(handle)))
        except Exception as exc:  # noqa: BLE001
            self.stats.resume_failed += 1
            logger.warning("HcclCommResume raised for handle=%s: %s", handle, exc)
            return False
        self.stats.last_resume_ms = (time.perf_counter() - started) * 1000.0
        if rc != HCCL_OK:
            self.stats.resume_failed += 1
            logger.warning("HcclCommResume failed with code %s for handle %s", rc, handle)
            return False
        self.stats.resume_ok += 1
        return True


def suspend_group(device_group: Any, backend: HcclSuspendBackend) -> tuple[bool, list[int]]:
    """Suspend every HcclComm reachable from ``device_group``."""
    handles = get_group_comm_handles(device_group)
    suspended: list[int] = []
    for handle in handles:
        if backend.suspend(handle):
            suspended.append(handle)
        else:
            for done in suspended:
                backend.resume(done)
            return False, []
    if suspended:
        backend.stats.suspended_handles.extend(suspended)
    return bool(suspended), suspended


def resume_group(device_group: Any, handles: list[int], backend: HcclSuspendBackend) -> bool:
    """Resume a suspended group.

    Prefer the torch_npu-supported ``ProcessGroupHCCL.resume_hccl_comm`` path:
    a raw ``HcclCommResume`` alone can leave the ProcessGroup state stale.
    """
    if not handles:
        return True
    started = time.perf_counter()
    backend_impl = get_group_backend(device_group)
    resume_fn = getattr(backend_impl, "resume_hccl_comm", None)
    if callable(resume_fn):
        try:
            resume_fn(int(torch.npu.current_device()))
            backend.stats.resume_calls += 1
            backend.stats.resume_ok += 1
            backend.stats.last_resume_ms = (time.perf_counter() - started) * 1000.0
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("ProcessGroupHCCL.resume_hccl_comm failed: %s", exc)
    ok = True
    for handle in handles:
        ok = backend.resume(handle) and ok
    return ok


__all__ = [
    "HCCL_E_SUSPENDING",
    "HCCL_OK",
    "HcclSuspendBackend",
    "HcclSuspendStats",
    "SLEEP_HCCL_POLICIES",
    "comm_to_int",
    "get_group_backend",
    "get_group_comm_handles",
    "resume_group",
    "suspend_group",
]
