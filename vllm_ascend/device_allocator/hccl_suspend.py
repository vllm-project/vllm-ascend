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

"""Lightweight HCCL communicator suspend/resume via CANN HcclCommSuspend/Resume.

Mirrors vLLM NCCL sleep suspension (ncclCommSuspend/Resume, PR #51485):
idempotent suspend/resume of communicator buffers while keeping topology.

Wake prefers torch_npu ProcessGroupHCCL.resume_hccl_comm(device_id) so the
ProcessGroup rebinds after raw HcclCommSuspend (see torch_npu.distributed
reinit_process_group(rebuild_link=False)).
"""

from __future__ import annotations

import ctypes
import os
from ctypes import c_int, c_void_p
from typing import Any

from vllm.logger import logger

_HCCL_OK = 0
_lib_hccl: ctypes.CDLL | None = None
_suspend_fn = None
_resume_fn = None
_support_suspend_fn = None
_support_resume_fn = None
# Idempotence guards (same idea as PyNcclCommunicator._suspended).
_suspended_handles: set[int] = set()


def comm_to_int(comm_handle: Any) -> int:
    """Convert raw/wrapper HCCL handles to int safely."""
    if comm_handle is None:
        return 0
    val = comm_handle.value if hasattr(comm_handle, "value") else comm_handle
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0


def _candidate_lib_paths() -> list[str]:
    roots: list[str] = []
    for env in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_AICPU_PATH"):
        val = os.environ.get(env)
        if val:
            roots.append(val)
    roots.extend(
        [
            "/usr/local/Ascend/cann",
            "/usr/local/Ascend/ascend-toolkit/latest",
            "/usr/local/Ascend/cann-9.0.0",
            "/usr/local/Ascend/cann-9.1.0",
        ]
    )
    paths: list[str] = []
    for root in roots:
        paths.extend(
            [
                os.path.join(root, "lib64", "libhccl.so"),
                os.path.join(root, "aarch64-linux", "lib64", "libhccl.so"),
            ]
        )
    return paths


def _load_hccl_lib() -> ctypes.CDLL | None:
    global _lib_hccl, _suspend_fn, _resume_fn, _support_suspend_fn, _support_resume_fn
    if _lib_hccl is not None:
        return _lib_hccl
    for path in _candidate_lib_paths():
        if not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            suspend = lib.HcclCommSuspend
            resume = lib.HcclCommResume
            suspend.argtypes = [c_void_p]
            suspend.restype = c_int
            resume.argtypes = [c_void_p]
            resume.restype = c_int
            _lib_hccl = lib
            _suspend_fn = suspend
            _resume_fn = resume
            for name, attr in (
                ("HcommIsSupportHcclCommSuspend", "_support_suspend_fn"),
                ("HcommIsSupportHcclCommResume", "_support_resume_fn"),
            ):
                try:
                    fn = getattr(lib, name)
                    fn.argtypes = []
                    fn.restype = c_int
                    globals()[attr] = fn
                except AttributeError:
                    pass
            logger.info("Loaded HCCL suspend/resume from %s", path)
            return _lib_hccl
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load HCCL symbols from %s: %s", path, exc)
    return None


def hccl_suspend_available() -> bool:
    if _load_hccl_lib() is None or _suspend_fn is None or _resume_fn is None:
        return False
    if _support_suspend_fn is not None and int(_support_suspend_fn()) == 0:
        return False
    return not (_support_resume_fn is not None and int(_support_resume_fn()) == 0)


def suspend_hccl_comm(comm_handle: int | Any) -> bool:
    """Suspend one HcclComm (idempotent)."""
    if not hccl_suspend_available():
        return False
    handle = comm_to_int(comm_handle)
    if handle == 0:
        return False
    if handle in _suspended_handles:
        return True
    assert _suspend_fn is not None
    rc = _suspend_fn(c_void_p(handle))
    if rc != _HCCL_OK:
        logger.warning("HcclCommSuspend failed with code %s for handle %s", rc, handle)
        return False
    _suspended_handles.add(handle)
    return True


def resume_hccl_comm(comm_handle: int | Any) -> bool:
    """Resume one HcclComm via raw CANN API (idempotent)."""
    if not hccl_suspend_available():
        return False
    handle = comm_to_int(comm_handle)
    if handle == 0:
        return False
    _suspended_handles.discard(handle)
    assert _resume_fn is not None
    rc = _resume_fn(c_void_p(handle))
    if rc != _HCCL_OK:
        logger.warning("HcclCommResume failed with code %s for handle %s", rc, handle)
        return False
    return True


def clear_suspended_handles() -> None:
    _suspended_handles.clear()


def _rank_for_backend(device_group: Any) -> int:
    for attr in ("rank", "rank_in_group"):
        fn = getattr(device_group, attr, None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:  # noqa: BLE001
                pass
        elif isinstance(fn, int):
            return int(fn)
    for env in ("LOCAL_RANK", "RANK"):
        if env in os.environ:
            try:
                return int(os.environ[env])
            except ValueError:
                pass
    return 0


def group_sort_key(group: Any) -> str:
    return str(getattr(group, "unique_name", None) or getattr(group, "group_name", None) or id(group))


def get_backend_hccl_comm(device_group: Any) -> Any | None:
    """Best-effort extract raw HcclComm from a torch ProcessGroup."""
    if device_group is None:
        return None
    try:
        import torch

        backend = device_group._get_backend(torch.device("npu"))
    except Exception:  # noqa: BLE001
        return None
    getter = getattr(backend, "get_hccl_comm", None)
    if not callable(getter):
        return None
    rank = _rank_for_backend(device_group)
    args_candidates: list[tuple[Any, ...]] = [(rank,), (rank, False), (rank, True), tuple()]
    for args in args_candidates:
        try:
            return getter(*args)
        except TypeError:
            continue
        except Exception:  # noqa: BLE001
            return None
    return None


def resume_process_group_hccl(device_group: Any) -> bool:
    """Resume HCCL via ProcessGroupHCCL.resume_hccl_comm(device_id).

    This is the torch_npu-supported path (reinit_process_group(rebuild_link=False)).
    Raw HcclCommResume alone can leave ProcessGroup state stale and hang TP generate.
    """
    if device_group is None:
        return False
    try:
        import torch

        device_id = int(torch.npu.current_device())
        backend = device_group._get_backend(torch.device("npu"))
        resume_fn = getattr(backend, "resume_hccl_comm", None)
        if callable(resume_fn):
            resume_fn(device_id)
            handle = get_backend_hccl_comm(device_group)
            if handle is not None:
                _suspended_handles.discard(comm_to_int(handle))
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("ProcessGroup resume_hccl_comm failed: %s", exc)

    handle = get_backend_hccl_comm(device_group)
    if handle is None:
        return False
    return resume_hccl_comm(handle)


def resume_all_registered_pgs() -> int:
    """Resume every HCCL ProcessGroup known to torch.distributed / torch_npu."""
    try:
        import torch_npu.distributed as npu_dist

        reinit = getattr(npu_dist, "reinit_process_group", None)
        if callable(reinit):
            reinit(rebuild_link=False)
            clear_suspended_handles()
            return 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("reinit_process_group(rebuild_link=False) failed: %s", exc)

    resumed = 0
    try:
        import torch
        import torch.distributed.distributed_c10d as dist_c10d

        device_id = int(torch.npu.current_device())
        npu_device = torch.device("npu")
        world = getattr(dist_c10d, "_world", None)
        pgs = []
        if world is not None:
            pgs = list(getattr(world, "pg_map", {}).keys())
        if not pgs and hasattr(dist_c10d, "_pg_map"):
            pgs = list(dist_c10d._pg_map.keys())
        for pg in pgs:
            try:
                if npu_device not in getattr(pg, "_device_types", [npu_device]):
                    continue
                backend = pg._get_backend(npu_device)
                resume_fn = getattr(backend, "resume_hccl_comm", None)
                if callable(resume_fn):
                    resume_fn(device_id)
                    resumed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("resume_hccl_comm on pg failed: %s", exc)
        if resumed:
            clear_suspended_handles()
    except Exception as exc:  # noqa: BLE001
        logger.warning("resume_all_registered_pgs failed: %s", exc)
    return resumed
