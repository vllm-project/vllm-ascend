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
"""Payload planes for the SSD row cache.

Two planes (int8 codes, float32 scales) of ``capacity`` rows each. The
owner allocates; consumers map the same physical bytes at their own
64KiB-aligned virtual addresses via :func:`am_map_shared_aligned`
(native/aligned_mapping.c). Every rank then UVA-registers its own mapping
with CANN exactly like ``SharedUvaBuffer`` does.
"""

from __future__ import annotations

import ctypes
from multiprocessing import shared_memory
from unittest.mock import patch

import torch
import torch.distributed as dist

from vllm_ascend.models.deepseek_v41.engram.npu import (
    CHUNK_ROWS,
    DeviceOperator,
    _host_library,
)

ALIGN_64K = 64 * 1024


def _bind_mapping(lib: ctypes.CDLL) -> None:
    if getattr(lib, "_am_bound", False):
        return
    lib.am_map_shared_aligned.restype = ctypes.c_void_p
    lib.am_map_shared_aligned.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t,
                                          ctypes.c_char_p, ctypes.c_int]
    lib.am_unmap.restype = ctypes.c_int
    lib.am_unmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.am_madvise_hugepage.restype = ctypes.c_int
    lib.am_madvise_hugepage.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib._am_bound = True


def map_shared_aligned(shm: shared_memory.SharedMemory, size: int,
                       align: int = ALIGN_64K) -> int:
    """Map ``shm`` at a fresh 64KiB-aligned VA; returns the address int."""
    lib = _host_library()
    if not hasattr(lib, "am_map_shared_aligned"):
        from vllm_ascend.models.deepseek_v41.engram.ssd.engine import load_native

        _bind_mapping(load_native())
        lib = load_native()
    _bind_mapping(lib)
    fd = getattr(shm, "fd", None)
    if fd is None:
        fd = getattr(shm, "_fd")
    err = ctypes.create_string_buffer(256)
    addr = lib.am_map_shared_aligned(int(fd), size, align, err, 256)
    if not addr:
        raise RuntimeError(f"aligned shared mapping failed: {err.value.decode()}")
    if lib.am_madvise_hugepage(ctypes.c_void_p(addr), size) != 0:
        # hugepage 建议失败不致命(THP 可能未启用), 记录到 stderr 由调用方观察
        pass
    return int(addr)


def unmap_shared_aligned(addr: int, size: int) -> None:
    from vllm_ascend.models.deepseek_v41.engram.ssd.engine import load_native

    lib = load_native()
    _bind_mapping(lib)
    rc = lib.am_unmap(ctypes.c_void_p(addr), size)
    if rc != 0:
        raise RuntimeError(f"aligned unmap failed: errno={rc}")


class AlignedSharedUvaPlane:
    """One 64KiB-aligned shared plane, UVA-registered in this process.

    Leader creates the SharedMemory segment and broadcasts its name on the
    group's CPU (gloo) group; every rank maps it at its own aligned address
    with physical offset 0, registers it with CANN and builds its device
    pointer table. Errors are aggregated across the group before the leader
    unlinks, matching ``SharedUvaBuffer``'s contract.
    """

    def __init__(self, rows: int, row_bytes: int, dtype, device, group):
        self.lib = _host_library()
        self.group = group
        self.rows = rows
        self.row_bytes = row_bytes
        self.size = rows * row_bytes
        self.shm = None
        self.tensor = None
        self.ptrs = None
        self.addr = 0
        self._mapped_size = 0

        cpu_group = group.cpu_group
        leader = dist.get_global_rank(cpu_group, 0)
        payload: list[str | None] = [None]
        if group.rank_in_group == 0:
            try:
                self.shm = shared_memory.SharedMemory(create=True, size=self.size)
                payload = [self.shm.name]
            except Exception as exc:  # noqa: BLE001 - reported to the group
                if self.shm is not None:
                    self.shm.unlink()
                    self.shm.close()
                payload = [f"ERROR: {type(exc).__name__}: {exc}"]
        dist.broadcast_object_list(payload, src=leader, group=cpu_group)
        name = payload[0]
        if name is None or name.startswith("ERROR:"):
            raise RuntimeError(f"aligned shared plane creation failed: {name}")

        error: str | None = None
        try:
            if self.shm is None:
                with patch("multiprocessing.resource_tracker.register",
                           lambda *args, **kwargs: None):
                    self.shm = shared_memory.SharedMemory(name=name)
            self.addr = map_shared_aligned(self.shm, self.size)
            self._mapped_size = self.size
            rc = self.lib.aclrtHostRegisterV2(ctypes.c_void_p(self.addr), self.size,
                                              DeviceOperator.host_register_flags())
            if rc:
                raise RuntimeError(f"aclrtHostRegisterV2 failed: rc={rc} size={self.size}")
            address = ctypes.c_void_p()
            rc = self.lib.aclrtHostGetDevicePointer(ctypes.c_void_p(self.addr),
                                                    ctypes.byref(address), 0)
            if rc:
                raise RuntimeError(f"aclrtHostGetDevicePointer failed: rc={rc}")
            buf = (ctypes.c_char * self.size).from_address(self.addr)
            self.tensor = torch.frombuffer(
                buf, dtype=dtype).reshape((rows, row_bytes // torch.empty((), dtype=dtype).element_size()))
            self.ptrs = torch.tensor(
                [address.value + start * row_bytes for start in range(0, rows, CHUNK_ROWS)],
                dtype=torch.int64, device=device)
        except Exception as exc:  # noqa: BLE001 - aggregated below
            error = f"{type(exc).__name__}: {exc}"

        errors: list[str | None] = [None] * group.world_size
        dist.all_gather_object(errors, error, group=cpu_group)
        failures = "; ".join(f"rank {r}: {e}" for r, e in enumerate(errors) if e is not None)
        dist.barrier(group=cpu_group)
        if failures:
            try:
                self.close()
            except RuntimeError as exc:
                failures = f"{failures}; release failed: {exc}"
            if group.rank_in_group == 0:
                self._unlink()
            raise RuntimeError(f"aligned shared plane initialization failed: {failures}")
        if group.rank_in_group == 0:
            self._unlink()

    def _unlink(self) -> None:
        try:
            if self.shm is not None:
                self.shm.unlink()
        except OSError as exc:
            import logging

            logging.getLogger(__name__).warning(
                "aligned shared plane unlink failed: %s", exc)

    def close(self) -> None:
        """Unregister → unmap → close shm. Keeps backing on unregister failure."""
        if self.addr:
            if self.ptrs is not None and self.ptrs.device.type == "npu":
                torch.npu.synchronize()
            rc = self.lib.aclrtHostUnregister(ctypes.c_void_p(self.addr))
            if rc:
                raise RuntimeError(f"aclrtHostUnregister failed: rc={rc}")
            unmap_shared_aligned(self.addr, self._mapped_size)
            self.addr = 0
        self.tensor = None
        self.ptrs = None
        if self.shm is not None:
            self.shm.close()
            self.shm = None


class SSDPlanes:
    """codes + scales planes for one cache capacity.

    DP1 single-consumer mode uses private ``HostUvaBuffer`` planes (already
    64KiB aligned); shared mode uses two ``AlignedSharedUvaPlane`` so the
    node-local DP group consumes one physical copy.
    """

    def __init__(self, capacity: int, device, group=None):
        self.capacity = capacity
        if group is not None:
            self.codes = AlignedSharedUvaPlane(capacity, 256, torch.uint8, device, group)
            self.scales = AlignedSharedUvaPlane(capacity, 32, torch.uint8, device, group)
            self.shared = True
        else:
            from vllm_ascend.models.deepseek_v41.engram.npu import HostUvaBuffer

            self.codes = HostUvaBuffer((capacity, 256), torch.uint8, device)
            self.scales = HostUvaBuffer((capacity, 32), torch.uint8, device)
            self.shared = False

    def numpy_views(self):
        """(codes, scales) as writable numpy planes [cap,256]/[cap,32]."""
        return self.codes.tensor.numpy(), self.scales.tensor.numpy()

    def close(self) -> None:
        for p in (self.scales, self.codes):
            if p is not None:
                p.close()
