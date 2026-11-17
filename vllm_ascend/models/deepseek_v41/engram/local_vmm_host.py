# SPDX-License-Identifier: Apache-2.0
"""Host VMM storage for independent Engram consumers.

Consumers expose metadata-only tensors and real NPU pointer tables. They never
dereference remote host addresses on the CPU or materialize the table in HBM.
Pool publication, identity and lease management belong to the caller.
"""

import ctypes
import os
from pathlib import Path

import torch

GIB = 1 << 30
CHUNK_ROWS = 1 << 22


class EngramReleaseError(RuntimeError):
    """A VMM release failed and remains eligible for retry."""


class _Allocation(ctypes.Structure):
    _fields_ = [
        ("host_ptr", ctypes.c_void_p),
        ("device_ptr", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("token", ctypes.c_uint64),
    ]


def _library(path: Path):
    lib = ctypes.CDLL(str(path))
    if lib.host_vmm_abi_version() != 3:
        raise RuntimeError("Engram host-VMM ABI mismatch")
    lib.host_shared_registered_alloc.argtypes = [
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(_Allocation),
    ]
    lib.host_shared_registered_free.argtypes = [ctypes.POINTER(_Allocation)]
    lib.host_shared_last_error.restype = ctypes.c_char_p
    return lib


def retry_pending_rollbacks(library_path=None, *, lib=None):
    library_path = library_path or Path(__file__).resolve().parents[3] / "libengram_host_vmm.so"
    lib = lib or _library(Path(library_path))
    if lib.host_vmm_retry_rollbacks():
        raise EngramReleaseError(f"Engram VMM rollback: {lib.host_shared_last_error().decode()}")


class MappedHostBuffer:
    """One physical range or its import; no CPU aliases escape this object.

    The creator must outlive consumers until the pool's lease manager releases
    them. Stop all launches/graphs before close. Failed release stays retryable.
    """

    def __init__(self, path, rows, width, dtype, *, owner, device, library_path=None, lib=None):
        if rows <= 0 or width <= 0 or dtype not in (torch.int8, torch.float32):
            raise ValueError("Host VMM requires positive INT8/FP32 dimensions")
        path = Path(path)
        parent = path.parent.stat()
        if parent.st_uid != os.getuid() or parent.st_mode & 0o077:
            raise ValueError("Capability directory must be owned by this uid and mode0700")
        self.rows, self.width, self.dtype = rows, width, dtype
        self.owner = owner
        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("An explicit NPU device index is required")
        self.row_bytes = width * torch.empty((), dtype=dtype).element_size()
        self.size = ((rows * self.row_bytes + GIB - 1) // GIB) * GIB
        library_path = library_path or Path(__file__).resolve().parents[3] / "libengram_host_vmm.so"
        self.lib = lib or _library(Path(library_path))
        self.allocation = _Allocation()
        self.closed = False
        with torch.npu.device(self.device):
            rc = self.lib.host_shared_registered_alloc(
                self.size, self.device.index, os.fsencode(path), int(owner), ctypes.byref(self.allocation)
            )
        if rc:
            raise RuntimeError(f"Engram host VMM: {self.lib.host_shared_last_error().decode()}")

    def publish(self, start, source):
        """Copy one CPU checkpoint chunk into creator-owned host memory."""
        if self.closed or not self.owner:
            raise RuntimeError("Only a live owner may publish table rows")
        if source.device.type != "cpu" or source.dtype != self.dtype:
            raise ValueError("Publication requires a matching CPU dtype")
        if source.ndim != 2 or source.shape[1] != self.width or start < 0 or start + len(source) > self.rows:
            raise ValueError("Publication is outside the table shape")
        source = source.contiguous()
        ctypes.memmove(
            self.allocation.host_ptr + start * self.row_bytes, source.data_ptr(), source.numel() * source.element_size()
        )

    def close(self):
        if self.closed:
            return
        with torch.npu.device(self.device):
            torch.npu.synchronize()
            rc = self.lib.host_shared_registered_free(ctypes.byref(self.allocation))
        if rc:
            raise EngramReleaseError(f"Engram VMM release: {self.lib.host_shared_last_error().decode()}")
        self.closed = True
