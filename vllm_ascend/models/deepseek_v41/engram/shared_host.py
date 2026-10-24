# SPDX-License-Identifier: Apache-2.0
"""Host VMM storage for independent Engram consumers.

Consumers expose metadata-only tensors and real NPU pointer tables. They never
dereference remote host addresses on the CPU or materialize the table in HBM.
Pool publication, identity and lease management belong to the caller.
"""

import ctypes
import os
import weakref
from pathlib import Path

import torch

GIB = 1 << 30
CHUNK_ROWS = 1 << 22


class _Allocation(ctypes.Structure):
    _fields_ = [
        ("host_ptr", ctypes.c_void_p),
        ("device_ptr", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("physical_device", ctypes.c_int32),
        ("owner", ctypes.c_int32),
        ("host_type", ctypes.c_int32),
        ("host_id", ctypes.c_int32),
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
        ("token", ctypes.c_uint64),
    ]


def _library(path: Path):
    lib = ctypes.CDLL(str(path))
    if lib.host_vmm_abi_version() != 2:
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


class MappedHostBuffer:
    """One physical range or its import; no CPU aliases escape this object.

    The creator must outlive consumers until the pool's lease manager releases
    them. Stop all launches/graphs before close. Failed release stays retryable.
    """

    def __init__(self, path, rows, width, dtype, *, owner, device, library_path=None):
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
        self.lib = _library(Path(library_path))
        self.allocation = _Allocation()
        self.closed = False
        self._views = weakref.WeakSet()
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

    def view(self, start=0, end=None):
        """TP-independent row slice consumed by the existing lookup kernel."""
        end = self.rows if end is None else end
        if self.closed or not 0 <= start < end <= self.rows:
            raise ValueError("Invalid or closed host-VMM slice")
        view = HostTableSlice(self, start, end)
        self._views.add(view)
        return view

    def close(self):
        if self.closed:
            return
        if self._views:
            raise RuntimeError("Release all Engram slices before closing their mapping")
        with torch.npu.device(self.device):
            torch.npu.synchronize()
            rc = self.lib.host_shared_registered_free(ctypes.byref(self.allocation))
        if rc:
            raise RuntimeError(f"Engram VMM release: {self.lib.host_shared_last_error().decode()}")
        self.closed = True


class HostTableSlice:
    """Metadata and device pointers only; no copy and no remote CPU access."""

    def __init__(self, mapping, start, end):
        self._mapping = mapping
        self.tensor = torch.empty((end - start, mapping.width), dtype=mapping.dtype, device="meta")
        self.ptrs = torch.tensor(
            [mapping.allocation.device_ptr + row * mapping.row_bytes for row in range(start, end, CHUNK_ROWS)],
            dtype=torch.int64,
            device=mapping.device,
        )

    def close(self):
        if self._mapping is None:
            return
        with torch.npu.device(self._mapping.device):
            torch.npu.synchronize()
        self._mapping._views.discard(self)
        self.ptrs = self.tensor = None
        self._mapping = None
