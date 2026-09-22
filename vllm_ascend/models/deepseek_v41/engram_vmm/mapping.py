# SPDX-License-Identifier: Apache-2.0
"""Persistent host VMM mappings; no allocation or registration in lookup."""

import atexit
import ctypes
import functools
import os
import sys
from pathlib import Path

import torch
import torch_npu

GIB = 1 << 30


class Allocation(ctypes.Structure):
    _fields_ = [
        ("host_ptr", ctypes.c_void_p),
        ("device_ptr", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("physical_device", ctypes.c_int32),
        ("owner", ctypes.c_int32),
        ("host_observed_location_type", ctypes.c_int32),
        ("host_observed_location_id", ctypes.c_int32),
        ("device_observed_location_type", ctypes.c_int32),
        ("device_observed_location_id", ctypes.c_int32),
        ("token", ctypes.c_uint64),
    ]


def tensor_at(ptr, shape, dtype, device):
    nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype, device="cpu").element_size()
    storage = torch_npu._C._construct_storage_from_data_pointer(ptr, device, nbytes)
    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(
        dict(
            data_ptr=ptr, device=device, nbytes=nbytes, dtype=dtype, size=shape, stride=(shape[1], 1), storage_offset=0
        ),
        storage,
    )


@functools.lru_cache(maxsize=1)
def library():
    path = Path(__file__).resolve().parents[3] / "libengram_host_vmm.so"
    if not path.is_file():
        raise RuntimeError("Engram VMM requires an A3 build with CANN host-VMM/fabric-handle support")
    lib = ctypes.CDLL(str(path))
    if lib.host_vmm_abi_version() != 2:
        raise RuntimeError("VMM native/Python ABI mismatch")
    lib.host_shared_registered_alloc.argtypes = [
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(Allocation),
    ]
    lib.host_shared_registered_free.argtypes = [ctypes.POINTER(Allocation)]
    lib.host_shared_registered_defer_free.argtypes = [ctypes.POINTER(Allocation)]
    lib.host_shared_last_error.restype = ctypes.c_char_p
    lib.host_vmm_lifecycle_counts.restype = ctypes.c_uint64
    lib.host_vmm_live_mappings.restype = ctypes.c_uint64
    atexit.register(_shutdown, lib)
    return lib


def _shutdown(lib):
    """Process-exit fallback only: no further inference may be submitted."""
    if not lib.host_vmm_live_mappings():
        return
    try:
        # All table users must already be quiescent. Each worker uses one NPU.
        torch.npu.synchronize()
        rc = lib.host_vmm_close_all()
        if rc:
            raise RuntimeError(lib.host_shared_last_error().decode())
        print("ENGRAM_VMM_EXIT_CLEAN", flush=True)
    except Exception as exc:
        print(f"ENGRAM_VMM_EXIT_ERROR: {exc}; driver process cleanup required", file=sys.stderr)


def retry_rollbacks():
    """Retry failed construction cleanup without opening a previously unused library."""
    if library.cache_info().currsize:
        lib = library()
        if lib.host_vmm_retry_rollbacks():
            raise RuntimeError(lib.host_shared_last_error().decode())


class VmmMapping:
    def __init__(self, path, rows, width, dtype, group):
        if rows <= 0 or width <= 0 or dtype not in (torch.int8, torch.float32):
            raise ValueError("Expected positive INT8/FP32 table dimensions")
        self.rows, self.width, self.dtype = rows, width, dtype
        self.device = torch.device("npu", torch.npu.current_device())
        self.itemsize = torch.empty((), dtype=dtype, device="cpu").element_size()
        self.size = ((rows * width * self.itemsize + GIB - 1) // GIB) * GIB
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        stat = path.parent.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise ValueError("VMM capability directory must be owned by this uid and mode0700")
        self.lib = library()
        self.allocation = Allocation()
        self.tensor = self.ptrs = None
        self._weak_storage = None
        self.closed = False
        with torch.npu.device(self.device):
            rc = self.lib.host_shared_registered_alloc(
                self.size, self.device.index, os.fsencode(path), int(group.rank == 0), ctypes.byref(self.allocation)
            )
            if rc:
                raise RuntimeError(f"Host VMM: {self.lib.host_shared_last_error().decode()}")
            try:
                self.tensor = tensor_at(self.allocation.device_ptr, (rows, width), dtype, self.device)
                self._weak_storage = self.tensor.untyped_storage()._weak_ref()
                self.ptrs = torch.tensor(
                    [self.tensor.data_ptr() + start * width * self.itemsize for start in range(0, rows, 1 << 22)],
                    dtype=torch.int64,
                    device=self.device,
                )
            except Exception as exc:
                self._rollback(exc)
                raise
        print(f"ENGRAM_VMM_MAP bytes={self.size} rank={group.rank} device={self.device}", flush=True)

    def publish(self, start, source):
        if self.closed or self.tensor is None:
            raise RuntimeError("VMM mapping is closed or closing")
        if start < 0 or start + len(source) > self.rows or source.shape[1:] != (self.width,):
            raise ValueError("Publication outside table shape")
        # Startup staging only: this CANN rejects CPU->imported-host memcpy.
        self.tensor[start : start + len(source)].copy_(source.to(self.device))

    def lifecycle_counts(self):
        value = self.lib.host_vmm_lifecycle_counts()
        return value >> 32, value & 0xFFFFFFFF

    def _rollback(self, original_error):
        """Only for failed construction, before any users receive this mapping."""
        try:
            self.close()
        except Exception as cleanup_error:
            rc = self.lib.host_shared_registered_defer_free(ctypes.byref(self.allocation))
            if rc:
                raise RuntimeError(f"{original_error}; rollback ownership transfer failed: {rc}") from cleanup_error
            raise RuntimeError(f"{original_error}; deferred rollback: {cleanup_error}") from original_error

    def close(self):
        """After all launches/graphs stop; rejects outstanding tensor aliases.

        On failure, remains retryable. A caller must drop parameter/views before
        retrying; do not use this object once close has started.
        """
        if self.closed:
            return
        with torch.npu.device(self.device):
            torch.npu.synchronize()
            self.tensor = self.ptrs = None
            if self._weak_storage is not None:
                if not torch.UntypedStorage._expired(self._weak_storage):
                    raise RuntimeError("VMM tensor aliases still alive; release them before close")
                torch.UntypedStorage._free_weak_ref(self._weak_storage)
                self._weak_storage = None
            rc = self.lib.host_shared_registered_free(ctypes.byref(self.allocation))
            if rc:
                raise RuntimeError(f"VMM free: {self.lib.host_shared_last_error().decode()}")
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
