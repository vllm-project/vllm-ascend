# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Own pinned HiSparse Host storage and its device mapping.

Core registers an aligned pageable allocation through ``pin_tensor``. AIV
mapping alone does not make that storage a valid asynchronous DMA destination
on Ascend. Use pinned storage and Core's existing release ownership.
Remove the allocation adapter when Core exposes a backend Host-allocation hook.
"""

from __future__ import annotations

import ctypes
import math
import mmap
from contextlib import ExitStack
from functools import wraps
from threading import Lock
from typing import TYPE_CHECKING

import torch
from vllm.triton_utils import triton

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig

_ACL_HOST_REGISTER_MAPPED = 0
_ACL_HAC_AIV = 3
_LEGACY_MEMFABRIC_POOL = Lock()


class MappedHostTensor:
    """Retain mapped CPU storage until explicit close.

    Close only after all streams and graphs using this mapping have completed
    and been destroyed. No destructor unmaps memory behind asynchronous work.
    """

    def __init__(self, tensor: torch.Tensor, *, memfabric_device_ptr: int | None = None) -> None:
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("HiSparse host mapping requires a contiguous CPU tensor")
        self.tensor = tensor
        self.host_address = tensor.data_ptr()
        self.nbytes = tensor.numel() * tensor.element_size()
        if not self.nbytes:
            raise ValueError("Cannot register an empty HiSparse host allocation")
        device_index = torch.npu.current_device()
        self.device = torch.device("npu", device_index)
        self.closed = False
        self._memfabric_cleanup: ExitStack | None = None
        if memfabric_device_ptr is not None:
            # MemFabric provides AIV host-memory mapping on older HDK releases such as 25.5.1.
            # Reuse the device address from the MemFabric arena's existing mapping.
            # The arena releases the mapping during cleanup.
            self.device_ptr = memfabric_device_ptr
            return
        # Native ACL AIV host-memory mapping requires HDK 26.1 or later.
        # Check support, then register the pinned host range. close() unregisters it.
        self._lib = ctypes.CDLL("libascendcl.so")
        self._lib.aclrtHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.aclrtHostRegister.restype = ctypes.c_int
        self._lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
        self._lib.aclrtHostUnregister.restype = ctypes.c_int
        self._lib.aclrtHostMemMapCapabilities.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self._lib.aclrtHostMemMapCapabilities.restype = ctypes.c_int
        capability = ctypes.c_int()
        status = self._lib.aclrtHostMemMapCapabilities(device_index, _ACL_HAC_AIV, ctypes.byref(capability))
        if status or not capability.value:
            raise RuntimeError(
                f"HiSparse requires AIV host-memory mapping: status={status}, capability={capability.value}"
            )
        # The upstream host-pool allocator supplies an aligned subview of a
        # larger backing allocation. Register that range, not its unaligned
        # storage base; retained tensor ownership still retains the backing.
        mapped = ctypes.c_void_p()
        status = self._lib.aclrtHostRegister(
            self.host_address, self.nbytes, _ACL_HOST_REGISTER_MAPPED, ctypes.byref(mapped)
        )
        if status:
            raise RuntimeError(f"aclrtHostRegister for HiSparse failed: {status}")
        device_ptr = mapped.value
        if device_ptr is None:
            self._lib.aclrtHostUnregister(self.host_address)
            raise RuntimeError("aclrtHostRegister returned a null device pointer for HiSparse")
        self.device_ptr = device_ptr

    def address_of(self, tensor: torch.Tensor) -> int:
        offset = tensor.data_ptr() - self.host_address
        size = tensor.numel() * tensor.element_size()
        if self.closed or tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("HiSparse source must be a contiguous view of a live mapped allocation")
        if offset < 0 or offset + size > self.nbytes:
            raise ValueError("HiSparse source is outside its registered host allocation")
        return self.device_ptr + offset

    def close(self) -> None:
        if self.closed:
            return
        if self._memfabric_cleanup is not None:
            self._memfabric_cleanup.close()
        else:
            status = self._lib.aclrtHostUnregister(self.host_address)
            if status:
                raise RuntimeError(f"aclrtHostUnregister for HiSparse failed: {status}")
        self.closed = True


def allocate_mapped_host(shape: tuple[int, ...], dtype: torch.dtype, *, memfabric: bool = False) -> torch.Tensor:
    """Allocate page-locked, aligned storage for both DMA and AIV access."""
    count = math.prod(shape)
    nbytes = count * torch.empty((), dtype=dtype).element_size()
    if nbytes <= 0:
        raise ValueError("HiSparse host allocation must be nonempty")
    allocated_bytes = triton.cdiv(nbytes, mmap.PAGESIZE) * mmap.PAGESIZE
    if memfabric:
        from memfabric_hybrid import offload

        config = offload.OffloadConfig()
        config.device_id = torch.npu.current_device()
        config.reserve_size = triton.cdiv(allocated_bytes, 1024**3) * 1024**3
        config.alloc_size = config.reserve_size
        config.world_size = 1
        config.rank_id = 0
        config.scene = offload.Scene.LOCAL
        with ExitStack() as cleanup:
            # SDK 1.3.0 exposes only a process-wide allocator.
            if hasattr(offload, "Offload"):
                arena = offload.Offload()
            else:
                if not _LEGACY_MEMFABRIC_POOL.acquire(blocking=False):
                    raise RuntimeError("This MemFabric SDK supports one HiSparse Host pool per process")
                cleanup.callback(_LEGACY_MEMFABRIC_POOL.release)
                arena = offload
            status = arena.initialize(config)
            if status:
                raise RuntimeError(f"Failed to initialize the MemFabric HiSparse Host pool: {status}")
            cleanup.callback(arena.uninitialize)
            raw = arena.empty([allocated_bytes], dtype=torch.uint8, pin_memory=True)
            if raw.data_ptr() % mmap.PAGESIZE:
                raise RuntimeError("MemFabric HiSparse Host allocation is not page aligned")
            device_ptr = arena.get_dva(raw.data_ptr())
            if not device_ptr:
                raise RuntimeError("MemFabric did not provide a HiSparse device address")
            mapping = MappedHostTensor(raw, memfabric_device_ptr=device_ptr)
            mapping._memfabric_cleanup = cleanup.pop_all()
    else:
        # Align a view of PTA's pinned allocation without creating a second pool.
        owner = torch.empty(allocated_bytes + mmap.PAGESIZE - 1, dtype=torch.uint8, pin_memory=True)
        offset = (-owner.data_ptr()) % mmap.PAGESIZE
        raw = owner[offset : offset + allocated_bytes]
        mapping = MappedHostTensor(raw)
    tensor = raw[:nbytes].view(dtype).view(shape)
    # Both aliases retain the pinned allocation. Runtime owns the registration
    # through this base tensor until its explicit, drained shutdown.
    tensor._hisparse_registration = mapping
    return tensor


def _uses_memfabric(transfer: KVTransferConfig | None) -> bool:
    if transfer is None:
        return False
    extra = transfer.kv_connector_extra_config or {}
    if transfer.kv_connector == "MultiConnector":
        connectors = extra.get("connectors", [])
    else:
        connectors = [{"kv_connector": transfer.kv_connector, "kv_connector_extra_config": extra}]
    return any(
        connector.get("kv_connector") == "HiSparseConnector"
        and (connector.get("kv_connector_extra_config") or {}).get("transfer_backend") == "memfabric"
        for connector in connectors
    )


def install() -> None:
    from vllm.config import get_current_vllm_config_or_none
    from vllm.v1.hisparse import runtime
    from vllm.v1.hisparse.backend import get_hisparse_backend

    from vllm_ascend import ascend_config
    from vllm_ascend.hisparse.backend import AscendHiSparseBackend

    original = runtime.allocate_pinned_host_pool
    if getattr(original, "_ascend_pinned_private_pool", False):
        return

    @wraps(original)
    def allocate(size):
        backend = get_hisparse_backend()
        if isinstance(backend, AscendHiSparseBackend):
            config = get_current_vllm_config_or_none() or ascend_config._INIT_VLLM_CONFIG
            transfer = config.kv_transfer_config if config is not None else None
            payload = allocate_mapped_host((size,), torch.int8, memfabric=_uses_memfabric(transfer))
            registration = payload._hisparse_registration
            registered = registration.tensor.view(torch.int8)
            registered._hisparse_registration = registration
            return registered[:size], registered
        return original(size)

    allocate.__dict__["_ascend_pinned_private_pool"] = True
    runtime.allocate_pinned_host_pool = allocate
