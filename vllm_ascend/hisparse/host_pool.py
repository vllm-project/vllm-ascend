# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Own pinned HiSparse Host storage and explicit AIV mapping.

The PD path allocates Host Source from MemFabric for direct local Pull destinations.
Core retains the mapping until its drained shutdown.
"""

from __future__ import annotations

import ctypes
import math
import mmap
from functools import wraps

import torch
from vllm.triton_utils import triton

_ACL_HOST_REGISTER_MAPPED = 0
_ACL_HAC_AIV = 3


def uses_memfabric(vllm_config) -> bool:
    transfer = vllm_config.kv_transfer_config if vllm_config else None
    if transfer is None:
        return False
    connectors = (
        (transfer.kv_connector_extra_config or {}).get("connectors", [])
        if transfer.kv_connector == "MultiConnector"
        else [{"kv_connector": transfer.kv_connector, "kv_connector_extra_config": transfer.kv_connector_extra_config}]
    )
    return any(
        child.get("kv_connector") in ("MooncakeConnectorV2", "MooncakePullConnector")
        and (child.get("kv_connector_extra_config") or {}).get("mooncake_transport") == "memfabric"
        for child in connectors
    )


class MappedHostTensor:
    """Retain a CPU allocation with its ACL or MemFabric device mapping.

    Close only after all streams and graphs using this mapping have completed
    and been destroyed. No destructor unmaps memory behind asynchronous work.
    """

    def __init__(self, tensor: torch.Tensor, *, memfabric: bool = False) -> None:
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("HiSparse host mapping requires a contiguous CPU tensor")
        self.tensor = tensor
        self.host_address = tensor.data_ptr()
        self.nbytes = tensor.numel() * tensor.element_size()
        if not self.nbytes:
            raise ValueError("Cannot register an empty HiSparse host allocation")
        self._memfabric = memfabric
        if memfabric:
            from memfabric_hybrid import offload

            self.device = torch.device("npu", torch.npu.current_device())
            self.device_ptr = offload.get_dva(self.host_address)
            if not self.device_ptr:
                raise RuntimeError("MemFabric did not expose a device address for HiSparse Host Source")
            self.closed = False
            return
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
        device_index = torch.npu.current_device()
        self.device = torch.device("npu", device_index)
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
        self.closed = False

    def address_of(self, tensor: torch.Tensor) -> int:
        offset = tensor.data_ptr() - self.host_address
        size = tensor.numel() * tensor.element_size()
        if self.closed or tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("HiSparse source must be a contiguous view of a live mapped allocation")
        if offset < 0 or offset + size > self.nbytes:
            raise ValueError("HiSparse source is outside its registered host allocation")
        return self.device_ptr + offset

    def close(self) -> None:
        if not self.closed:
            if not self._memfabric:
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
    # Mapping permits AIV access; the owner remains page-locked for DMA.
    allocator = torch.empty
    owner_bytes = allocated_bytes + mmap.PAGESIZE - 1
    if memfabric:
        from memfabric_hybrid import offload

        config = offload.OffloadConfig()
        config.device_id = torch.npu.current_device()
        pool_size = triton.cdiv(owner_bytes, 1024**3) * 1024**3
        config.reserve_size = pool_size
        config.alloc_size = pool_size
        config.world_size = 1
        config.rank_id = 0
        config.scene = offload.Scene.LOCAL
        config.flags = offload.OFFLOAD_FLAG_GIANT_PAGE
        if offload.initialize(config) != 0:
            raise RuntimeError("MemFabric HiSparse Host pool initialization failed")
        allocator = offload.empty
        # URMA registration must match the existing Offload allocation, including its padding.
        owner_bytes = pool_size
    owner = allocator([owner_bytes], dtype=torch.uint8, pin_memory=True)
    offset = (-owner.data_ptr()) % mmap.PAGESIZE
    raw = owner[offset : offset + allocated_bytes]
    mapping = MappedHostTensor(raw, memfabric=memfabric)
    tensor = raw[:nbytes].view(dtype).view(shape)
    # Both aliases retain the pinned allocation. Runtime owns the registration
    # through this base tensor until its explicit, drained shutdown.
    tensor._hisparse_registration = mapping
    return tensor


def install() -> None:
    from vllm.v1.hisparse import runtime
    from vllm.v1.hisparse.backend import get_hisparse_backend

    from vllm_ascend.hisparse.backend import AscendHiSparseBackend

    original = runtime.allocate_pinned_host_pool
    if getattr(original, "_ascend_pinned_private_pool", False):
        return

    @wraps(original)
    def allocate(size):
        backend = get_hisparse_backend()
        if isinstance(backend, AscendHiSparseBackend):
            from vllm.config import get_current_vllm_config_or_none

            config = get_current_vllm_config_or_none()
            payload = allocate_mapped_host((size,), torch.int8, memfabric=uses_memfabric(config))
            registration = payload._hisparse_registration
            registered = registration.tensor.view(torch.int8)
            registered._hisparse_registration = registration
            return registered[:size], registered
        return original(size)

    allocate.__dict__["_ascend_pinned_private_pool"] = True
    runtime.allocate_pinned_host_pool = allocate
