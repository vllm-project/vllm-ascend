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

# shortcut: TransferEngine has process-lifetime MRs; retain shared owners until worker exit.
_shared_owners: list[torch.Tensor] = []


def validate_hisparse_pd_order(vllm_config) -> None:
    transfer = vllm_config.kv_transfer_config
    if transfer is None:
        return
    extra = transfer.kv_connector_extra_config or {}
    children = (
        extra.get("connectors", [])
        if transfer.kv_connector == "MultiConnector"
        else [{"kv_connector": transfer.kv_connector, "kv_connector_extra_config": extra}]
    )
    names = [child.get("kv_connector") for child in children]
    for index, child in enumerate(children):
        if (
            names[index] == "MooncakeConnectorV2"
            and (child.get("kv_connector_extra_config") or {}).get("transfer_backend") == "memfabric"
        ):
            if "HiSparseConnector" not in names:
                raise ValueError("MemFabric requires Ascend's HiSparseConnector")
            if names.index("HiSparseConnector") > index:
                raise ValueError("Ascend HiSparseConnector must precede MooncakeConnectorV2 in MultiConnector")


def shared_source_enabled(vllm_config) -> bool:
    # Standalone HiSparse retains a Host pool per rank, including with PCP.
    transfer = vllm_config.kv_transfer_config
    if transfer is None or transfer.kv_connector != "MultiConnector":
        return False
    parallel = vllm_config.parallel_config
    return (
        parallel.tensor_parallel_size > 1
        and parallel.pipeline_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and parallel.world_size == parallel.tensor_parallel_size
        and parallel.distributed_executor_backend == "mp"
        and parallel.nnodes_within_dp == 1
    )


class MemFabricSharedHostRegion:
    """One Offload SHARED contribution, exposed through Core's region contract."""

    def __init__(self, tensor_sizes: list[int], *, group=None) -> None:
        from memfabric_hybrid import offload
        from vllm.utils.network_utils import get_open_port
        from vllm.v1.hisparse.runtime import check_hisparse_host_memory

        self._size = sum(tensor_sizes)
        if self._size <= 0 or any(size <= 0 for size in tensor_sizes):
            raise ValueError("Shared HiSparse Host tensors must be nonempty")
        rank = group.rank_in_group if group is not None else 0
        leader = rank == 0
        pool_size = triton.cdiv(self._size, 1024**3) * 1024**3
        port = get_open_port() if leader else None
        if group is not None:
            port = group.broadcast_object(port, src=0)
        config = offload.OffloadConfig()
        config.device_id = torch.npu.current_device()
        config.reserve_size = pool_size
        config.alloc_size = pool_size if leader else 0
        config.world_size = group.world_size if group is not None else 1
        config.rank_id = rank
        config.scene = offload.Scene.SHARED
        config.store_url = f"tcp://127.0.0.1:{port}"
        # The SDK owns both the shared storage and its per-device mappings.
        if offload.initialize(config) != 0:
            raise RuntimeError("MemFabric shared HiSparse Host pool initialization failed")
        if group is not None:
            group.barrier()
        result: tuple[int | None, int | None, int, str | None] | None = None
        owner = None
        if leader:
            try:
                check_hisparse_host_memory(pool_size)
                owner = offload.empty([pool_size], dtype=torch.int8, pin_memory=True)
                if owner.numel() != pool_size or not owner.is_contiguous():
                    raise RuntimeError("MemFabric returned an invalid shared Host allocation")
                mapping = MappedHostTensor(owner, memfabric=True)
                result = (owner.data_ptr(), mapping.device_ptr, pool_size, None)
            except Exception as exc:
                result = (None, None, pool_size, str(exc))
        if group is not None:
            result = group.broadcast_object(result, src=0)
        assert result is not None
        address, device_ptr, registered_size, error = result
        if error is not None:
            raise RuntimeError(f"MemFabric shared Host allocation failed: {error}")
        if not address or address % mmap.PAGESIZE:
            raise RuntimeError("MemFabric shared Host address is null or not page aligned")
        if registered_size != pool_size or not device_ptr:
            raise RuntimeError("MemFabric shared Host mapping differs from the initialized pool")
        if not leader:
            # Metadata-only view: SDK import supplies device access, not a follower CPU mapping.
            buffer = (ctypes.c_ubyte * pool_size).from_address(address)
            owner = torch.frombuffer(buffer, dtype=torch.int8)
            # Legacy VMM imports expose the common VA; mapped Host pools need a local DVA lookup.
            mapping = MappedHostTensor(
                owner, memfabric=True, shared_device_ptr=device_ptr if device_ptr == address else None
            )
        if owner is None:
            raise RuntimeError("MemFabric shared HiSparse Host allocation is missing")
        self.base_tensor = owner
        self.base_tensor._hisparse_registration = mapping
        _shared_owners.append(owner)

    def cleanup(self) -> None:
        # TransferEngine MRs retain the backing until process exit.
        pass


class MappedHostTensor:
    """Retain a CPU allocation with its ACL or MemFabric device mapping.

    Close only after all streams and graphs using this mapping have completed
    and been destroyed. No destructor unmaps memory behind asynchronous work.
    """

    def __init__(self, tensor: torch.Tensor, *, memfabric: bool = False, shared_device_ptr: int | None = None) -> None:
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("HiSparse host mapping requires a contiguous CPU tensor")
        self.tensor = tensor
        self.host_address = tensor.data_ptr()
        self.nbytes = tensor.numel() * tensor.element_size()
        if not self.nbytes:
            raise ValueError("Cannot register an empty HiSparse host allocation")
        if shared_device_ptr is not None and (not memfabric or shared_device_ptr != self.host_address):
            raise ValueError("Imported SHARED mappings require identical MemFabric Host/device addresses")
        self._memfabric = memfabric
        if memfabric:
            from memfabric_hybrid import offload

            self.device = torch.device("npu", torch.npu.current_device())
            self.device_ptr = offload.get_dva(self.host_address) if shared_device_ptr is None else shared_device_ptr
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


def allocate_mapped_host(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Allocate page-locked, aligned storage for both DMA and AIV access."""
    count = math.prod(shape)
    nbytes = count * torch.empty((), dtype=dtype).element_size()
    if nbytes <= 0:
        raise ValueError("HiSparse host allocation must be nonempty")
    allocated_bytes = triton.cdiv(nbytes, mmap.PAGESIZE) * mmap.PAGESIZE
    # Mapping permits AIV access; the owner remains page-locked for DMA.
    owner = torch.empty([allocated_bytes + mmap.PAGESIZE - 1], dtype=torch.uint8, pin_memory=True)
    offset = (-owner.data_ptr()) % mmap.PAGESIZE
    raw = owner[offset : offset + allocated_bytes]
    mapping = MappedHostTensor(raw)
    tensor = raw[:nbytes].view(dtype).view(shape)
    # Both aliases retain the pinned allocation. Runtime owns the registration
    # through this base tensor until its explicit, drained shutdown.
    tensor._hisparse_registration = mapping
    return tensor


class AscendHiSparseOffloadManager:
    def __init__(self, vllm_config, size: int) -> None:
        from vllm.distributed import get_tp_group

        if size <= 0:
            raise ValueError("HiSparse Source must be nonempty")
        group = get_tp_group() if shared_source_enabled(vllm_config) else None
        region = MemFabricSharedHostRegion([size], group=group)
        self.shared_region = region if group is not None else None
        self.is_host_writer = group is None or group.rank_in_group == 0
        self.source = region.base_tensor[:size]
        self.source._hisparse_registration = region.base_tensor._hisparse_registration
        self.source._hisparse_registration.offload_manager = self

    @classmethod
    def from_host_pool(cls, pool):
        registration = getattr(pool.registered, "_hisparse_registration", None)
        manager = getattr(registration, "offload_manager", None)
        if not isinstance(manager, cls):
            raise RuntimeError("Ascend HiSparse worker requires a manager-owned Source")
        return manager

    def bind_host_pool(self, pool, kv_cache_config) -> None:
        if self.from_host_pool(pool) is not self or pool.backing is None:
            raise RuntimeError("HiSparse Source does not belong to this manager")
        # Allocation is complete. Publish effective sharing to Core cleanup and PD.
        pool.shared_region = self.shared_region
        kv_cache_config.hisparse_shared_host_pool = self.shared_region is not None
        pool.kv_cache_config.hisparse_shared_host_pool = self.shared_region is not None


def install() -> None:
    from vllm.v1.hisparse import runtime
    from vllm.v1.hisparse.backend import get_hisparse_backend

    from vllm_ascend.hisparse.backend import AscendHiSparseBackend

    original = runtime.HiSparseHostPool.allocate
    if getattr(original, "_ascend_pinned_private_pool", False):
        return

    @wraps(original)
    def allocate(pool, size):
        if (
            pool.backing is not None
            or pool.kv_cache_config.hisparse_shared_host_pool
            or not isinstance(get_hisparse_backend(), AscendHiSparseBackend)
        ):
            return original(pool, size)
        runtime.check_hisparse_host_memory(size)
        payload = AscendHiSparseOffloadManager(pool.vllm_config, size).source
        registration = payload._hisparse_registration
        registered = registration.tensor.view(torch.int8)
        registered._hisparse_registration = registration
        pool.registered = registered
        pool.pinned = [registered]
        pool.backing = registered[:size]
        return pool.backing

    allocate.__dict__["_ascend_pinned_private_pool"] = True
    runtime.HiSparseHostPool.allocate = allocate
