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
        port = error = None
        if leader:
            try:
                # Offload commits the physical pool in initialize(), before empty().
                check_hisparse_host_memory(pool_size)
                port = get_open_port()
            except Exception as exc:
                error = str(exc)
        if group is not None:
            port, error = group.broadcast_object((port, error), src=0)
        if error is not None:
            raise RuntimeError(f"MemFabric shared Host allocation failed: {error}")
        config = offload.OffloadConfig()
        config.device_id = torch.npu.current_device()
        config.reserve_size = pool_size
        config.alloc_size = pool_size if leader else 0
        config.world_size = group.world_size if group is not None else 1
        config.rank_id = rank
        config.scene = offload.Scene.SHARED
        config.store_url = f"tcp://127.0.0.1:{port}"
        # The SDK owns both the shared storage and its per-device mappings.
        try:
            status = offload.initialize(config)
            init_error = f"status {status}" if status != 0 else None
        except Exception as exc:
            init_error = str(exc)
        if group is not None:
            failures = torch.tensor([init_error is not None], dtype=torch.int32)
            group.cpu_all_reduce_(failures)
            failed = bool(failures.item())
        else:
            failed = init_error is not None
        if failed:
            detail = f": {init_error}" if init_error is not None else " on another TP rank"
            raise RuntimeError(f"MemFabric shared HiSparse Host pool initialization failed{detail}")
        if group is not None:
            group.barrier()
        self._bind_host_pool(offload, pool_size, group, leader=leader)

    def _bind_host_pool(self, offload, pool_size: int, group, *, leader: bool) -> None:
        result: tuple[int | None, int | None, int, str | None] | None = None
        owner = None
        if leader:
            try:
                owner = offload.empty([pool_size], dtype=torch.int8, pin_memory=True)
                if owner.numel() != pool_size or not owner.is_contiguous():
                    raise RuntimeError("MemFabric returned an invalid shared Host allocation")
                device_ptr = offload.get_dva(owner.data_ptr())
                if not device_ptr:
                    raise RuntimeError("MemFabric did not expose a device address for HiSparse Host Source")
                mapping = MappedHostTensor(owner, memfabric_device_ptr=device_ptr)
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
            local_device_ptr = device_ptr if device_ptr == address else offload.get_dva(address)
            if not local_device_ptr:
                raise RuntimeError("MemFabric did not expose a device address for HiSparse Host Source")
            mapping = MappedHostTensor(owner, memfabric_device_ptr=local_device_ptr)
        if owner is None:
            raise RuntimeError("MemFabric shared HiSparse Host allocation is missing")
        self.base_tensor = owner
        self.base_tensor._hisparse_registration = mapping
        _shared_owners.append(owner)

    def cleanup(self) -> None:
        # TransferEngine MRs retain the backing until process exit.
        pass


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
        self._lib: ctypes.CDLL | None = None
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
        elif self._lib is not None:
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
        from vllm.utils.network_utils import get_open_port

        config = offload.OffloadConfig()
        config.device_id = torch.npu.current_device()
        config.reserve_size = triton.cdiv(allocated_bytes, 1024**3) * 1024**3
        config.alloc_size = config.reserve_size
        config.world_size = 1
        config.rank_id = 0
        config.scene = offload.Scene.SHARED
        config.store_url = f"tcp://127.0.0.1:{get_open_port()}"
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


def uses_memfabric_pd(transfer: KVTransferConfig | None) -> bool:
    if transfer is None or transfer.kv_connector != "MultiConnector":
        return False
    for child in (transfer.kv_connector_extra_config or {}).get("connectors", []):
        if child.get("kv_connector") != "MooncakeConnectorV2" or child.get("kv_role") not in ("kv_consumer", "kv_both"):
            continue
        extra = child.get("kv_connector_extra_config") or {}
        if extra.get("transfer_backend") == "memfabric" and not extra.get("use_layerwise", False):
            return True
    return False


class AscendHiSparseOffloadManager:
    def __init__(self, size: int, *, shared: bool) -> None:
        from vllm.distributed import get_tp_group

        if size <= 0:
            raise ValueError("HiSparse Source must be nonempty")
        group = get_tp_group() if shared else None
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


def install() -> None:
    from vllm.config import get_current_vllm_config_or_none
    from vllm.v1.hisparse import runtime
    from vllm.v1.hisparse.backend import get_hisparse_backend

    from vllm_ascend import ascend_config
    from vllm_ascend.hisparse.backend import AscendHiSparseBackend

    original_pinned = runtime.allocate_pinned_host_pool
    if getattr(original_pinned, "_ascend_pinned_private_pool", False):
        return

    @wraps(original_pinned)
    def allocate_pinned(size):
        backend = get_hisparse_backend()
        if isinstance(backend, AscendHiSparseBackend):
            config = get_current_vllm_config_or_none() or ascend_config._INIT_VLLM_CONFIG
            transfer = config.kv_transfer_config if config is not None else None
            payload = allocate_mapped_host((size,), torch.int8, memfabric=_uses_memfabric(transfer))
            registration = payload._hisparse_registration
            registered = registration.tensor.view(torch.int8)
            registered._hisparse_registration = registration
            return registered[:size], registered
        return original_pinned(size)

    allocate_pinned.__dict__["_ascend_pinned_private_pool"] = True
    runtime.allocate_pinned_host_pool = allocate_pinned

    original = runtime.HiSparseHostPool.allocate
    if getattr(original, "_ascend_memfabric_host_pool", False):
        return

    @wraps(original)
    def allocate(pool, size):
        shared = pool.kv_cache_config.hisparse_shared_host_pool
        if (
            pool.backing is not None
            or not isinstance(get_hisparse_backend(), AscendHiSparseBackend)
            or not (shared or uses_memfabric_pd(pool.vllm_config.kv_transfer_config))
        ):
            return original(pool, size)
        manager = AscendHiSparseOffloadManager(size, shared=shared)
        registration = manager.source._hisparse_registration
        registered = registration.tensor.view(torch.int8)
        registered._hisparse_registration = registration
        pool.registered = registered
        pool.pinned = [registered]
        pool.shared_region = manager.shared_region
        pool.backing = registered[:size]
        return pool.backing

    allocate.__dict__["_ascend_memfabric_host_pool"] = True
    runtime.HiSparseHostPool.allocate = allocate  # type: ignore[method-assign, assignment]
