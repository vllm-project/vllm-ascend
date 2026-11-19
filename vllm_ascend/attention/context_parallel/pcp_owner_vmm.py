# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Device-side VMM owner slabs for the PCP prefill KV cache.

A slab is a virtual address range that is *complete* on every PCP rank: the
same slab byte offset addresses the same logical KV bytes everywhere.  Only
``1 / pcp_size`` of the two-megabyte chunks backing that range is allocated on
a given rank.  By default each rank owns one contiguous run of chunks (the
placement that matches a PCP sequence chunk); the modulo placement is kept as
a compatibility path.  Every other rank maps the chunk through a CANN V2
shareable physical handle.

Every rank therefore keeps a non-owning Torch-NPU tensor that spans the full
logical slab.  Stock attention kernels address it exactly like the replicated
cache; a write to a peer chunk lands in the owner's HBM and a read of a peer
chunk reads the owner's HBM.  No HCCL staging, no device-to-host copy and no
request-time import/map is involved.

The physical layout is deliberately chunk-granular rather than page-granular:
CANN maps host/device virtual memory in two-megabyte units, so a page-level
1/PCP split would need sub-granularity physical allocations that the runtime
does not provide.

All export/import/map work happens once, during ``initialize_kv_cache``.  A
failure in any rank is reported to the whole PCP group over the Gloo CPU group
and every rank then falls back to a replicated allocation in the caller.  The
fallback is safe because this mode does not change ``num_blocks``: the KV
budget was planned for the replicated layout.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import logger

CANN_VMM_GRANULARITY_BYTES = 2 * 1024 * 1024
# Contiguous gives every rank one contiguous physical run, so a single
# aclrtMapMem call covers its whole share.  This is a *storage* placement and
# is independent of which rank computes a token.  Modulo is retained as a
# compatibility path.
SLAB_PLACEMENT_CONTIGUOUS = "contiguous"
SLAB_PLACEMENT_MODULO = "modulo"
SLAB_PLACEMENTS = (SLAB_PLACEMENT_CONTIGUOUS, SLAB_PLACEMENT_MODULO)
ACL_SUCCESS = 0
ACL_HBM_MEM_NORMAL = 5
ACL_MEM_LOCATION_TYPE_DEVICE = 1
ACL_MEM_ALLOCATION_TYPE_PINNED = 0
ACL_MEM_HANDLE_TYPE_NONE = 0
ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM = 0
ACL_RT_MEM_ACCESS_FLAGS_READWRITE = 0x3
ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT = 0x1
ACL_RT_VMM_EXPORT_FLAG_DEFAULT = 0
# CANN 9.x V2 shares aclrtMemFabricHandle { uint8_t data[128]; } through
# these calls (acl_rt.h).  Truncating it to uint64 passes a pointer that the
# runtime reads 128 bytes from, which can select the wrong owner handle.
ACL_V2_SHAREABLE_HANDLE_BYTES = 128

_REQUIRED_ACL_SYMBOLS = (
    "aclrtSetDevice",
    "aclrtDeviceGetBareTgid",
    "aclrtDeviceCanAccessPeer",
    "aclrtDeviceEnablePeerAccess",
    "aclrtDeviceDisablePeerAccess",
    "aclrtMemGetAllocationGranularity",
    "aclrtMemExportToShareableHandleV2",
    "aclrtMemSetPidToShareableHandleV2",
    "aclrtMemImportFromShareableHandleV2",
    "aclrtReserveMemAddress",
    "aclrtReleaseMemAddress",
    "aclrtMallocPhysical",
    "aclrtFreePhysical",
    "aclrtMapMem",
    "aclrtUnmapMem",
    "aclrtMemSetAccess",
)


class PcpOwnerVmmUnavailable(RuntimeError):
    """The installed runtime cannot provide the owner-slab VMM seam."""


class PcpOwnerVmmError(RuntimeError):
    """An AscendCL VMM operation returned a non-success status."""

    def __init__(self, operation: str, status: int) -> None:
        self.operation = operation
        self.status = status
        super().__init__(f"{operation} failed with aclError={status}")


class PcpOwnerVmmRollbackError(RuntimeError):
    """A VMM operation and its immediate rollback both failed."""

    def __init__(self, *, operation_error: BaseException, rollback_error: BaseException) -> None:
        self.operation_error = operation_error
        self.rollback_error = rollback_error
        super().__init__(
            "owner-slab VMM operation failed and rollback was incomplete: "
            f"{operation_error}; rollback: {rollback_error}"
        )


class _AclrtMemLocation(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("type", ctypes.c_int),
    ]


class _AclrtPhysicalMemProp(ctypes.Structure):
    _fields_ = [
        ("handleType", ctypes.c_int),
        ("allocationType", ctypes.c_int),
        ("memAttr", ctypes.c_int),
        ("location", _AclrtMemLocation),
        ("reserve", ctypes.c_uint64),
    ]


class _AclrtMemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_int),
        ("location", _AclrtMemLocation),
        ("rsv", ctypes.c_uint8 * 12),
    ]


@dataclass(frozen=True)
class SlabChunkPlan:
    """Pure chunk-to-owner arithmetic for one slab.

    Ownership is assigned in two-megabyte chunks so a rank can hold one
    contiguous physical allocation while still mapping only its own chunks.
    ``contiguous`` gives every rank one contiguous virtual byte range and one
    map call per owner; ``modulo`` interleaves chunks and remains available as
    a compatibility path.  Neither placement affects where a token is
    computed: the full VA lets any rank write any page.
    """

    size_bytes: int
    pcp_size: int
    rank: int
    granularity: int = CANN_VMM_GRANULARITY_BYTES
    placement: str = SLAB_PLACEMENT_CONTIGUOUS

    def __post_init__(self) -> None:
        if self.size_bytes <= 0:
            raise ValueError(f"slab size must be positive, got {self.size_bytes}")
        if self.pcp_size <= 0:
            raise ValueError(f"pcp_size must be positive, got {self.pcp_size}")
        if not 0 <= self.rank < self.pcp_size:
            raise ValueError(f"rank={self.rank} is outside PCP size {self.pcp_size}")
        if self.granularity <= 0:
            raise ValueError(f"granularity must be positive, got {self.granularity}")
        if self.placement not in SLAB_PLACEMENTS:
            raise ValueError(f"placement must be one of {SLAB_PLACEMENTS}, got {self.placement!r}")

    @property
    def chunk_count(self) -> int:
        return (self.size_bytes + self.granularity - 1) // self.granularity

    @property
    def virtual_bytes(self) -> int:
        return self.chunk_count * self.granularity

    @property
    def chunks_per_rank(self) -> int:
        return (self.chunk_count + self.pcp_size - 1) // self.pcp_size

    @property
    def first_chunk(self) -> int:
        return self.rank * self.chunks_per_rank

    def owner_of(self, chunk_index: int) -> int:
        if not 0 <= chunk_index < self.chunk_count:
            raise ValueError(f"chunk_index={chunk_index} is outside {self.chunk_count} chunks")
        if self.placement == SLAB_PLACEMENT_MODULO:
            return chunk_index % self.pcp_size
        return min(chunk_index // self.chunks_per_rank, self.pcp_size - 1)

    @property
    def own_chunks(self) -> tuple[int, ...]:
        if self.placement == SLAB_PLACEMENT_MODULO:
            return tuple(range(self.rank, self.chunk_count, self.pcp_size))
        start = self.first_chunk
        end = min(start + self.chunks_per_rank, self.chunk_count)
        return tuple(range(start, end))

    @property
    def own_bytes(self) -> int:
        return len(self.own_chunks) * self.granularity

    def offset_in_owner(self, chunk_index: int) -> int:
        """Byte offset of ``chunk_index`` inside its owner's physical block."""
        owner = self.owner_of(chunk_index)
        if self.placement == SLAB_PLACEMENT_MODULO:
            return ((chunk_index - owner) // self.pcp_size) * self.granularity
        return (chunk_index - owner * self.chunks_per_rank) * self.granularity


@dataclass(frozen=True)
class AclShareableHandle:
    """Opaque CANN V2 handle; safe to exchange through a CPU object group."""

    payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.payload) != ACL_V2_SHAREABLE_HANDLE_BYTES:
            raise ValueError(f"V2 shareable handle must contain exactly {ACL_V2_SHAREABLE_HANDLE_BYTES} bytes")

    def __repr__(self) -> str:
        return f"AclShareableHandle(payload=<redacted {ACL_V2_SHAREABLE_HANDLE_BYTES} bytes>)"


@dataclass(frozen=True)
class AclPhysicalHandle:
    value: int
    size_bytes: int
    device_index: int


def _physical_memory_properties(device_index: int) -> _AclrtPhysicalMemProp:
    return _AclrtPhysicalMemProp(
        handleType=ACL_MEM_HANDLE_TYPE_NONE,
        allocationType=ACL_MEM_ALLOCATION_TYPE_PINNED,
        memAttr=ACL_HBM_MEM_NORMAL,
        location=_AclrtMemLocation(id=device_index, type=ACL_MEM_LOCATION_TYPE_DEVICE),
        reserve=0,
    )


def _configure_acl_signatures(library: object) -> None:
    signatures: dict[str, tuple[list[object], object]] = {
        "aclrtSetDevice": ([ctypes.c_int32], ctypes.c_int),
        "aclrtDeviceGetBareTgid": ([ctypes.POINTER(ctypes.c_int32)], ctypes.c_int),
        "aclrtDeviceCanAccessPeer": (
            [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32],
            ctypes.c_int,
        ),
        "aclrtDeviceEnablePeerAccess": ([ctypes.c_int32, ctypes.c_uint32], ctypes.c_int),
        "aclrtDeviceDisablePeerAccess": ([ctypes.c_int32], ctypes.c_int),
        "aclrtMemGetAllocationGranularity": (
            [ctypes.POINTER(_AclrtPhysicalMemProp), ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)],
            ctypes.c_int,
        ),
        "aclrtMemExportToShareableHandleV2": (
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p],
            ctypes.c_int,
        ),
        "aclrtMemSetPidToShareableHandleV2": (
            [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t],
            ctypes.c_int,
        ),
        "aclrtMemImportFromShareableHandleV2": (
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)],
            ctypes.c_int,
        ),
        "aclrtReserveMemAddress": (
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint64],
            ctypes.c_int,
        ),
        "aclrtReleaseMemAddress": ([ctypes.c_void_p], ctypes.c_int),
        "aclrtMallocPhysical": (
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.POINTER(_AclrtPhysicalMemProp), ctypes.c_uint64],
            ctypes.c_int,
        ),
        "aclrtFreePhysical": ([ctypes.c_void_p], ctypes.c_int),
        "aclrtMapMem": (
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint64],
            ctypes.c_int,
        ),
        "aclrtUnmapMem": ([ctypes.c_void_p], ctypes.c_int),
        "aclrtMemSetAccess": (
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(_AclrtMemAccessDesc), ctypes.c_size_t],
            ctypes.c_int,
        ),
    }
    for name, (argtypes, restype) in signatures.items():
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = restype


def _load_acl_library(library_path: str, loader: Callable[..., object]) -> object:
    try:
        library = loader(library_path)
    except OSError as error:
        raise PcpOwnerVmmUnavailable(f"cannot load AscendCL library {library_path!r}: {error}") from error
    missing = [name for name in _REQUIRED_ACL_SYMBOLS if not hasattr(library, name)]
    if missing:
        raise PcpOwnerVmmUnavailable("AscendCL runtime is missing owner-slab VMM symbols: " + ", ".join(missing))
    _configure_acl_signatures(library)
    return library


def _tensor_constructors(torch_npu_module: object) -> tuple[Any, Any]:
    extension = getattr(torch_npu_module, "_C", None)
    storage_constructor = getattr(extension, "_construct_storage_from_data_pointer", None)
    tensor_constructor = getattr(extension, "_construct_NPU_Tensor_From_Storage_And_Metadata", None)
    if not callable(storage_constructor) or not callable(tensor_constructor):
        raise PcpOwnerVmmUnavailable("torch_npu lacks the external-storage tensor constructors")
    return storage_constructor, tensor_constructor


class AclVmmBackend:
    """Thin owner-slab wrapper over the CANN 9.x AscendCL VMM API."""

    def __init__(
        self,
        *,
        library: object | None = None,
        library_path: str = "libascendcl.so",
        loader: Callable[..., object] = ctypes.CDLL,
    ) -> None:
        if library is None:
            self._library = _load_acl_library(library_path, loader)
        else:
            missing = [name for name in _REQUIRED_ACL_SYMBOLS if not hasattr(library, name)]
            if missing:
                raise PcpOwnerVmmUnavailable(
                    "AscendCL runtime is missing owner-slab VMM symbols: " + ", ".join(missing)
                )
            _configure_acl_signatures(library)
            self._library = library
        self._peer_access_lock = threading.Lock()
        self._peer_access_enabled: set[tuple[int, int]] = set()

    def _call(self, operation: str, *args: Any) -> None:
        result = getattr(self._library, operation)(*args)
        raw_status = getattr(result, "value", result)
        try:
            status = int(raw_status)
        except (TypeError, ValueError) as error:
            raise PcpOwnerVmmUnavailable(f"{operation} returned a non-integral ACL status: {result!r}") from error
        if status != ACL_SUCCESS:
            raise PcpOwnerVmmError(operation, status)

    def set_device(self, device_index: int) -> None:
        if device_index < 0:
            raise ValueError("device_index must be non-negative")
        self._call("aclrtSetDevice", device_index)

    def bare_tgid(self, device_index: int) -> int:
        self.set_device(device_index)
        bare_tgid = ctypes.c_int32()
        self._call("aclrtDeviceGetBareTgid", ctypes.byref(bare_tgid))
        if bare_tgid.value <= 0:
            raise PcpOwnerVmmUnavailable("aclrtDeviceGetBareTgid returned a non-positive process id")
        return int(bare_tgid.value)

    def enable_peer_access(self, device_index: int, peer_device_indices: Sequence[int]) -> None:
        """Enable every directed local->peer access pair exactly once."""
        peers = tuple(sorted({int(peer) for peer in peer_device_indices if int(peer) != device_index}))
        if any(peer < 0 for peer in peers):
            raise ValueError("peer device indices must be non-negative")
        with self._peer_access_lock:
            for peer in peers:
                pair = (device_index, peer)
                if pair in self._peer_access_enabled:
                    continue
                self.set_device(device_index)
                can_access = ctypes.c_int32()
                self._call("aclrtDeviceCanAccessPeer", ctypes.byref(can_access), device_index, peer)
                if can_access.value != 1:
                    raise PcpOwnerVmmUnavailable(
                        f"aclrtDeviceCanAccessPeer rejected directed device pair {device_index}->{peer}"
                    )
                self._call("aclrtDeviceEnablePeerAccess", peer, 0)
                self._peer_access_enabled.add(pair)

    def allocate_granularity(self, device_index: int) -> int:
        self.set_device(device_index)
        properties = _physical_memory_properties(device_index)
        granularity = ctypes.c_size_t()
        self._call(
            "aclrtMemGetAllocationGranularity",
            ctypes.byref(properties),
            ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM,
            ctypes.byref(granularity),
        )
        if granularity.value <= 0:
            raise PcpOwnerVmmUnavailable("aclrtMemGetAllocationGranularity returned zero")
        return int(granularity.value)

    def reserve_address(self, *, size_bytes: int, device_index: int) -> int:
        if size_bytes <= 0 or size_bytes % CANN_VMM_GRANULARITY_BYTES:
            raise ValueError("reserved size must be a positive 2-MiB multiple")
        self.set_device(device_index)
        virtual_address = ctypes.c_void_p()
        self._call("aclrtReserveMemAddress", ctypes.byref(virtual_address), size_bytes, 0, None, 0)
        base_address = int(virtual_address.value or 0)
        if base_address <= 0 or base_address % CANN_VMM_GRANULARITY_BYTES:
            invalid = ValueError(
                f"aclrtReserveMemAddress returned a null or non-2-MiB-aligned address: {base_address:#x}"
            )
            if base_address:
                try:
                    self._call("aclrtReleaseMemAddress", ctypes.c_void_p(base_address))
                except BaseException as rollback_error:
                    raise PcpOwnerVmmRollbackError(operation_error=invalid, rollback_error=rollback_error) from invalid
            raise invalid
        return base_address

    def release_address(self, *, base_address: int, device_index: int) -> None:
        if base_address <= 0:
            raise ValueError("base_address must be positive")
        self.set_device(device_index)
        self._call("aclrtReleaseMemAddress", ctypes.c_void_p(base_address))

    def malloc_physical(self, *, size_bytes: int, device_index: int) -> AclPhysicalHandle:
        if size_bytes <= 0 or size_bytes % CANN_VMM_GRANULARITY_BYTES:
            raise ValueError("physical size must be a positive 2-MiB multiple")
        self.set_device(device_index)
        handle = ctypes.c_void_p()
        self._call(
            "aclrtMallocPhysical",
            ctypes.byref(handle),
            size_bytes,
            ctypes.byref(_physical_memory_properties(device_index)),
            0,
        )
        if not handle.value:
            raise PcpOwnerVmmUnavailable("aclrtMallocPhysical succeeded but returned a null handle")
        return AclPhysicalHandle(value=int(handle.value), size_bytes=size_bytes, device_index=device_index)

    def free_physical(self, *, handle: AclPhysicalHandle, device_index: int) -> None:
        if handle.device_index != device_index:
            raise ValueError("physical handle belongs to another device")
        self.set_device(device_index)
        self._call("aclrtFreePhysical", ctypes.c_void_p(handle.value))

    def export(self, *, handle: AclPhysicalHandle, device_index: int) -> AclShareableHandle:
        if handle.device_index != device_index:
            raise ValueError("physical handle belongs to another device")
        self.set_device(device_index)
        output = (ctypes.c_ubyte * ACL_V2_SHAREABLE_HANDLE_BYTES)()
        self._call(
            "aclrtMemExportToShareableHandleV2",
            ctypes.c_void_p(handle.value),
            ACL_RT_VMM_EXPORT_FLAG_DEFAULT,
            ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT,
            ctypes.byref(output),
        )
        return AclShareableHandle(bytes(output))

    def authorize(self, *, shareable_handle: AclShareableHandle, bare_tgids: Sequence[int], device_index: int) -> None:
        trusted = tuple(dict.fromkeys(int(pid) for pid in bare_tgids))
        if not trusted:
            return
        if any(pid <= 0 for pid in trusted):
            raise ValueError("bare TGIDs must be positive")
        self.set_device(device_index)
        handle_buffer = (ctypes.c_ubyte * ACL_V2_SHAREABLE_HANDLE_BYTES).from_buffer_copy(shareable_handle.payload)
        pid_buffer = (ctypes.c_int32 * len(trusted))(*trusted)
        self._call(
            "aclrtMemSetPidToShareableHandleV2",
            ctypes.byref(handle_buffer),
            ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT,
            pid_buffer,
            len(trusted),
        )

    def import_handle(self, *, shareable_handle: AclShareableHandle, device_index: int) -> AclPhysicalHandle:
        self.set_device(device_index)
        handle_buffer = (ctypes.c_ubyte * ACL_V2_SHAREABLE_HANDLE_BYTES).from_buffer_copy(shareable_handle.payload)
        imported = ctypes.c_void_p()
        self._call(
            "aclrtMemImportFromShareableHandleV2",
            ctypes.byref(handle_buffer),
            ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT,
            0,
            ctypes.byref(imported),
        )
        if not imported.value:
            raise PcpOwnerVmmUnavailable("aclrtMemImportFromShareableHandleV2 returned a null handle")
        # The imported handle spans the peer's whole physical block.  The size
        # is filled in by the caller, which knows the owner chunk count.
        return AclPhysicalHandle(value=int(imported.value), size_bytes=0, device_index=device_index)

    def map(
        self,
        *,
        base_address: int,
        size_bytes: int,
        offset_bytes: int,
        handle: AclPhysicalHandle,
        device_index: int,
    ) -> None:
        if base_address % CANN_VMM_GRANULARITY_BYTES or size_bytes % CANN_VMM_GRANULARITY_BYTES:
            raise ValueError("mapped VA and size must be 2-MiB aligned")
        if offset_bytes % CANN_VMM_GRANULARITY_BYTES:
            raise ValueError("physical mapping offset must be 2-MiB aligned")
        self.set_device(device_index)
        self._call(
            "aclrtMapMem",
            ctypes.c_void_p(base_address),
            size_bytes,
            offset_bytes,
            ctypes.c_void_p(handle.value),
            0,
        )
        access = _AclrtMemAccessDesc(
            flags=ACL_RT_MEM_ACCESS_FLAGS_READWRITE,
            location=_AclrtMemLocation(id=device_index, type=ACL_MEM_LOCATION_TYPE_DEVICE),
        )
        try:
            self._call(
                "aclrtMemSetAccess",
                ctypes.c_void_p(base_address),
                size_bytes,
                ctypes.byref(access),
                1,
            )
        except BaseException as operation_error:
            try:
                self._call("aclrtUnmapMem", ctypes.c_void_p(base_address))
            except BaseException as rollback_error:
                raise PcpOwnerVmmRollbackError(
                    operation_error=operation_error, rollback_error=rollback_error
                ) from operation_error
            raise

    def unmap(self, *, base_address: int, device_index: int) -> None:
        self.set_device(device_index)
        self._call("aclrtUnmapMem", ctypes.c_void_p(base_address))


@dataclass
class _ExternalTensor:
    """Keep a non-owning Torch-NPU storage and its root tensor alive."""

    storage: object
    tensor: torch.Tensor
    base_address: int
    size_bytes: int


def bind_external_tensor(*, base_address: int, size_bytes: int, device_index: int) -> _ExternalTensor:
    """Build a non-owning uint8 tensor over mapped NPU virtual memory."""
    if base_address <= 0 or size_bytes <= 0:
        raise ValueError("external tensor address and size must be positive")
    try:
        torch_npu_module = importlib.import_module("torch_npu")
    except ImportError as error:
        raise PcpOwnerVmmUnavailable(f"Torch-NPU tensor binding is unavailable: {error}") from error
    storage_constructor, tensor_constructor = _tensor_constructors(torch_npu_module)
    device = torch.device(f"npu:{device_index}")
    dtype = torch.uint8
    storage = None
    tensor = None
    try:
        storage = storage_constructor(base_address, device, size_bytes)
        metadata = {
            "data_ptr": base_address,
            "device": device,
            "nbytes": size_bytes,
            "dtype": dtype,
            "size": (size_bytes,),
            "stride": (1,),
            "storage_offset": 0,
        }
        tensor = tensor_constructor(metadata, storage)
        if int(tensor.data_ptr()) != base_address:
            raise PcpOwnerVmmError("external tensor data_ptr mismatch", -1)
        if int(tensor.numel()) != size_bytes or int(tensor.element_size()) != 1:
            raise PcpOwnerVmmError("external tensor does not span the mapped byte range", -1)
        if tensor.dtype != dtype or tensor.device != device:
            raise PcpOwnerVmmError("external tensor has the wrong dtype/device tag", -1)
    except BaseException:
        tensor = None
        storage = None
        raise
    return _ExternalTensor(
        storage=storage,
        tensor=tensor,
        base_address=base_address,
        size_bytes=size_bytes,
    )


@dataclass
class _OwnerSlab:
    """Resources retained for one successfully mapped slab (startup lifetime)."""

    layer_names: tuple[str, ...]
    plan: SlabChunkPlan
    base_address: int
    virtual_bytes: int
    external: _ExternalTensor
    reserved: bool = True

    def tensor(self) -> torch.Tensor:
        return self.external.tensor


def owner_slab_support() -> tuple[bool, str]:
    """Cheap, device-free probe used by the fail-closed mode selector."""
    try:
        torch_npu_module = importlib.import_module("torch_npu")
        _tensor_constructors(torch_npu_module)
    except (ImportError, PcpOwnerVmmUnavailable) as error:
        return False, f"Torch-NPU external tensor binding is unavailable: {error}"
    try:
        _load_acl_library("libascendcl.so", ctypes.CDLL)
    except PcpOwnerVmmUnavailable as error:
        return False, str(error)
    return True, ""


class PcpOwnerSlabRuntime:
    """Worker-scoped owner-slab allocator over one PCP group.

    Construction and every ``allocate_slab`` call are collective over the PCP
    CPU group.  Callers must therefore invoke them from every PCP rank in the
    same order; ``initialize_kv_cache`` does exactly that.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        pcp_group: object,
        backend: AclVmmBackend | None = None,
        dist_module: object | None = None,
        placement: str = SLAB_PLACEMENT_CONTIGUOUS,
        local_only: bool = False,
    ) -> None:
        if device.type != "npu":
            raise PcpOwnerVmmUnavailable(f"owner slabs require an NPU device, got {device}")
        if placement not in SLAB_PLACEMENTS:
            raise ValueError(f"placement must be one of {SLAB_PLACEMENTS}, got {placement!r}")
        self.placement = placement
        # Control mode: same VMM allocation, no peer export/import.  Every
        # rank maps the full slab locally, so HBM use matches replicated KV.
        self.local_only = bool(local_only)
        # Diagnostic knob: shard only the first N slabs, keep the rest
        # replicated, to measure the HBM/latency tradeoff curve.
        raw_limit = os.environ.get("VLLM_ASCEND_PCP_OWNER_MAX_SLABS", "").strip()
        self.max_slabs = int(raw_limit) if raw_limit else None
        self.device = device
        self.device_index = int(device.index or 0)
        self.pcp_group = pcp_group
        self.cpu_group = getattr(pcp_group, "cpu_group", None)
        if self.cpu_group is None:
            raise PcpOwnerVmmUnavailable("PCP group does not expose a Gloo CPU group")
        self.pcp_size = int(getattr(pcp_group, "world_size", 0))
        self.rank = int(getattr(pcp_group, "rank_in_group", -1))
        if self.pcp_size <= 1 or self.rank < 0:
            raise PcpOwnerVmmUnavailable(f"invalid PCP group rank/world size {self.rank}/{self.pcp_size}")
        try:
            self._dist = dist_module or importlib.import_module("torch.distributed")
        except ImportError as error:
            raise PcpOwnerVmmUnavailable(f"torch.distributed is unavailable: {error}") from error
        self._backend = backend or AclVmmBackend()
        self._slabs: list[_OwnerSlab] = []
        self._devices: tuple[int, ...] = ()
        self._peer_devices: tuple[int, ...] | None = None
        self._peer_tgids: tuple[int, ...] | None = None
        self._prepared = False
        self._active = False
        self.map_calls = 0
        self.unmap_calls = 0
        self.map_seconds = 0.0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def slabs(self) -> tuple[_OwnerSlab, ...]:
        return tuple(self._slabs)

    def _all_gather(self, value: object) -> list[object]:
        gathered: list[object] = [None] * self.pcp_size
        self._dist.all_gather_object(gathered, value, group=self.cpu_group)
        return gathered

    def _barrier(self, stage: str) -> None:
        try:
            self._dist.barrier(group=self.cpu_group)
        except BaseException as error:
            raise PcpOwnerVmmError(f"PCP owner-slab barrier failed at {stage}", -1) from error

    def _prepare(self) -> None:
        """Exchange device ids/TGIDs and enable directed peer access once."""
        if self._prepared:
            return
        local_error: BaseException | None = None
        tgid = 0
        try:
            tgid = self._backend.bare_tgid(self.device_index)
        except BaseException as error:  # noqa: BLE001 - reported collectively
            local_error = error
        endpoints = self._all_gather(
            {
                "ok": local_error is None,
                "error": "" if local_error is None else f"{type(local_error).__name__}: {local_error}",
                "rank": self.rank,
                "device": self.device_index,
                "tgid": tgid,
            }
        )
        failures = [entry.get("error") for entry in endpoints if not entry.get("ok")]
        local_error: BaseException | None = None
        if failures:
            local_error = PcpOwnerVmmUnavailable("PCP endpoint exchange failed: " + "; ".join(map(str, failures)))
        devices: tuple[int, ...] = ()
        tgids: tuple[int, ...] = ()
        if local_error is None:
            try:
                devices = tuple(int(entry["device"]) for entry in endpoints)
                if len(set(devices)) != self.pcp_size or any(device < 0 for device in devices):
                    raise PcpOwnerVmmUnavailable(f"PCP ranks must use distinct devices, got {devices}")
                if sorted(devices) != list(range(self.pcp_size)):
                    # Valid for a process whose visible set is not 0..P-1, but
                    # peer access uses in-process logical ordinals, so surface
                    # it instead of silently failing later.
                    logger.warning(
                        "PCP device ordinals are %s, not 0..%d; peer access will fail unless "
                        "ASCEND_RT_VISIBLE_DEVICES makes them local ordinals",
                        devices,
                        self.pcp_size - 1,
                    )
                tgids = tuple(int(entry["tgid"]) for entry in endpoints)
                if len(set(tgids)) != self.pcp_size:
                    raise PcpOwnerVmmUnavailable(f"PCP ranks must be distinct processes, got TGIDs {tgids}")
                self._backend.set_device(self.device_index)
                granularity = self._backend.allocate_granularity(self.device_index)
                if granularity != CANN_VMM_GRANULARITY_BYTES:
                    raise PcpOwnerVmmUnavailable(
                        f"CANN allocation granularity is {granularity} bytes, expected {CANN_VMM_GRANULARITY_BYTES}"
                    )
                self._devices = devices
                self._peer_devices = tuple(device for device in devices if device != self.device_index)
                self._peer_tgids = tuple(tgid for index, tgid in enumerate(tgids) if index != self.rank)
                self._backend.enable_peer_access(self.device_index, self._peer_devices)
            except BaseException as error:  # noqa: BLE001 - reported collectively
                local_error = error
        # A rank-local granularity/peer-access failure must not let peers move
        # on to the slab handshake alone.
        statuses = self._all_gather(
            {
                "ok": local_error is None,
                "error": "" if local_error is None else f"{type(local_error).__name__}: {local_error}",
                "rank": self.rank,
            }
        )
        failures = [f"rank {entry['rank']}: {entry['error']}" for entry in statuses if not entry["ok"]]
        if failures:
            raise PcpOwnerVmmUnavailable("PCP owner-slab preparation failed: " + "; ".join(failures))
        self._prepared = True
        logger.info(
            "PCP owner-slab VMM peer access ready: rank=%d/%d device=%d peers=%s tgids=%s",
            self.rank,
            self.pcp_size,
            self.device_index,
            self._peer_devices,
            tgids,
        )

    def _map_slab(
        self,
        *,
        plan: SlabChunkPlan,
        base_address: int,
        physical: AclPhysicalHandle | None,
        imported: dict[int, AclPhysicalHandle],
    ) -> list[int]:
        """Map each owner's chunks with one call when they are contiguous.

        A contiguous slab needs ``pcp_size`` ``aclrtMapMem`` calls instead of
        one per two-megabyte chunk, which keeps a 16 GiB startup well inside a
        second instead of thousands of driver calls.
        """
        started = time.perf_counter()
        calls = 0
        mapped: list[int] = []
        for owner in range(self.pcp_size):
            owner_plan = SlabChunkPlan(
                size_bytes=plan.size_bytes,
                pcp_size=self.pcp_size,
                rank=owner,
                placement=plan.placement,
            )
            chunks = owner_plan.own_chunks
            if not chunks:
                continue
            handle = physical if owner == self.rank else imported.get(owner)
            if handle is None:
                raise PcpOwnerVmmError(f"owner {owner} has no physical handle", -1)
            if plan.placement == SLAB_PLACEMENT_CONTIGUOUS:
                address = base_address + chunks[0] * CANN_VMM_GRANULARITY_BYTES
                size_bytes = len(chunks) * CANN_VMM_GRANULARITY_BYTES
                self._backend.map(
                    base_address=address,
                    size_bytes=size_bytes,
                    offset_bytes=0,
                    handle=handle,
                    device_index=self.device_index,
                )
                mapped.append(address)
                calls += 1
                continue
            for chunk in chunks:
                address = base_address + chunk * CANN_VMM_GRANULARITY_BYTES
                self._backend.map(
                    base_address=address,
                    size_bytes=CANN_VMM_GRANULARITY_BYTES,
                    offset_bytes=owner_plan.offset_in_owner(chunk),
                    handle=handle,
                    device_index=self.device_index,
                )
                mapped.append(address)
                calls += 1
        self.map_calls += calls
        self.map_seconds += time.perf_counter() - started
        return mapped

    def _release_local(
        self,
        *,
        base_address: int | None,
        physical: AclPhysicalHandle | None,
        mapped: list[int],
        imported: dict[int, AclPhysicalHandle],
    ) -> None:
        """Best-effort release of a slab that never became visible to peers."""
        errors: list[str] = []
        for address in reversed(mapped):
            try:
                self._backend.unmap(base_address=address, device_index=self.device_index)
            except BaseException as error:  # noqa: BLE001 - release all resources
                errors.append(f"unmap {address:#x}: {error}")
        for handle in imported.values():
            try:
                self._backend.free_physical(handle=handle, device_index=self.device_index)
            except BaseException as error:  # noqa: BLE001
                errors.append(f"free imported {handle.value:#x}: {error}")
        if physical is not None:
            try:
                self._backend.free_physical(handle=physical, device_index=self.device_index)
            except BaseException as error:  # noqa: BLE001
                errors.append(f"free owner {physical.value:#x}: {error}")
        if base_address is not None:
            try:
                self._backend.release_address(base_address=base_address, device_index=self.device_index)
            except BaseException as error:  # noqa: BLE001
                errors.append(f"release {base_address:#x}: {error}")
        if errors:
            logger.error("PCP owner-slab rollback was incomplete: %s", "; ".join(errors))

    def _release_imports(
        self,
        *,
        mapped: list[int],
        imported: dict[int, AclPhysicalHandle],
    ) -> None:
        """Phase one of a collective rollback: drop this rank's peer access."""
        errors: list[str] = []
        for address in reversed(mapped):
            try:
                self._backend.unmap(base_address=address, device_index=self.device_index)
            except BaseException as error:  # noqa: BLE001
                errors.append(f"unmap {address:#x}: {error}")
        for handle in imported.values():
            try:
                self._backend.free_physical(handle=handle, device_index=self.device_index)
            except BaseException as error:  # noqa: BLE001
                errors.append(f"free imported {handle.value:#x}: {error}")
        if errors:
            logger.error("PCP owner-slab import release was incomplete: %s", "; ".join(errors))

    @staticmethod
    def _smoke_magic(rank: int, chunk: int) -> int:
        return (0x5A5A_0000_0000_0000 | (rank << 32) | chunk) & 0x7FFF_FFFF_FFFF_FFFF

    @staticmethod
    def _smoke_sample(plan: SlabChunkPlan) -> list[int]:
        own = plan.own_chunks
        if not own:
            return []
        return sorted({own[0], own[len(own) // 2], own[-1]})

    @staticmethod
    def _smoke_word_index(plan: SlabChunkPlan, chunk: int, slot: int = 0) -> int | None:
        """Word index of a sampled chunk word, or None if it is out of range."""
        offset = chunk * CANN_VMM_GRANULARITY_BYTES + slot * 8
        if plan.size_bytes % 8 or offset + 8 > plan.size_bytes:
            return None
        return offset // 8

    def _smoke_write(self, tensor: torch.Tensor, plan: SlabChunkPlan) -> None:
        """Write own-chunk and peer-chunk magic words.

        Every rank writes its own sample at word 0 of a chunk it owns and, to
        prove remote writes, writes the next rank's chunks at word 1.  The
        following zero fill wipes both words.
        """
        if plan.size_bytes % 8:
            return
        words = tensor.view(torch.int64)
        for chunk in self._smoke_sample(plan):
            index = self._smoke_word_index(plan, chunk, 0)
            if index is not None:
                words[index] = self._smoke_magic(self.rank, chunk)
        target = (self.rank + 1) % self.pcp_size
        target_plan = SlabChunkPlan(
            size_bytes=plan.size_bytes,
            pcp_size=self.pcp_size,
            rank=target,
            placement=plan.placement,
        )
        for chunk in self._smoke_sample(target_plan):
            index = self._smoke_word_index(plan, chunk, 1)
            if index is not None:
                words[index] = self._smoke_magic(self.rank, chunk)
        torch.npu.synchronize()

    def _smoke_verify(self, tensor: torch.Tensor, plan: SlabChunkPlan, layer_names: Sequence[str]) -> None:
        """Read local and peer magic words back through this rank's alias."""
        if plan.size_bytes % 8:
            return
        words = tensor.view(torch.int64)
        failures: list[str] = []
        for owner_rank in range(self.pcp_size):
            owner_plan = SlabChunkPlan(
                size_bytes=plan.size_bytes,
                pcp_size=self.pcp_size,
                rank=owner_rank,
                placement=plan.placement,
            )
            # Rank r writes into owner (r + 1) % P, so owner o was written
            # by (o - 1) % P.
            writer_rank = (owner_rank - 1) % self.pcp_size
            for chunk in self._smoke_sample(owner_plan):
                own_index = self._smoke_word_index(plan, chunk, 0)
                if own_index is not None:
                    observed = int(words[own_index].item())
                    expected = self._smoke_magic(owner_rank, chunk)
                    if observed != expected:
                        failures.append(f"owner {owner_rank} chunk {chunk}: {observed:#x} != {expected:#x}")
                peer_index = self._smoke_word_index(plan, chunk, 1)
                if peer_index is not None:
                    observed = int(words[peer_index].item())
                    expected = self._smoke_magic(writer_rank, chunk)
                    if observed != expected:
                        failures.append(
                            f"peer write {writer_rank}->{owner_rank} chunk {chunk}: {observed:#x} != {expected:#x}"
                        )
        if failures:
            raise PcpOwnerVmmError(
                f"owner-slab peer mapping verification failed for {list(layer_names)[:1]}: " + "; ".join(failures[:8]),
                -1,
            )

    def _zero_owned(self, tensor: torch.Tensor, plan: SlabChunkPlan) -> None:
        if plan.size_bytes % 8:
            # Match ``torch.zeros`` for the replicated path bit-for-bit.
            tensor.zero_()
            return
        words = tensor.view(torch.int64)
        for chunk in plan.own_chunks:
            start = chunk * CANN_VMM_GRANULARITY_BYTES // 8
            end = min(start + CANN_VMM_GRANULARITY_BYTES // 8, words.numel())
            words[start:end].zero_()

    def _allocate_local_slab(self, names: tuple[str, ...], size_bytes: int) -> torch.Tensor | None:
        """Control path: reserve and map the full slab on this rank only.

        Used to separate "VMM allocation" from "peer mapping" when explaining
        a performance delta.  It keeps the same virtual reservation and the
        same external-tensor binding as owner mode, but allocates all physical
        memory locally and never exports or imports a handle.
        """
        plan = SlabChunkPlan(size_bytes=size_bytes, pcp_size=1, rank=0, placement=self.placement)
        base_address: int | None = None
        physical: AclPhysicalHandle | None = None
        started = time.perf_counter()
        try:
            self._backend.set_device(self.device_index)
            base_address = self._backend.reserve_address(size_bytes=plan.virtual_bytes, device_index=self.device_index)
            physical = self._backend.malloc_physical(size_bytes=plan.virtual_bytes, device_index=self.device_index)
            self._backend.map(
                base_address=base_address,
                size_bytes=plan.virtual_bytes,
                offset_bytes=0,
                handle=physical,
                device_index=self.device_index,
            )
            external = bind_external_tensor(
                base_address=base_address,
                size_bytes=size_bytes,
                device_index=self.device_index,
            )
            external.tensor.zero_()
        except BaseException as error:  # noqa: BLE001 - fail closed to replicated
            logger.error(
                "PCP local VMM slab failed for %s (%s: %s); falling back to replicated KV",
                names[:1],
                type(error).__name__,
                error,
            )
            return None
        self.map_calls += 1
        self.map_seconds += time.perf_counter() - started
        self._slabs.append(
            _OwnerSlab(
                layer_names=names,
                plan=plan,
                base_address=base_address,
                virtual_bytes=plan.virtual_bytes,
                external=external,
            )
        )
        _register_slab(base_address, plan.virtual_bytes, self)
        self._active = True
        logger.info(
            "PCP local VMM slab mapped (control): layers=%s size=%.3f GiB, full physical per rank",
            list(names)[:2],
            plan.virtual_bytes / (1024**3),
        )
        return external.tensor

    def allocate_slab(self, layer_names: Sequence[str], size_bytes: int) -> torch.Tensor | None:
        """Allocate one owner slab or return ``None`` if the group falls back.

        The decision is collective: one failing rank makes every rank return
        None so the caller can allocate a replicated tensor of the same size.
        """
        names = tuple(str(name) for name in layer_names)
        if size_bytes <= 0:
            raise ValueError(f"slab size must be positive, got {size_bytes}")
        if self.local_only:
            return self._allocate_local_slab(names, size_bytes)
        if self.max_slabs is not None and len(self._slabs) >= self.max_slabs:
            logger.info_once(
                "PCP owner-slab limit VLLM_ASCEND_PCP_OWNER_MAX_SLABS=%d reached; %s stays replicated.",
                self.max_slabs,
                names[:1],
            )
            return None
        try:
            self._prepare()
        except BaseException as error:  # noqa: BLE001 - fail closed to replicated
            self._active = False
            logger.error(
                "PCP owner-slab preparation failed (%s: %s); falling back to replicated KV for %s",
                type(error).__name__,
                error,
                names[:1],
            )
            return None

        plan = SlabChunkPlan(
            size_bytes=size_bytes,
            pcp_size=self.pcp_size,
            rank=self.rank,
            placement=self.placement,
        )
        base_address: int | None = None
        physical: AclPhysicalHandle | None = None
        shareable: AclShareableHandle | None = None
        local_error: BaseException | None = None
        try:
            base_address = self._backend.reserve_address(size_bytes=plan.virtual_bytes, device_index=self.device_index)
            if plan.own_bytes:
                physical = self._backend.malloc_physical(size_bytes=plan.own_bytes, device_index=self.device_index)
                shareable = self._backend.export(handle=physical, device_index=self.device_index)
                self._backend.authorize(
                    shareable_handle=shareable,
                    bare_tgids=self._peer_tgids or (),
                    device_index=self.device_index,
                )
        except BaseException as error:  # noqa: BLE001 - reported collectively
            local_error = error

        endpoints = self._all_gather(
            {
                "ok": local_error is None,
                "error": "" if local_error is None else f"{type(local_error).__name__}: {local_error}",
                "rank": self.rank,
                "size": size_bytes,
                "own_chunks": len(plan.own_chunks),
                "handle": None if shareable is None else shareable.payload,
            }
        )
        failures = [f"rank {entry['rank']}: {entry['error']}" for entry in endpoints if not entry["ok"]]
        if len({int(entry["size"]) for entry in endpoints}) != 1:
            failures.append("PCP ranks disagree on the owner-slab size")
        if failures:
            self._release_local(base_address=base_address, physical=physical, mapped=[], imported={})
            self._active = False
            logger.error(
                "PCP owner-slab export failed for %s (%s); falling back to replicated KV",
                names[:1],
                "; ".join(failures),
            )
            return None

        imported: dict[int, AclPhysicalHandle] = {}
        mapped: list[int] = []
        external: _ExternalTensor | None = None
        map_error: BaseException | None = None
        try:
            for entry in endpoints:
                owner = int(entry["rank"])
                if owner == self.rank or entry["handle"] is None:
                    continue
                imported[owner] = self._backend.import_handle(
                    shareable_handle=AclShareableHandle(entry["handle"]),
                    device_index=self.device_index,
                )
            mapped = self._map_slab(
                plan=plan,
                base_address=base_address,
                physical=physical,
                imported=imported,
            )
            external = bind_external_tensor(
                base_address=base_address,
                size_bytes=size_bytes,
                device_index=self.device_index,
            )
        except BaseException as error:  # noqa: BLE001 - reported collectively
            map_error = error

        # Both barriers are entered by every rank even when its own mapping
        # failed, so the rollback path cannot deadlock.
        self._barrier("map-complete")
        if map_error is None:
            try:
                self._smoke_write(external.tensor, plan)
            except BaseException as error:  # noqa: BLE001
                map_error = error
        self._barrier("smoke-write")
        if map_error is None:
            try:
                self._smoke_verify(external.tensor, plan, names)
            except BaseException as error:  # noqa: BLE001
                map_error = error
        # Nobody may zero its chunks until every peer finished reading the
        # magic words, otherwise the zero fill races the verification.
        self._barrier("smoke-verify")
        if map_error is None:
            try:
                self._zero_owned(external.tensor, plan)
            except BaseException as error:  # noqa: BLE001
                map_error = error

        statuses = self._all_gather(
            {
                "ok": map_error is None,
                "error": "" if map_error is None else f"{type(map_error).__name__}: {map_error}",
                "rank": self.rank,
            }
        )
        failures = [f"rank {entry['rank']}: {entry['error']}" for entry in statuses if not entry["ok"]]
        if failures:
            # Two-phase rollback: every rank drops peer mappings first, then a
            # barrier guarantees nobody is still mapping a block before each
            # owner frees its physical memory.
            self._release_imports(mapped=mapped, imported=imported)
            try:
                self._barrier("rollback-imports")
            except BaseException as error:  # noqa: BLE001 - original error is more useful
                logger.error("PCP owner-slab rollback barrier failed: %s", error)
            if physical is not None:
                try:
                    self._backend.free_physical(handle=physical, device_index=self.device_index)
                except BaseException as error:  # noqa: BLE001
                    logger.error("PCP owner-slab owner release failed: %s", error)
            if base_address is not None:
                try:
                    self._backend.release_address(base_address=base_address, device_index=self.device_index)
                except BaseException as error:  # noqa: BLE001
                    logger.error("PCP owner-slab address release failed: %s", error)
            self._active = False
            logger.error(
                "PCP owner-slab map failed for %s (%s); falling back to replicated KV",
                names[:1],
                "; ".join(failures),
            )
            return None

        slab = _OwnerSlab(
            layer_names=names,
            plan=plan,
            base_address=base_address,
            virtual_bytes=plan.virtual_bytes,
            external=external,
        )
        self._slabs.append(slab)
        _register_slab(base_address, plan.virtual_bytes, self)
        self._active = True
        logger.info(
            "PCP owner slab mapped: layers=%s size=%.3f GiB virtual, %.3f GiB per rank "
            "(chunks=%d own=%d pcp=%d placement=%s map_calls=%d map_s=%.3f)",
            list(names),
            plan.virtual_bytes / (1024**3),
            plan.own_bytes / (1024**3),
            plan.chunk_count,
            len(plan.own_chunks),
            self.pcp_size,
            plan.placement,
            self.map_calls,
            self.map_seconds,
        )
        return external.tensor

    def close(self) -> None:
        """Drop local references without freeing peer-visible physical memory.

        Owner blocks are exported to peers, so releasing them safely needs a
        collective teardown over the PCP group.  Until that lifecycle hook
        exists, the process exit path is the release point; dropping the
        Python aliases here only removes this worker's references.
        """
        count = len(self._slabs)
        self._slabs.clear()
        self._active = False
        if count:
            logger.info("PCP owner-slab runtime closed: slabs=%d (physical released at process exit)", count)


_runtime: PcpOwnerSlabRuntime | None = None
_runtime_lock = threading.Lock()

# (base_address, size_bytes, runtime) for every mapped slab, so consumers can
# tell whether a cache tensor is backed by owner (possibly peer) memory.
_slab_registry: list[tuple[int, int, PcpOwnerSlabRuntime]] = []


def _register_slab(base_address: int, size_bytes: int, runtime: PcpOwnerSlabRuntime) -> None:
    _slab_registry.append((int(base_address), int(size_bytes), runtime))


def owner_slab_for_tensor(tensor: torch.Tensor | None) -> PcpOwnerSlabRuntime | None:
    """Return the owner runtime backing ``tensor``, or ``None`` for local tensors."""
    if tensor is None or tensor.device.type != "npu":
        return None
    pointer = int(tensor.data_ptr())
    for base, size, runtime in _slab_registry:
        if base <= pointer < base + size:
            return runtime
    return None


def get_owner_slab_allocator(vllm_config, device: torch.device) -> PcpOwnerSlabRuntime | None:
    """Return this worker's owner-slab runtime when owner mode is usable."""
    from .pcp_owner_kv import PCP_KV_SHARING_OWNER, pcp_kv_sharing_mode_from_config, validate_owner_sharing

    if pcp_kv_sharing_mode_from_config(vllm_config) != PCP_KV_SHARING_OWNER:
        return None
    supported, reason = validate_owner_sharing(vllm_config)
    if not supported:
        logger.warning_once(
            "PCP owner-sharded KV requested but unsupported (%s); using the replicated cache path.", reason
        )
        return None
    ok, reason = owner_slab_support()
    if not ok:
        logger.warning_once(
            "PCP owner-sharded KV requested but the VMM backend is unavailable (%s); using the replicated cache path.",
            reason,
        )
        return None

    from vllm.distributed import get_pcp_group

    additional = getattr(vllm_config, "additional_config", None) or {}
    placement = str(additional.get("pcp_owner_placement", SLAB_PLACEMENT_CONTIGUOUS)).strip().lower()
    if placement not in SLAB_PLACEMENTS:
        logger.warning_once(
            "Unknown pcp_owner_placement %r; using %s.",
            placement,
            SLAB_PLACEMENT_CONTIGUOUS,
        )
        placement = SLAB_PLACEMENT_CONTIGUOUS

    import os

    local_only = os.environ.get("VLLM_ASCEND_PCP_OWNER_LOCAL_VMM", "").strip().lower() in ("1", "true", "yes", "on")
    if local_only:
        logger.warning(
            "PCP owner KV control mode: full local VMM slab per rank "
            "(VLLM_ASCEND_PCP_OWNER_LOCAL_VMM=1); no HBM saving, no peer mapping."
        )

    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        try:
            _runtime = PcpOwnerSlabRuntime(
                device=device, pcp_group=get_pcp_group(), placement=placement, local_only=local_only
            )
        except BaseException as error:  # noqa: BLE001 - fail closed to replicated
            logger.warning_once(
                "PCP owner-sharded KV could not be prepared (%s: %s); using the replicated cache path.",
                type(error).__name__,
                error,
            )
            _runtime = None
        return _runtime


def owner_slab_runtime_active() -> bool:
    """Whether this process finished mapping at least one owner slab."""
    runtime = _runtime
    return runtime is not None and runtime.active


def reset_owner_slab_allocator_for_tests() -> None:
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            _runtime.close()
        _runtime = None
