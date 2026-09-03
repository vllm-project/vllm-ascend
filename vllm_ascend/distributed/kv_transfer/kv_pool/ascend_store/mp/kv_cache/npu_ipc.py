"""NPU IPC mappings and ordering events shared with KVCacheServer.

KV cache allocations and stream events require torch-npu IPC handles rather
than ordinary serialization. Both paths identify the physical NPU by UUID so
the server can recover the correct local device even when logical indices
differ between processes.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import cloudpickle
import torch

TORCH_NPU_IPC_HANDLE = "torch_npu_ipc"
TORCH_NPU_IPC_VERSION = 1


# ==============================
# Worker KV cache mappings
# ==============================

# Registration exports each allocation once and describes every tensor view
# separately. Source allocations and imported mappings remain referenced for
# the lifetime of the corresponding client registration and Worker service.


@dataclass(frozen=True)
class KVCacheTensorSpec:
    """Process-neutral layout of one tensor in a worker KV cache.

    ``storage_index`` points to the corresponding opaque handle in
    ``WorkerKVCacheSpec.storages``; no process-local address crosses the wire.
    """

    storage_index: int
    storage_offset_bytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class KVCacheStorageSpec:
    """Opaque IPC handle and source-device identity for one allocation."""

    size_bytes: int
    device_type: str
    device_uuid: str
    handle_type: str
    handle_version: int
    handle: bytes


@dataclass(frozen=True)
class WorkerKVCacheSpec:
    """Storage handles and tensor layouts registered by one Worker."""

    caches: dict[str, tuple[KVCacheTensorSpec, ...]]
    storages: tuple[KVCacheStorageSpec, ...]


class KVCacheStorageAdapter(Protocol):
    """Device-specific storage sharing hidden from KV cache registration."""

    def export_storage(self, storage: torch.Tensor) -> KVCacheStorageSpec: ...

    def import_storage(self, spec: KVCacheStorageSpec) -> tuple[torch.Tensor, int | None]: ...


@dataclass
class ExportedKVCache:
    """Serializable specification plus references keeping exported allocations alive."""

    spec: WorkerKVCacheSpec
    _storages: tuple[torch.Tensor, ...]

    def close(self) -> None:
        self._storages = ()


@dataclass
class ImportedKVCache:
    """Reconstructed cache tensors plus references keeping IPC mappings alive."""

    tensors: dict[str, tuple[torch.Tensor, ...]]
    device_index: int | None
    _storages: tuple[torch.Tensor, ...]

    def close(self) -> None:
        self.tensors.clear()
        self._storages = ()


class TorchNPUIPCAdapter:
    """Share NPU allocations through torch-npu multiprocessing handles."""

    def export_storage(self, storage: torch.Tensor) -> KVCacheStorageSpec:
        if storage.device.type != "npu":
            raise ValueError(f"TorchNPUIPCAdapter only supports NPU storage, got {storage.device}")

        from torch.multiprocessing.reductions import reduce_tensor

        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

        _, ipc_args = reduce_tensor(storage)
        return KVCacheStorageSpec(
            size_bytes=_storage_size_bytes(storage),
            device_type=storage.device.type,
            device_uuid=npu_generate_uuid(storage.device.index),
            handle_type=TORCH_NPU_IPC_HANDLE,
            handle_version=TORCH_NPU_IPC_VERSION,
            handle=cloudpickle.dumps(tuple(ipc_args)),
        )

    def import_storage(self, spec: KVCacheStorageSpec) -> tuple[torch.Tensor, int]:
        if spec.handle_type != TORCH_NPU_IPC_HANDLE or spec.handle_version != TORCH_NPU_IPC_VERSION:
            raise ValueError(f"Unsupported KV cache handle {spec.handle_type!r} version {spec.handle_version}")

        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        device_index = _resolve_device(spec.device_uuid, "KV cache")
        torch.npu.set_device(device_index)
        ipc_args = list(cloudpickle.loads(spec.handle))
        if len(ipc_args) <= 6:
            raise ValueError("Malformed torch-npu IPC handle")

        # Logical device indices may differ between the Worker and server.
        ipc_args[6] = device_index
        return rebuild_npu_tensor(*ipc_args), device_index


def export_worker_kv_caches(
    kv_caches: dict[str, torch.Tensor],
    adapter: KVCacheStorageAdapter | None = None,
) -> ExportedKVCache:
    """Export each allocation once and describe every tensor view over it."""
    if not kv_caches:
        raise ValueError("kv_caches must not be empty")

    adapter = TorchNPUIPCAdapter() if adapter is None else adapter
    storage_indices: dict[tuple[str, int | None, int], int] = {}
    storages: list[torch.Tensor] = []
    storage_specs: list[KVCacheStorageSpec] = []
    caches: dict[str, tuple[KVCacheTensorSpec, ...]] = {}

    for name, cache_or_caches in kv_caches.items():
        tensors = _normalize_cache_tensors(name, cache_or_caches)
        tensor_specs: list[KVCacheTensorSpec] = []
        for tensor in tensors:
            storage = _untyped_storage(tensor)
            storage_key = (tensor.device.type, tensor.device.index, storage.data_ptr())
            storage_index = storage_indices.get(storage_key)
            if storage_index is None:
                storage_index = len(storages)
                storage_indices[storage_key] = storage_index
                storage_tensor = _storage_as_bytes(tensor)
                storage_spec = adapter.export_storage(storage_tensor)
                if storage_spec.size_bytes != _storage_size_bytes(storage_tensor):
                    raise ValueError("KV cache adapter returned an incorrect storage size")
                if storage_spec.device_type != tensor.device.type:
                    raise ValueError("KV cache adapter returned an incorrect device type")
                storages.append(storage_tensor)
                storage_specs.append(storage_spec)

            tensor_specs.append(
                KVCacheTensorSpec(
                    storage_index=storage_index,
                    storage_offset_bytes=tensor.storage_offset() * tensor.element_size(),
                    shape=tuple(tensor.shape),
                    stride=tuple(tensor.stride()),
                    dtype=str(tensor.dtype),
                )
            )
        caches[name] = tuple(tensor_specs)

    spec = WorkerKVCacheSpec(caches=caches, storages=tuple(storage_specs))
    _validate_worker_spec(spec)
    return ExportedKVCache(spec, tuple(storages))


def import_worker_kv_caches(spec: WorkerKVCacheSpec, adapter: KVCacheStorageAdapter | None = None) -> ImportedKVCache:
    """Import each allocation once and rebuild the registered tensor views."""
    _validate_worker_spec(spec)
    adapter = TorchNPUIPCAdapter() if adapter is None else adapter
    imported_storages: list[tuple[torch.Tensor, int | None]] = []
    try:
        for storage_spec in spec.storages:
            storage, device_index = adapter.import_storage(storage_spec)
            if _storage_size_bytes(storage) < storage_spec.size_bytes:
                raise ValueError("Imported KV cache storage is smaller than its specification")
            if storage.device.type != storage_spec.device_type:
                raise ValueError(
                    f"Imported KV cache storage is on {storage.device.type}, expected {storage_spec.device_type}"
                )
            imported_storages.append((storage, device_index))

        device_indices = {device_index for _, device_index in imported_storages}
        if len(device_indices) > 1:
            raise ValueError("One Worker registration cannot span multiple server devices")
        device_index = next(iter(device_indices), None)
        caches = _rebuild_cache_tensors(spec, imported_storages)
        return ImportedKVCache(caches, device_index, tuple(storage for storage, _ in imported_storages))
    except Exception:
        imported_storages.clear()
        raise


def _rebuild_cache_tensors(
    spec: WorkerKVCacheSpec,
    storages: list[tuple[torch.Tensor, int | None]],
) -> dict[str, tuple[torch.Tensor, ...]]:
    caches: dict[str, tuple[torch.Tensor, ...]] = {}
    for name, tensor_specs in spec.caches.items():
        tensors = []
        for tensor_spec in tensor_specs:
            storage = storages[tensor_spec.storage_index][0]
            dtype = _decode_dtype(tensor_spec.dtype)
            element_size = torch.empty((), dtype=dtype).element_size()
            tensor = torch.empty(0, dtype=dtype, device=storage.device)
            tensor.set_(
                storage.untyped_storage(),
                tensor_spec.storage_offset_bytes // element_size,
                tensor_spec.shape,
                tensor_spec.stride,
            )
            tensors.append(tensor)
        caches[name] = tuple(tensors)
    return caches


def _validate_worker_spec(spec: WorkerKVCacheSpec) -> None:
    if not spec.storages:
        raise ValueError("KV cache storage handles are required")
    if not spec.caches:
        raise ValueError("KV cache tensor layouts are required")
    for storage in spec.storages:
        if storage.size_bytes <= 0:
            raise ValueError("KV cache storage size must be greater than 0")
        if not storage.device_type or not storage.device_uuid:
            raise ValueError("KV cache storage device identity is required")
        if not storage.handle_type or storage.handle_version <= 0 or not storage.handle:
            raise ValueError("KV cache storage handle is invalid")
    for name, tensors in spec.caches.items():
        if not isinstance(name, str) or not name or not tensors:
            raise ValueError("KV cache tensor layouts must have non-empty names and values")
        for tensor in tensors:
            if not 0 <= tensor.storage_index < len(spec.storages):
                raise ValueError(f"KV cache {name!r} references an unknown storage")
            dtype = _decode_dtype(tensor.dtype)
            element_size = torch.empty((), dtype=dtype).element_size()
            _validate_tensor_spec(name, tensor, spec.storages[tensor.storage_index], element_size)


def _validate_tensor_spec(name: str, tensor: KVCacheTensorSpec, storage: KVCacheStorageSpec, element_size: int) -> None:
    if tensor.storage_offset_bytes < 0 or tensor.storage_offset_bytes % element_size:
        raise ValueError(f"KV cache {name!r} has an invalid storage offset")
    if len(tensor.shape) != len(tensor.stride) or any(size < 0 for size in tensor.shape):
        raise ValueError(f"KV cache {name!r} has an invalid shape or stride")
    if any(stride < 0 for stride in tensor.stride):
        raise ValueError(f"KV cache {name!r} has a negative stride")

    required_bytes = tensor.storage_offset_bytes
    if all(tensor.shape):
        last_element = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride))
        required_bytes += (last_element + 1) * element_size
    if required_bytes > storage.size_bytes:
        raise ValueError(f"KV cache {name!r} exceeds its storage allocation")


def _normalize_cache_tensors(name: str, cache_or_caches) -> tuple[torch.Tensor, ...]:
    if not isinstance(name, str) or not name:
        raise ValueError("KV cache names must be non-empty strings")
    tensors = (cache_or_caches,) if isinstance(cache_or_caches, torch.Tensor) else tuple(cache_or_caches)
    if not tensors or any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError(f"KV cache {name!r} must contain one or more tensors")
    return tensors


def _storage_as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    storage = _untyped_storage(tensor)
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(storage, 0, (storage.nbytes(),), (1,))


def _untyped_storage(tensor: torch.Tensor):
    try:
        return tensor.untyped_storage()
    except AttributeError:
        return tensor.storage()


def _storage_size_bytes(tensor: torch.Tensor) -> int:
    return _untyped_storage(tensor).nbytes()


def _decode_dtype(value: str) -> torch.dtype:
    if not value.startswith("torch."):
        raise ValueError(f"Invalid KV cache dtype {value!r}")
    dtype = getattr(torch, value.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported KV cache dtype {value!r}")
    return dtype


# ==============================
# NPU event ordering across processes
# ==============================

# Events are recorded on the source Worker stream and only their IPC handles
# cross the process boundary. Keeping the source event alive and importing it
# on the matching NPU preserves that stream order without recording a substitute
# event inside KVCacheServer.


@dataclass(frozen=True)
class NPUEventSpec:
    """Process-neutral identity and IPC handle for one NPU event."""

    device_uuid: str
    handle: bytes


@dataclass
class ExportedNPUEvent:
    """Keep the source event alive while another process imports it."""

    spec: NPUEventSpec
    _event: Any | None

    def close(self) -> None:
        self._event = None


def record_npu_event(stream: Any | None = None) -> ExportedNPUEvent:
    """Record and export an event on the current logical NPU device."""
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

    device_index = torch.npu.current_device()
    event = torch.npu.Event(interprocess=True)
    if stream is None:
        event.record()
    else:
        event.record(stream)
    handle = event.ipc_handle()
    if not isinstance(handle, bytes) or not handle:
        raise RuntimeError("torch-npu returned an invalid NPU event IPC handle")
    return ExportedNPUEvent(NPUEventSpec(npu_generate_uuid(device_index), handle), event)


def import_npu_event(spec: NPUEventSpec) -> Any:
    """Rebuild an event on the local logical device matching its UUID."""
    if not isinstance(spec, NPUEventSpec):
        raise TypeError(f"spec must be NPUEventSpec, got {type(spec).__name__}")
    if not spec.device_uuid or not spec.handle:
        raise ValueError("NPU event specification is incomplete")

    device_index = _resolve_device(spec.device_uuid, "event")
    torch.npu.set_device(device_index)
    return torch.npu.Event.from_ipc_handle(device_index, spec.handle)


def _resolve_device(device_uuid: str, resource: str) -> int:
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

    for device_index in range(torch.npu.device_count()):
        if npu_generate_uuid(device_index) == device_uuid:
            return device_index
    raise ValueError(f"No local NPU matches {resource} device UUID {device_uuid!r}")


__all__ = [
    "ExportedKVCache",
    "ExportedNPUEvent",
    "ImportedKVCache",
    "KVCacheStorageAdapter",
    "KVCacheStorageSpec",
    "KVCacheTensorSpec",
    "NPUEventSpec",
    "TorchNPUIPCAdapter",
    "WorkerKVCacheSpec",
    "export_worker_kv_caches",
    "import_npu_event",
    "import_worker_kv_caches",
    "record_npu_event",
]
