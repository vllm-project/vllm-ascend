"""NPU IPC mappings for KV cache allocations imported by the subprocess.

The transfer subprocess inherits its Worker's device visibility. Import rejects
handles whose encoded logical device differs from the configured child device.
"""

from dataclasses import dataclass
from typing import Protocol

import cloudpickle  # type: ignore
import torch

_STORAGE_DEVICE_INDEX_ARG = 6
_STORAGE_SIZE_BYTES_ARG = 8
_TORCH_NPU_IPC_ARG_COUNT = 15


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
    """Opaque IPC handle for one allocation."""

    handle: bytes


@dataclass(frozen=True)
class WorkerKVCacheSpec:
    """Storage handles and tensor layouts registered by one Worker."""

    caches: dict[str, tuple[KVCacheTensorSpec, ...]]
    storages: tuple[KVCacheStorageSpec, ...]


class KVCacheStorageAdapter(Protocol):
    """Device-specific storage sharing hidden from KV cache registration."""

    def export_storage(self, storage: torch.Tensor) -> KVCacheStorageSpec: ...

    def import_storage(
        self, spec: KVCacheStorageSpec, device_index: int | None
    ) -> tuple[torch.Tensor, int | None, int]: ...


@dataclass
class ExportedKVCache:
    """Serializable specification plus references keeping exported allocations alive."""

    spec: WorkerKVCacheSpec
    _storages: tuple[torch.Tensor, ...]
    _storage_sizes: tuple[int, ...]

    def describe_range(self, address: int, size: int) -> tuple[int, int, int]:
        if size < 0:
            raise ValueError("A transfer range cannot have a negative size")
        for index, storage in enumerate(self._storages):
            offset = address - storage.data_ptr()
            allocation_size = self._storage_sizes[index]
            if 0 <= offset < allocation_size and offset + size <= allocation_size:
                return index, offset, size
        raise ValueError("Transfer range is outside the exported KV allocations")

    def close(self) -> None:
        self._storages = ()
        self._storage_sizes = ()


@dataclass
class ImportedKVCache:
    """Reconstructed cache tensors plus references keeping IPC mappings alive."""

    tensors: dict[str, tuple[torch.Tensor, ...]]
    device_index: int | None
    _storages: tuple[torch.Tensor, ...]
    _storage_sizes: tuple[int, ...]

    def resolve_range(self, storage_index: int, offset: int, size: int) -> int:
        if not 0 <= storage_index < len(self._storages):
            raise ValueError("Transfer range references an unknown KV allocation")
        storage = self._storages[storage_index]
        if offset < 0 or size < 0 or offset + size > self._storage_sizes[storage_index]:
            raise ValueError("Transfer range exceeds the imported KV allocation")
        return storage.data_ptr() + offset

    def close(self) -> None:
        self.tensors.clear()
        self._storages = ()


class TorchNPUIPCAdapter:
    """Share NPU allocations through torch-npu multiprocessing handles."""

    def export_storage(self, storage: torch.Tensor) -> KVCacheStorageSpec:
        if storage.device.type != "npu":
            raise ValueError(f"TorchNPUIPCAdapter only supports NPU storage, got {storage.device}")

        from torch.multiprocessing.reductions import reduce_tensor

        _, ipc_args = reduce_tensor(storage)
        return KVCacheStorageSpec(
            handle=cloudpickle.dumps(tuple(ipc_args)),
        )

    def import_storage(self, spec: KVCacheStorageSpec, device_index: int | None) -> tuple[torch.Tensor, int, int]:
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        if type(device_index) is not int or device_index < 0:
            raise ValueError("A valid transfer subprocess device index is required")
        try:
            ipc_args = tuple(cloudpickle.loads(spec.handle))
        except Exception as exc:
            raise ValueError("Malformed torch-npu IPC handle") from exc
        if len(ipc_args) != _TORCH_NPU_IPC_ARG_COUNT:
            raise ValueError("Malformed torch-npu IPC handle")

        # torch-npu stores the producer's logical device index in _share_npu_().
        source_device_index = ipc_args[_STORAGE_DEVICE_INDEX_ARG]
        if type(source_device_index) is not int or source_device_index < 0:
            raise ValueError("Malformed torch-npu IPC source device")
        if source_device_index != device_index:
            raise ValueError(
                f"KV cache handle source device npu:{source_device_index} does not match "
                f"transfer subprocess device npu:{device_index}"
            )
        storage_size_bytes = ipc_args[_STORAGE_SIZE_BYTES_ARG]
        if type(storage_size_bytes) is not int or storage_size_bytes <= 0:
            raise ValueError("Malformed torch-npu IPC storage size")

        torch.npu.set_device(device_index)
        return rebuild_npu_tensor(*ipc_args), device_index, storage_size_bytes


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
    storage_sizes = tuple(_storage_size_bytes(storage) for storage in storages)
    _validate_worker_spec(spec, storage_sizes)
    return ExportedKVCache(spec, tuple(storages), storage_sizes)


def import_worker_kv_caches(
    spec: WorkerKVCacheSpec,
    adapter: KVCacheStorageAdapter | None = None,
    *,
    device_index: int | None = None,
) -> ImportedKVCache:
    """Import each allocation once and rebuild the registered tensor views."""
    _validate_worker_spec(spec)
    adapter = TorchNPUIPCAdapter() if adapter is None else adapter
    imported_storages: list[tuple[torch.Tensor, int | None]] = []
    storage_sizes: list[int] = []
    try:
        for storage_spec in spec.storages:
            storage, imported_device_index, storage_size = adapter.import_storage(storage_spec, device_index)
            if type(storage_size) is not int or storage_size <= 0 or _storage_size_bytes(storage) < storage_size:
                raise ValueError("Imported KV cache storage is smaller than its handle")
            imported_storages.append((storage, imported_device_index))
            storage_sizes.append(storage_size)

        device_indices = {imported_device_index for _, imported_device_index in imported_storages}
        if len(device_indices) > 1:
            raise ValueError("One Worker registration cannot span multiple transfer subprocess devices")
        imported_device_index = next(iter(device_indices), None)
        imported_storage_sizes = tuple(storage_sizes)
        _validate_worker_spec(spec, imported_storage_sizes)
        caches = _rebuild_cache_tensors(spec, imported_storages)
        return ImportedKVCache(
            caches,
            imported_device_index,
            tuple(storage for storage, _ in imported_storages),
            imported_storage_sizes,
        )
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


def _validate_worker_spec(spec: WorkerKVCacheSpec, storage_sizes: tuple[int, ...] | None = None) -> None:
    if not spec.storages:
        raise ValueError("KV cache storage handles are required")
    if not spec.caches:
        raise ValueError("KV cache tensor layouts are required")
    if storage_sizes is not None and (
        len(storage_sizes) != len(spec.storages) or any(size <= 0 for size in storage_sizes)
    ):
        raise ValueError("KV cache storage sizes are invalid")
    for storage in spec.storages:
        if not storage.handle:
            raise ValueError("KV cache storage handle is invalid")
    for name, tensors in spec.caches.items():
        if not isinstance(name, str) or not name or not tensors:
            raise ValueError("KV cache tensor layouts must have non-empty names and values")
        for tensor in tensors:
            if not 0 <= tensor.storage_index < len(spec.storages):
                raise ValueError(f"KV cache {name!r} references an unknown storage")
            dtype = _decode_dtype(tensor.dtype)
            element_size = torch.empty((), dtype=dtype).element_size()
            storage_size = None if storage_sizes is None else storage_sizes[tensor.storage_index]
            _validate_tensor_spec(name, tensor, storage_size, element_size)


def _validate_tensor_spec(name: str, tensor: KVCacheTensorSpec, storage_size: int | None, element_size: int) -> None:
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
    if storage_size is not None and required_bytes > storage_size:
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


__all__ = [
    "ExportedKVCache",
    "ImportedKVCache",
    "KVCacheStorageAdapter",
    "KVCacheStorageSpec",
    "KVCacheTensorSpec",
    "TorchNPUIPCAdapter",
    "WorkerKVCacheSpec",
    "export_worker_kv_caches",
    "import_worker_kv_caches",
]
