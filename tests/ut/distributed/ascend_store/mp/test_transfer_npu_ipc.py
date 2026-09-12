import sys
import types
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.npu_ipc import (
    ExportedKVCache,
    KVCacheStorageSpec,
    TorchNPUIPCAdapter,
    WorkerKVCacheSpec,
    export_worker_kv_caches,
    import_worker_kv_caches,
)

# isort: on


class _CPUMemoryAdapter:
    def __init__(self):
        self.storages: list[torch.Tensor] = []
        self.import_count = 0

    def export_storage(self, storage: torch.Tensor) -> KVCacheStorageSpec:
        index = len(self.storages)
        self.storages.append(storage)
        size_bytes = storage.untyped_storage().nbytes()
        return KVCacheStorageSpec(
            handle=index.to_bytes(4, byteorder="big") + size_bytes.to_bytes(8, byteorder="big"),
        )

    def import_storage(
        self, spec: KVCacheStorageSpec, device_index: int | None
    ) -> tuple[torch.Tensor, int | None, int]:
        self.import_count += 1
        index = int.from_bytes(spec.handle[:4], byteorder="big")
        size_bytes = int.from_bytes(spec.handle[4:], byteorder="big")
        return self.storages[index], None, size_bytes


def _patch_rebuild_npu_tensor(rebuild_func):
    fake_mod = types.ModuleType("torch_npu.multiprocessing.reductions")
    fake_mod.__dict__["rebuild_npu_tensor"] = rebuild_func
    return patch.dict(
        sys.modules,
        {
            "torch_npu.multiprocessing": types.ModuleType("torch_npu.multiprocessing"),
            "torch_npu.multiprocessing.reductions": fake_mod,
        },
    )


def _npu_ipc_spec(device_index: int, size_bytes: int = 16) -> KVCacheStorageSpec:
    return KVCacheStorageSpec(b"opaque torch-npu handle")


def _patch_npu_ipc_args(device_index: int, size_bytes: int = 16):
    ipc_args = (None, None, None, None, None, None, device_index, b"storage", size_bytes) + (None,) * 6
    return patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.npu_ipc.cloudpickle.loads",
        return_value=ipc_args,
    )


def test_shared_storage_is_exported_once_rebuilt_and_released() -> None:
    adapter = _CPUMemoryAdapter()
    storage = torch.arange(32, dtype=torch.float32).view(4, 8)
    expected_slice = storage[1:, ::2]

    exported = export_worker_kv_caches({"layer.0": storage, "layer.1": expected_slice}, adapter=adapter)
    imported = import_worker_kv_caches(exported.spec, adapter)

    assert len(exported.spec.storages) == 1
    assert adapter.import_count == 1
    assert torch.equal(imported.tensors["layer.0"][0], storage)
    assert torch.equal(imported.tensors["layer.1"][0], expected_slice)
    assert imported.tensors["layer.0"][0].untyped_storage().data_ptr() == (
        imported.tensors["layer.1"][0].untyped_storage().data_ptr()
    )

    imported.close()
    exported.close()
    assert imported.tensors == {}
    assert imported._storages == ()
    assert exported._storages == ()


def test_invalid_tensor_layout_is_rejected_before_import() -> None:
    adapter = _CPUMemoryAdapter()
    exported = export_worker_kv_caches({"layer.0": torch.zeros(8)}, adapter)
    tensor = replace(exported.spec.caches["layer.0"][0], storage_index=1)
    invalid_spec = replace(exported.spec, caches={"layer.0": (tensor,)})

    with pytest.raises(ValueError, match="unknown storage"):
        import_worker_kv_caches(invalid_spec, adapter)

    assert adapter.import_count == 0


def test_ranges_rebase_and_stay_within_the_exported_allocation():
    adapter = _CPUMemoryAdapter()
    source = torch.arange(16, dtype=torch.int32)
    exported = export_worker_kv_caches({"layer": source[2:8]}, adapter)
    # Simulate a different virtual address in the importing process.
    adapter.storages[0] = adapter.storages[0].clone()
    imported = import_worker_kv_caches(exported.spec, adapter)
    transfer_range = exported.describe_range(source.data_ptr() + 8, 24)
    assert transfer_range == (0, 8, 24)
    assert imported.resolve_range(*transfer_range) == imported.tensors["layer"][0].data_ptr()
    assert imported.resolve_range(*transfer_range) != source.data_ptr() + 8
    with pytest.raises(ValueError, match="outside"):
        exported.describe_range(source.data_ptr() + source.nbytes - 1, 4)
    with pytest.raises(ValueError, match="exceeds"):
        imported.resolve_range(0, 60, 8)
    with pytest.raises(ValueError, match="unknown"):
        imported.resolve_range(1, 0, 1)

    # An address exactly at an adjacent allocation belongs to the next
    # allocation, including a zero-length registration range.
    specs = tuple(KVCacheStorageSpec(b"handle") for _ in range(2))
    first, second = MagicMock(), MagicMock()
    first.data_ptr.return_value = 100
    second.data_ptr.return_value = 116
    adjacent = ExportedKVCache(WorkerKVCacheSpec({}, specs), (first, second), (16, 16))
    assert adjacent.describe_range(116, 0) == (1, 0, 0)

    # The allocation size decoded from the handle bounds every transfer range.
    adapter = _CPUMemoryAdapter()
    exported = export_worker_kv_caches({"layer": torch.zeros(4)}, adapter)
    adapter.storages[0] = torch.zeros(64, dtype=torch.uint8)
    imported = import_worker_kv_caches(exported.spec, adapter)
    with pytest.raises(ValueError, match="exceeds"):
        imported.resolve_range(0, 16, 1)


def test_npu_import_accepts_matching_inherited_logical_device() -> None:
    rebuilt = MagicMock()
    rebuild = MagicMock(return_value=rebuilt)

    with (
        _patch_npu_ipc_args(2),
        _patch_rebuild_npu_tensor(rebuild),
        patch.object(torch.npu, "set_device") as set_device,
    ):
        storage, device_index, size_bytes = TorchNPUIPCAdapter().import_storage(_npu_ipc_spec(2), 2)

    assert storage is rebuilt
    assert device_index == 2
    assert size_bytes == 16
    set_device.assert_called_once_with(2)
    assert rebuild.call_args.args[6] == 2
    assert rebuild.call_args.args[8] == 16


def test_npu_import_rejects_different_parent_and_child_logical_devices() -> None:
    rebuild = MagicMock()

    with (
        _patch_npu_ipc_args(1),
        _patch_rebuild_npu_tensor(rebuild),
        pytest.raises(ValueError, match="source device npu:1.*subprocess device npu:2"),
    ):
        TorchNPUIPCAdapter().import_storage(_npu_ipc_spec(1), 2)

    rebuild.assert_not_called()
