from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.npu_ipc import (
    ExportedKVCache,
    KVCacheStorageSpec,
    NPUEventSpec,
    WorkerKVCacheSpec,
    export_worker_kv_caches,
    import_npu_event,
    import_worker_kv_caches,
)

# isort: on

_UUID_TARGET = "vllm_ascend.distributed.weight_transfer.npu_ipc_engine.npu_generate_uuid"


class _CPUMemoryAdapter:
    def __init__(self):
        self.storages: list[torch.Tensor] = []
        self.import_count = 0

    def export_storage(self, storage: torch.Tensor) -> KVCacheStorageSpec:
        index = len(self.storages)
        self.storages.append(storage)
        return KVCacheStorageSpec(
            size_bytes=storage.untyped_storage().nbytes(),
            device_type="cpu",
            device_uuid="cpu",
            handle_type="test_cpu",
            handle_version=1,
            handle=index.to_bytes(4, byteorder="big"),
        )

    def import_storage(self, spec: KVCacheStorageSpec) -> tuple[torch.Tensor, int | None]:
        self.import_count += 1
        index = int.from_bytes(spec.handle, byteorder="big")
        return self.storages[index], None


def test_shared_storage_is_exported_once_and_views_are_rebuilt() -> None:
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


def test_invalid_tensor_layout_is_rejected_before_import() -> None:
    adapter = _CPUMemoryAdapter()
    exported = export_worker_kv_caches({"layer.0": torch.zeros(8)}, adapter)
    tensor = replace(exported.spec.caches["layer.0"][0], storage_index=1)
    invalid_spec = replace(exported.spec, caches={"layer.0": (tensor,)})

    with pytest.raises(ValueError, match="unknown storage"):
        import_worker_kv_caches(invalid_spec, adapter)

    assert adapter.import_count == 0


def test_exported_and_imported_cache_release_owned_references() -> None:
    adapter = _CPUMemoryAdapter()
    exported = export_worker_kv_caches({"layer.0": torch.zeros(8)}, adapter)
    imported = import_worker_kv_caches(exported.spec, adapter)

    imported.close()
    exported.close()

    assert imported.tensors == {}
    assert imported._storages == ()
    assert exported._storages == ()


def test_ranges_rebase_on_imported_storage_and_reject_out_of_bounds():
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


def test_zero_length_base_address_selects_next_adjacent_allocation():
    specs = tuple(KVCacheStorageSpec(16, "cpu", "cpu", "test", 1, b"handle") for _ in range(2))
    first, second = MagicMock(), MagicMock()
    first.data_ptr.return_value = 100
    second.data_ptr.return_value = 116
    exported = ExportedKVCache(WorkerKVCacheSpec({}, specs), (first, second))
    assert exported.describe_range(116, 0) == (1, 0, 0)


def test_imported_range_is_bounded_by_exported_size_even_if_mapping_is_larger():
    adapter = _CPUMemoryAdapter()
    exported = export_worker_kv_caches({"layer": torch.zeros(4)}, adapter)
    adapter.storages[0] = torch.zeros(64, dtype=torch.uint8)
    imported = import_worker_kv_caches(exported.spec, adapter)
    with pytest.raises(ValueError, match="exceeds"):
        imported.resolve_range(0, 16, 1)


def test_import_npu_event_resolves_local_device_by_uuid() -> None:
    imported_event = MagicMock()
    event_type = MagicMock()
    event_type.from_ipc_handle.return_value = imported_event

    with (
        patch.object(torch.npu, "device_count", return_value=3),
        patch.object(torch.npu, "set_device") as set_device,
        patch.object(torch.npu, "Event", event_type),
        patch(_UUID_TARGET, side_effect=lambda index: f"device-{index}"),
    ):
        result = import_npu_event(NPUEventSpec("device-2", b"event-handle"))

    assert result is imported_event
    set_device.assert_called_once_with(2)
    event_type.from_ipc_handle.assert_called_once_with(2, b"event-handle")


def test_import_npu_event_rejects_unknown_device() -> None:
    with (
        patch.object(torch.npu, "device_count", return_value=2),
        patch(_UUID_TARGET, side_effect=lambda index: f"device-{index}"),
        pytest.raises(ValueError, match="No local NPU matches"),
    ):
        import_npu_event(NPUEventSpec("missing-device", b"event-handle"))
