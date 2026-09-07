from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.npu_ipc import (
    KVCacheStorageSpec,
    NPUEventSpec,
    export_worker_kv_caches,
    import_npu_event,
    import_worker_kv_caches,
    record_npu_event,
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


def test_record_npu_event_exports_current_device_event() -> None:
    event = MagicMock()
    event.ipc_handle.return_value = b"event-handle"
    event_type = MagicMock(return_value=event)

    with (
        patch.object(torch.npu, "current_device", return_value=2),
        patch.object(torch.npu, "Event", event_type),
        patch(_UUID_TARGET, return_value="device-2"),
    ):
        exported = record_npu_event()

    assert exported.spec == NPUEventSpec("device-2", b"event-handle")
    event_type.assert_called_once_with(interprocess=True)
    event.record.assert_called_once_with()
    event.ipc_handle.assert_called_once_with()

    exported.close()
    assert exported._event is None


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
