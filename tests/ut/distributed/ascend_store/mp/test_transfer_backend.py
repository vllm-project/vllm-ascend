import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    MooncakeStoreConfig,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.mooncake_backend import MPMooncakeBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.service import TransferService
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer import KVTransferProcess
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer_backend import (
    TransferBackend,
    create_transfer_backend,
)


def test_mooncake_ssd_setup_uses_worker_identity_without_distributed_groups(tmp_path):
    backend = MPMooncakeBackend.__new__(MPMooncakeBackend)
    backend._device_index = 2
    backend._process_global_rank = 7
    backend.parallel_config = None
    backend.config = MooncakeStoreConfig(
        metadata_server="P2PHANDSHAKE",
        global_segment_size=1 << 30,
        local_buffer_size=1 << 30,
        protocol="ascend",
        device_name="",
        master_server_address="127.0.0.1:50051",
        preferred_segment=False,
        prefer_alloc_in_same_node=True,
        enable_ssd_offload=True,
        ssd_offload_path=str(tmp_path),
    )
    backend.local_seg = None
    backend._use_fabric_mem = True
    backend._use_store_independent_te = False
    backend._contribute_memory = True
    store = MagicMock()
    store.setup.return_value = 0
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend"
    with (
        patch.object(sys.modules["mooncake.store"], "MooncakeDistributedStore", return_value=store, create=True),
        patch(f"{module}.get_ip", return_value="127.0.0.1"),
        patch(f"{module}.get_global_rank", side_effect=AssertionError("distributed rank lookup")),
        patch(f"{module}._mooncake_setup_supports_ssd_offload", return_value=True),
    ):
        backend._setup_store()

    assert store.setup.call_args.kwargs["ssd_offload_path"] == os.path.join(str(tmp_path), "rank_7")
    with patch.object(torch.npu, "set_device") as set_device:
        backend.set_device()
    set_device.assert_called_once_with(2)


def test_transfer_backend_factories_keep_worker_identity(tmp_path, monkeypatch):
    target = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.mooncake_backend.MPMooncakeBackend"
    with patch(target) as factory:
        mooncake = create_transfer_backend("mooncake", device_index=2, global_rank=7, lazy_init=True)

    factory.assert_called_once_with(2, 7, lazy_init=True)
    assert mooncake.backend is factory.return_value

    config_path = tmp_path / "memcache.conf"
    config_path.write_text("ock.mmc.local_service.protocol=device_sdma\n", encoding="utf-8")
    monkeypatch.setenv("MMC_LOCAL_CONFIG_PATH", str(config_path))

    memcache = create_transfer_backend("memcache", device_index=2, global_rank=7, lazy_init=True)

    assert memcache.device_index == 2
    assert memcache.backend.device_id == 2
    assert memcache.backend._lazy_init is True


def test_registration_rollback_retains_only_failed_unregistrations():
    backend = MagicMock()
    backend.store.register_buffer.side_effect = [0, 0, -1]
    backend.store.unregister_buffer.side_effect = [-2, 0, 0]
    adapter = TransferBackend("memcache", backend, 0)
    with pytest.raises(RuntimeError, match="unregistration failed"):
        adapter.register_buffer([100, 200, 300], [10, 10, 10])
    assert adapter._registered == [(200, 10)]
    adapter.close()
    assert not adapter._registered
    assert backend.store.unregister_buffer.call_args_list[-1].args == (200, 10)


def test_control_operations_cross_process_and_service_boundaries():
    backend = MagicMock()
    service = TransferService.__new__(TransferService)
    service.backend = TransferBackend("memcache", backend, 0)
    service.backend.set_device = MagicMock()

    assert service.execute("batch_alloc", (["k1"], [64], 300_000)) is backend.batch_alloc.return_value

    backend.batch_alloc.assert_called_once_with(["k1"], [64], 300_000)

    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.TransferProcess"):
        process = KVTransferProcess({})
    process.client.call.reset_mock()
    process.batch_alloc(["k1"], [64], 300_000)
    process.client.call.assert_called_once_with("batch_alloc", (["k1"], [64], 300_000))

    backend = MagicMock()
    service = TransferService.__new__(TransferService)
    service.backend = TransferBackend("mooncake", backend, 0)
    service.backend.set_device = MagicMock()
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.TransferProcess"):
        process = KVTransferProcess({})

    for method_name, operation, args, payload in (
        ("batch_put_start", "batch_put_start", (["k1"], [64]), (["k1"], [64])),
        ("batch_get_start", "batch_get_start", (["k1"],), ["k1"]),
        ("batch_get_end", "batch_get_end", (["k1"],), ["k1"]),
    ):
        process.client.call.reset_mock()
        getattr(process, method_name)(*args)
        process.client.call.assert_called_once_with(operation, payload)

        service.execute(operation, payload)
        getattr(backend, method_name).assert_called_once_with(*args)


def test_transfer_backend_forwards_capabilities_and_range_operations():
    backend = MagicMock()
    backend.requires_exists_before_put = False
    adapter = TransferBackend("mooncake", backend, 0)
    assert adapter.requires_exists_before_put is False

    calls = (
        ("batch_put_start", (["k"], [8])),
        ("batch_copy_put", (["k"], [[100]], [[8]], [[0]])),
        ("batch_commit", (["k"],)),
        ("batch_revoke", (["k"],)),
        ("batch_get_start", (["k"],)),
        ("batch_copy_get", (["k"], [[100]], [[8]], [[0]])),
        ("batch_get_end", (["k"],)),
    )
    for method_name, args in calls:
        assert getattr(adapter, method_name)(*args) is getattr(backend, method_name).return_value
        getattr(backend, method_name).assert_called_once_with(*args)


def test_lazy_backends_keep_their_native_registration_paths():
    backend = MagicMock()
    backend._lazy_init = True
    backend._store_initialized = False
    backend.store = None
    adapter = TransferBackend("memcache", backend, 0)

    adapter.register_buffer([100], [10])

    backend.register_buffer.assert_called_once_with([100], [10])
    assert adapter._registered == [(100, 10)]

    backend._store_initialized = True
    backend.store = MagicMock()
    backend.store.unregister_buffer.return_value = 0
    adapter.close()
    backend.store.unregister_buffer.assert_called_once_with(100, 10)

    yuanrong = MagicMock()
    adapter = TransferBackend("yuanrong", yuanrong, 0)

    adapter.register_buffer([100], [10])

    yuanrong.register_buffer.assert_called_once_with([100], [10])
    adapter.close()
