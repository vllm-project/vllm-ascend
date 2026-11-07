# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed import mooncake_store
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker


@pytest.fixture
def common_worker_methods(monkeypatch):
    """Restore all methods touched by the patch, including its config class."""
    targets = (
        (worker, "MooncakeStoreConfig"),
        (worker.MooncakeStoreWorker, "_setup_store"),
        (worker.MooncakeStoreWorker, "lookup"),
        (worker.KVTransferThread, "__init__"),
        (worker.KVCacheStoreSendingThread, "_handle_request"),
        (worker.KVCacheStoreRecvingThread, "_handle_request"),
        (worker.MooncakeStoreWorker, "start_load_kv"),
        (worker.MooncakeStoreWorker, "wait_for_save"),
        (worker.MooncakeStoreWorker, "build_connector_worker_meta"),
    )
    originals = [(target, name, getattr(target, name)) for target, name in targets]
    for target, name, original in originals:
        monkeypatch.setattr(target, name, original)
    return originals


@pytest.fixture
def fabric_patch(monkeypatch, request, common_worker_methods):
    """Load only this patch and restore the shared upstream classes afterwards."""
    monkeypatch.setenv("ASCEND_ENABLE_USE_FABRIC_MEM", "1")
    monkeypatch.delenv("MOONCAKE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("MOONCAKE_MASTER", raising=False)
    monkeypatch.delenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", raising=False)
    config = getattr(request, "param", {"enable_ascend_fabric_patch": True})
    kv_transfer_config = (
        KVTransferConfig(kv_connector="MooncakeStoreConnector", kv_role="kv_both", kv_connector_extra_config=config)
        if config is not None
        else None
    )
    npu = SimpleNamespace(current_device=lambda: 3, set_device=MagicMock())
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_mooncake_store.py"
    spec = importlib.util.spec_from_file_location("mooncake_fabric_patch_under_test", source)
    module = importlib.util.module_from_spec(spec)
    if kv_transfer_config is None:
        spec.loader.exec_module(module)
    else:
        with set_current_vllm_config(SimpleNamespace(kv_transfer_config=kv_transfer_config)):
            spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "fabric_patch,enabled",
    [
        (None, False),
        ({}, False),
        ({"enable_ascend_fabric_patch": False}, False),
        ({"enable_ascend_fabric_patch": True}, True),
    ],
    indirect=["fabric_patch"],
    ids=["no-context", "default-off", "explicit-off", "enabled"],
)
def test_patch_switch_and_repeated_installation(fabric_patch, common_worker_methods, enabled):
    replacements = []
    for target, name, original in common_worker_methods:
        replacement = getattr(target, name)
        assert (replacement is not original) is enabled
        replacements.append((target, name, replacement))
    torch.npu.set_device.assert_not_called()
    if not enabled:
        return
    fabric_patch.apply_mooncake_store_patch(
        KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"enable_ascend_fabric_patch": True},
        )
    )
    for target, name, replacement in replacements:
        assert getattr(target, name) is replacement


@pytest.mark.parametrize("fabric_patch", [None], indirect=True)
@pytest.mark.parametrize(
    "store_flags,error",
    [
        ([True], None),
        ([], None),
        (["true"], "must be a JSON boolean"),
        ([True, False], "must use the same enable_ascend_fabric_patch"),
    ],
)
def test_multi_connector_patch_options(fabric_patch, common_worker_methods, store_flags, error):
    children = [
        {"kv_connector": "MooncakeStoreConnector", "kv_connector_extra_config": {"enable_ascend_fabric_patch": value}}
        for value in store_flags
    ]
    children.append(
        {"kv_connector": "MooncakeConnectorV1", "kv_connector_extra_config": {"enable_ascend_fabric_patch": True}}
    )
    config = KVTransferConfig(
        kv_connector="MultiConnector", kv_role="kv_both", kv_connector_extra_config={"connectors": children}
    )
    if error:
        with pytest.raises(ValueError, match=error):
            fabric_patch.apply_mooncake_store_patch(config)
    else:
        fabric_patch.apply_mooncake_store_patch(config)
    expected = bool(store_flags) and error is None
    for target, name, original in common_worker_methods:
        assert (getattr(target, name) is not original) is expected


def _fabric_config(module, **overrides):
    values = dict(
        metadata_server="P2PHANDSHAKE",
        master_server_address="127.0.0.1:50051",
        protocol="ascend",
        device_name="",
    )
    values.update(overrides)
    return module.AscendMooncakeStoreConfig(**values)


def test_load_config_preserves_fabric_subclass(fabric_patch, tmp_path, monkeypatch):
    path = tmp_path / "store.json"
    path.write_text(
        json.dumps(
            dict(
                protocol="ascend",
                mode="standalone-store",
                master_server_address="127.0.0.1:50051",
                global_segment_size=0,
            )
        )
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(path))
    config = worker.MooncakeStoreConfig.load_from_config()
    assert isinstance(config, fabric_patch.AscendMooncakeStoreConfig)
    assert config.use_ascend_fabric_mem
    assert config.global_segment_size == 0
    assert config.local_buffer_size == mooncake_store.DEFAULT_LOCAL_BUFFER_SIZE


def test_fabric_config_requires_ascend_protocol(fabric_patch):
    with pytest.raises(ValueError, match="requires protocol='ascend'"):
        _fabric_config(fabric_patch, protocol="rdma")


def _setup_fabric_worker(module):
    instance = object.__new__(worker.MooncakeStoreWorker)
    backend = MagicMock()
    backend.setup.return_value = 0
    backend.batch_is_exist.return_value = [1]
    instance.store = backend
    instance._setup_store(_fabric_config(module))
    return instance, backend


def test_fabric_initializes_on_serving_not_registration_or_cold_lookup(fabric_patch):
    instance, backend = _setup_fabric_worker(fabric_patch)
    instance.kv_send_thread = None
    assert instance.store.register_buffer(1024, 512) == 0
    assert instance.lookup(1024, [b"hash"]).hit_length == 0
    backend.setup.assert_not_called()
    backend.register_buffer.assert_not_called()
    backend.batch_is_exist.assert_not_called()
    torch.npu.set_device.assert_not_called()
    assert instance.build_connector_worker_meta() is None
    assert instance.build_connector_worker_meta() is None
    backend.setup.assert_called_once()
    backend.batch_is_exist.assert_not_called()
    assert instance.store.batch_is_exist(["key"]) == [1]
    backend.setup.assert_called_once()
    torch.npu.set_device.assert_called_once_with(3)


def test_fabric_setup_is_once_under_concurrent_first_uses(fabric_patch):
    instance, backend = _setup_fabric_worker(fabric_patch)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(instance.store.batch_is_exist(["key"]))) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert results == [[1]] * 4
    backend.setup.assert_called_once()
    assert torch.npu.set_device.call_count == 4


def test_fabric_setup_failure_retries_before_executing_operation(fabric_patch):
    instance, backend = _setup_fabric_worker(fabric_patch)
    backend.setup.side_effect = [-1, 0]
    with pytest.raises(RuntimeError, match="Initialize MooncakeDistributedStore failed"):
        instance.store.batch_is_exist(["key"])
    backend.batch_is_exist.assert_not_called()
    assert instance.store.batch_is_exist(["key"]) == [1]
    assert backend.setup.call_count == 2


@pytest.mark.parametrize("initialize", [False, True])
def test_fabric_close_is_idempotent_and_rejects_later_operations(fabric_patch, initialize):
    instance, backend = _setup_fabric_worker(fabric_patch)
    if initialize:
        instance.store.batch_is_exist(["key"])
    instance.store.close()
    instance.store.close()
    assert backend.close.call_count == int(initialize)
    with pytest.raises(RuntimeError, match="store is closed"):
        instance.store.batch_is_exist(["key"])


@pytest.mark.parametrize("thread_cls", [worker.KVCacheStoreSendingThread, worker.KVCacheStoreRecvingThread])
def test_transfer_handler_binds_device_before_event_or_io(fabric_patch, thread_cls):
    args = [MagicMock(), MagicMock(), [], 16, 0]
    if thread_cls is worker.KVCacheStoreSendingThread:
        args.extend([[], "kv_both"])
    args.append(threading.Event())
    thread = thread_cls(*args)
    torch.npu.set_device.assert_not_called()

    class RequestReached(Exception):
        pass

    class Request:
        def __getattr__(self, name):
            # Both handlers must bind before accessing any request data.
            torch.npu.set_device.assert_called_once_with(3)
            raise RequestReached

    for _ in range(2):
        with pytest.raises(RequestReached):
            thread._handle_request(Request())
    torch.npu.set_device.assert_called_once_with(3)
