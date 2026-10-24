import ast
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import store_backend as module
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_backend import (
    AscendDeferredStoreBackend,
    PreparedStorePut,
)


def _tensor(storage_ptr: int = 1000, storage_bytes: int = 128) -> MagicMock:
    tensor = MagicMock()
    tensor.is_contiguous.return_value = True
    tensor.device.type = "npu"
    storage = tensor.untyped_storage.return_value
    storage.data_ptr.return_value = storage_ptr
    storage.nbytes.return_value = storage_bytes
    return tensor


def _backend(*, max_items: int = 32, max_bytes: int = 2 * 1024**3):
    client = MagicMock()
    client.exists.return_value = False
    backend = AscendDeferredStoreBackend(
        client,
        "namespace",
        max_pending_items=max_items,
        max_pending_bytes=max_bytes,
    )
    backend._step_candidates = {"first", "second"}
    return backend, client


def test_deferred_store_admission_and_exists_rejection_do_not_submit_put():
    backend, client = _backend()
    tensor = _tensor()
    client.exists.return_value = True

    backend.save_output("first", tensor)
    assert backend.prepare_save("first") is None

    client.exists.assert_called_once()
    client.put_tensor.assert_not_called()
    client.put_prepared_tensor.assert_not_called()
    assert backend._pending == {}
    assert backend._pending_bytes == 0
    backend.shutdown()


def test_deferred_store_pending_budget_rejection_leaves_only_admitted_save():
    backend, _ = _backend(max_items=1)

    backend.save_output("first", _tensor(1000))
    backend.save_output("second", _tensor(2000))

    assert list(backend._pending) == ["first"]
    backend.shutdown()


def test_prepared_store_put_runs_on_single_publisher_and_reaps_storage():
    backend, client = _backend()
    tensor = _tensor()
    backend.save_output("first", tensor)
    ticket = backend.prepare_save("first")
    assert ticket is not None

    future = backend.submit_prepared(
        PreparedStorePut(
            ticket=ticket,
            addresses=[10, 20, 30],
            sizes=[24, 32, 72],
            owners=[tensor],
        )
    )
    future.result(timeout=1)
    backend.reap()

    client.put_prepared_tensor.assert_called_once_with(
        ticket.pool_key,
        [10, 20, 30],
        [24, 32, 72],
        [tensor],
    )
    assert backend._pending == {}
    assert backend._pending_bytes == 0
    backend.shutdown()


def test_pending_views_charge_storage_until_last_save_is_reaped():
    backend, client = _backend(max_bytes=128)
    first = _tensor(1000)
    second = _tensor(1000)
    second.untyped_storage.return_value = first.untyped_storage()
    backend.save_output("first", first)
    backend.save_output("second", second)
    assert backend._pending_bytes == 128
    assert len(backend._pending) == 2

    for identifier in ("first", "second"):
        ticket = backend.prepare_save(identifier)
        future = backend.submit_prepared(PreparedStorePut(ticket, [10], [128], [ticket.tensor]))
        future.result(timeout=1)
        backend.reap()
        assert backend._pending_bytes == (128 if identifier == "first" else 0)
    assert backend._pending == {}
    backend.shutdown()


def test_shutdown_drains_unselected_deferred_save_through_original_path():
    backend, client = _backend()
    tensor = _tensor()
    event = MagicMock()
    with patch.object(module.torch.npu, "Event", return_value=event):
        backend.save_output("first", tensor)
    event.record.assert_called_once_with(module.torch.npu.current_stream(tensor.device))
    client.put_tensor.side_effect = lambda *_: event.synchronize.assert_called_once_with()
    backend.shutdown()

    client.exists.assert_called_once()
    pool_key = client.exists.call_args.args[0]
    client.put_tensor.assert_called_once_with(pool_key, tensor)
    client.close.assert_called_once_with()


def test_prepared_submission_failure_leaves_ticket_abortable():
    backend, _ = _backend()
    backend.save_output("first", _tensor())
    ticket = backend.prepare_save("first")
    put = PreparedStorePut(ticket, [10], [128], [ticket.tensor])
    with (
        patch.object(backend._executor, "submit", side_effect=RuntimeError("executor stopped")),
        pytest.raises(RuntimeError, match="executor stopped"),
    ):
        backend.submit_prepared(put)
    backend.abort_prepared(ticket)
    assert backend._pending == {}
    assert backend._pending_bytes == 0
    backend.shutdown()


def _make_store_config():
    return MagicMock(
        metadata_server="P2PHANDSHAKE",
        global_segment_size=1024,
        local_buffer_size=512,
        protocol="ascend",
        device_name="",
        master_server_address="127.0.0.1:50051",
        tenant_id=module.DEFAULT_TENANT_ID,
        enable_offload=False,
        mode="embedded",
    )


def _make_mooncake_modules(store):
    mooncake = ModuleType("mooncake")
    mooncake.__path__ = []
    mooncake_store = ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = MagicMock(return_value=store)
    mooncake_store.ObjectDataType = SimpleNamespace(TENSOR="tensor")
    mooncake_store.ReplicateConfig = MagicMock(return_value=MagicMock())
    mooncake.store = mooncake_store
    return mooncake, mooncake_store


@pytest.mark.parametrize(
    ("ec_config", "expected_selectors"),
    [
        ({"protocol_desc": ["roce:device"]}, ("0", "1")),
        ({"protocol_desc": ["hccs:device"]}, (None, None)),
        ({"protocol_desc": ["hccs:device", "roce:device"]}, (None, None)),
        ({}, (None, None)),
        (None, (None, None)),
    ],
    ids=["explicit-roce", "hccs", "multiple-protocols", "implicit", "store-only"],
)
def test_independent_ec_transport_keeps_only_explicit_roce_selectors(ec_config, expected_selectors):
    resource_config = json.dumps(
        {
            **({"comm_resource_config": ec_config} if ec_config is not None else {}),
            "store": {
                "comm_resource_config": {"protocol_desc": ["roce:device"]},
            },
        }
    )
    with patch.dict(
        module.os.environ,
        {
            "ASCEND_ENABLE_USE_FABRIC_MEM": "0",
            "ASCEND_GLOBAL_RESOURCE_CONFIG": resource_config,
            "HCCL_INTRA_PCIE_ENABLE": "1",
            "HCCL_INTRA_ROCE_ENABLE": "0",
        },
        clear=True,
    ):
        module.configure_non_fabric_transport()

        assert "ASCEND_ENABLE_USE_FABRIC_MEM" not in module.os.environ
        assert (
            tuple(module.os.environ.get(key) for key in ("HCCL_INTRA_PCIE_ENABLE", "HCCL_INTRA_ROCE_ENABLE"))
            == expected_selectors
        )
        assert module.os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] == resource_config


@pytest.mark.parametrize("has_store", [False, True], ids=["consumer", "producer"])
def test_roce_selectors_survive_until_lazy_channel_creation_with_or_without_store(has_store):
    resource_config = json.dumps(
        {
            "comm_resource_config": {"protocol_desc": ["roce:device"]},
            "store": {"comm_resource_config": {"protocol_desc": ["roce:device"]}},
        }
    )
    store = MagicMock()
    store.setup.return_value = 0
    mooncake, mooncake_store = _make_mooncake_modules(store)

    def selectors():
        return tuple(module.os.environ.get(key) for key in ("HCCL_INTRA_PCIE_ENABLE", "HCCL_INTRA_ROCE_ENABLE"))

    with (
        patch.dict(module.os.environ, {"ASCEND_GLOBAL_RESOURCE_CONFIG": resource_config}, clear=True),
        patch.dict(sys.modules, {"mooncake": mooncake, "mooncake.store": mooncake_store}),
        patch.object(module.MooncakeStoreConfig, "load_from_config", return_value=_make_store_config()),
        patch.object(module, "get_ip", return_value="127.0.0.1"),
    ):
        # Both roles execute this before creating their EC TE. Only producers
        # subsequently create a Store. HCCL may read the env much later.
        module.configure_non_fabric_transport()
        assert selectors() == ("0", "1")
        if has_store:
            module.create_ascend_mooncake_embedding_store_client(MagicMock(), MagicMock())
            store.setup.assert_called_once()
        else:
            store.setup.assert_not_called()
        assert selectors() == ("0", "1")
        assert module.os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] == resource_config


@pytest.mark.parametrize("topology", ["shared", "independent", "fabric"])
def test_factory_selects_engine_and_local_buffer(topology):
    store, transfer, pool = MagicMock(), MagicMock(), MagicMock()
    store.setup.return_value = 0
    transfer._ensure_engine.return_value.get_rpc_port.return_value = 12345
    environment = {}
    if topology == "fabric":
        environment["ASCEND_ENABLE_USE_FABRIC_MEM"] = "1"
    elif topology == "independent":
        environment["ASCEND_GLOBAL_RESOURCE_CONFIG"] = "{}"
    mooncake, mooncake_store = _make_mooncake_modules(store)
    with (
        patch.dict(module.os.environ, environment, clear=True),
        patch.dict(sys.modules, {"mooncake": mooncake, "mooncake.store": mooncake_store}),
        patch.object(module.MooncakeStoreConfig, "load_from_config", return_value=_make_store_config()),
        patch.object(module, "get_ip", return_value="127.0.0.1"),
    ):
        client = module.create_ascend_mooncake_embedding_store_client(pool, transfer)
        if topology != "fabric":
            assert module.os.environ["HCCL_INTRA_PCIE_ENABLE"] == "0"
            assert module.os.environ["HCCL_INTRA_ROCE_ENABLE"] == "1"
    shared = topology == "shared"
    engine = transfer._ensure_engine.return_value
    kwargs = {"engine": engine.get_engine.return_value} if shared else {}
    store.setup.assert_called_once_with(
        "127.0.0.1:12345" if shared else "127.0.0.1",
        "P2PHANDSHAKE",
        1024,
        512 if shared else 0,
        "ascend",
        "",
        "127.0.0.1:50051",
        **kwargs,
    )
    if shared:
        transfer._ensure_engine.assert_called_once_with()
    else:
        transfer._ensure_engine.assert_not_called()
    assert client._staging_pool is pool
    store.register_buffer.assert_not_called()


@pytest.mark.parametrize("native_only", [False, True], ids=["python-env-change", "native-env-change"])
def test_ec_engine_restores_selectors_after_initialization_and_registration(native_only):
    import ctypes

    from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import transfer as transfer_module

    config = json.dumps({"comm_resource_config": {"protocol_desc": ["roce:device"]}})
    libc = ctypes.CDLL(None)
    libc.getenv.argtypes = [ctypes.c_char_p]
    libc.getenv.restype = ctypes.c_char_p
    libc.unsetenv.argtypes = [ctypes.c_char_p]
    libc.unsetenv.restype = ctypes.c_int

    def clear_selectors(*args, **kwargs):
        for key in ("HCCL_INTRA_PCIE_ENABLE", "HCCL_INTRA_ROCE_ENABLE"):
            if native_only:
                assert libc.unsetenv(key.encode()) == 0
                assert key in module.os.environ  # Python mapping is stale.
            else:
                module.os.environ.pop(key, None)
        return engine

    engine = MagicMock()
    engine.batch_register_memory.side_effect = lambda *args: (clear_selectors(), 0)[1]
    with (
        patch.dict(module.os.environ, {"ASCEND_GLOBAL_RESOURCE_CONFIG": config}, clear=True),
        patch.object(transfer_module.global_te, "get_transfer_engine", side_effect=clear_selectors),
        patch.object(transfer_module.torch.npu, "set_device"),
    ):
        transfer = transfer_module.AscendMooncakeTransfer("host", 0)
        assert transfer._ensure_engine() is engine
        assert libc.getenv(b"HCCL_INTRA_PCIE_ENABLE") == b"0"
        assert libc.getenv(b"HCCL_INTRA_ROCE_ENABLE") == b"1"
        tensor = MagicMock(nbytes=4096)
        tensor.data_ptr.return_value = 2097152
        assert transfer.register_memory(tensor) == 0
        assert libc.getenv(b"HCCL_INTRA_PCIE_ENABLE") == b"0"
        assert libc.getenv(b"HCCL_INTRA_ROCE_ENABLE") == b"1"
        assert module.os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] == config


def test_four_card_roce_config_keeps_both_roles_on_roce():
    # Read only the constant: importing E2E would require its NPU/server fixtures.
    root = Path(__file__).resolve().parents[4]
    path = root / "tests/e2e/pull_request/four_card/test_ec_mooncake_cross_encoder_store.py"
    tree = ast.parse(path.read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "STORE_ROCE_RESOURCE_CONFIG" for target in node.targets)
    )
    resource_config = ast.literal_eval(assignment.value.args[0])
    assert resource_config["store"]["comm_resource_config"]["protocol_desc"] == ["roce:device"]
    assert resource_config["comm_resource_config"]["protocol_desc"] == ["roce:device"]
    # Exercise the same selector policy as both actual EC worker roles.
    with patch.dict(module.os.environ, {"ASCEND_GLOBAL_RESOURCE_CONFIG": json.dumps(resource_config)}, clear=True):
        for _ in ("producer", "consumer"):
            module.configure_non_fabric_transport()
            assert module.os.environ["HCCL_INTRA_PCIE_ENABLE"] == "0"
            assert module.os.environ["HCCL_INTRA_ROCE_ENABLE"] == "1"
