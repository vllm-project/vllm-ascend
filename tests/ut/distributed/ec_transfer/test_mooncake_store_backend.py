import json
import sys
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
    client.batch_exists.side_effect = lambda keys: [False] * len(keys)
    client.put_prepared_tensors.side_effect = lambda items: [0] * len(items)
    backend = AscendDeferredStoreBackend(
        client,
        "namespace",
        max_pending_items=max_items,
        max_pending_bytes=max_bytes,
    )
    backend._step_candidates = {"first", "second"}
    return backend, client


def test_deferred_store_budget_and_exists_rejection_do_not_submit_put():
    backend, client = _backend(max_items=1)
    backend.save_output("first", _tensor(1000))
    backend.save_output("second", _tensor(2000))
    assert list(backend._pending) == ["first"]
    client.batch_exists.side_effect = lambda keys: [True] * len(keys)
    assert backend.prepare_saves(["first"]).get("first") is None

    client.batch_exists.assert_called_once()
    client.put_tensor.assert_not_called()
    client.put_prepared_tensors.assert_not_called()
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
        ticket = backend.prepare_saves([identifier]).get(identifier)
        future = backend.submit_prepared_batch([PreparedStorePut(ticket, [10], [128], [ticket.tensor])])
        future.result(timeout=1)
        backend.reap()
        assert backend._pending_bytes == (128 if identifier == "first" else 0)
    assert backend._pending == {}
    backend.shutdown()


def test_encoder_batch_uses_one_exists_and_one_put_call():
    backend, client = _backend()
    first, second = _tensor(1000), _tensor(2000)
    backend.save_output("first", first)
    backend.save_output("second", second)

    prepared = backend.prepare_saves(["first", "second"])
    assert list(prepared) == ["first", "second"]
    client.batch_exists.assert_called_once_with([prepared["first"].pool_key, prepared["second"].pool_key])

    puts = [
        PreparedStorePut(prepared["first"], [10], [128], [first]),
        PreparedStorePut(prepared["second"], [20], [128], [second]),
    ]
    backend.submit_prepared_batch(puts).result(timeout=1)
    backend.reap()

    client.put_prepared_tensors.assert_called_once_with(
        [
            (prepared["first"].pool_key, [10], [128], [first]),
            (prepared["second"].pool_key, [20], [128], [second]),
        ]
    )
    assert backend._pending == {}
    assert backend._pending_bytes == 0
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
    ticket = backend.prepare_saves(["first"]).get("first")
    put = PreparedStorePut(ticket, [10], [128], [ticket.tensor])
    with (
        patch.object(backend._executor, "submit", side_effect=RuntimeError("executor stopped")),
        pytest.raises(RuntimeError, match="executor stopped"),
    ):
        backend.submit_prepared_batch([put])
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
    mooncake_store.__dict__.update(
        MooncakeDistributedStore=MagicMock(return_value=store),
        ObjectDataType=SimpleNamespace(TENSOR="tensor"),
        ReplicateConfig=MagicMock(return_value=MagicMock()),
    )
    mooncake.__dict__["store"] = mooncake_store
    return mooncake, mooncake_store


@pytest.mark.parametrize("topology", ["shared", "independent", "fabric"])
@pytest.mark.parametrize(
    "selectors",
    [
        {},
        {"HCCL_INTRA_PCIE_ENABLE": "0", "HCCL_INTRA_ROCE_ENABLE": "1"},
        {"HCCL_INTRA_PCIE_ENABLE": "1", "HCCL_INTRA_ROCE_ENABLE": "0"},
    ],
)
def test_factory_selects_engine_and_local_buffer(topology, selectors):
    store, transfer, pool = MagicMock(), MagicMock(), MagicMock()
    store.setup.return_value = 0
    transfer._ensure_engine.return_value.get_rpc_port.return_value = 12345
    environment = {"ASCEND_ENABLE_USE_FABRIC_MEM": "0", **selectors}
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
        assert dict(module.os.environ) == environment
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


@pytest.mark.parametrize("protocol", ["hccs:device", "roce:device"])
def test_explicit_store_protocol_is_preserved_without_global_overrides(protocol):
    config = json.dumps(
        {
            "comm_resource_config": {"protocol_desc": [protocol]},
            "store": {"comm_resource_config": {"protocol_desc": [protocol]}},
        }
    )
    with patch.dict(module.os.environ, {"ASCEND_GLOBAL_RESOURCE_CONFIG": config}, clear=True):
        assert module.use_store_independent_te()
        assert dict(module.os.environ) == {"ASCEND_GLOBAL_RESOURCE_CONFIG": config}
