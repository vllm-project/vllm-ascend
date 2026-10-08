import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (
    EmbeddingStoreError,
)

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import (
    store_client as store_client_module,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.bounce import (
    ASCEND_DIRECT_MEMORY_ALIGNMENT,
    _SourceFragmentPlan,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_client import (
    AscendMooncakeEmbeddingStoreClient,
)

_BOUNCE_ADDRESS = 4 * ASCEND_DIRECT_MEMORY_ALIGNMENT
_BOUNCE_IO_ADDRESS = _BOUNCE_ADDRESS + 128
_DIRECT_ADDRESS = 8 * ASCEND_DIRECT_MEMORY_ALIGNMENT
_HEADER_ADDRESS = 12 * ASCEND_DIRECT_MEMORY_ALIGNMENT
_METADATA = b"meta"


def _make_client():
    store = MagicMock()
    store.register_buffer.return_value = 0
    store.unregister_buffer.return_value = 0
    store.batch_put_from_multi_buffers.return_value = [0]
    bounce_arena = MagicMock()
    bounce = MagicMock(nbytes=4096)
    bounce.data_ptr.return_value = _BOUNCE_ADDRESS
    bounce_arena.tensor = bounce
    bounce_arena.copy.return_value = _BOUNCE_IO_ADDRESS
    registration_transfer = MagicMock()
    registration_transfer.acquire_registration_ranges.side_effect = (
        lambda ranges: [item.address for item in ranges]
    )
    registration_transfer.release_registration_ranges.return_value = True
    client = AscendMooncakeEmbeddingStoreClient(
        store,
        bounce_arena,
        registration_transfer,
        replicate_config="replicate",
    )
    tensor = MagicMock(nbytes=128)
    tensor.device.type = "npu"
    tensor.is_contiguous.return_value = True
    return client, store, bounce_arena, bounce, tensor


def _make_plan(tensor, prefix_nbytes, direct_address, direct_nbytes):
    return _SourceFragmentPlan(
        owner=tensor,
        prefix_nbytes=prefix_nbytes,
        direct_address=direct_address,
        direct_nbytes=direct_nbytes,
        registration_address=direct_address,
        registration_nbytes=direct_nbytes,
    )


def _put(client, tensor, plan):
    header = MagicMock()
    header.data_ptr.return_value = _HEADER_ADDRESS
    header.numel.return_value = len(_METADATA)
    with (
        patch.object(store_client_module.torch.npu, "set_device") as set_device,
        patch.object(store_client_module.torch.npu, "synchronize") as synchronize,
        patch.object(
            store_client_module,
            "_encode_mooncake_tensor_metadata",
            return_value=_METADATA,
        ),
        patch.object(store_client_module, "_plan_source", return_value=plan),
        patch.object(
            store_client_module.torch,
            "empty",
            return_value=header,
        ) as allocate_header,
    ):
        client.put_tensor("key", tensor)
    return set_device, synchronize, allocate_header, header


@pytest.mark.parametrize(
    (
        "prefix_nbytes",
        "direct_address",
        "direct_nbytes",
        "expected_addresses",
        "expected_sizes",
    ),
    [
        (128, None, 0, [_BOUNCE_IO_ADDRESS], [128]),
        (0, _DIRECT_ADDRESS, 128, [_DIRECT_ADDRESS], [128]),
        (32, _DIRECT_ADDRESS, 96, [_BOUNCE_IO_ADDRESS, _DIRECT_ADDRESS], [32, 96]),
    ],
    ids=["all-bounce", "direct-only", "prefix-and-direct"],
)
def test_put_tensor_uses_planned_ascend_buffer_layout(
    prefix_nbytes,
    direct_address,
    direct_nbytes,
    expected_addresses,
    expected_sizes,
):
    client, store, bounce_arena, _, tensor = _make_client()
    lease = MagicMock()
    bounce_arena.acquire.return_value = lease
    plan = _make_plan(tensor, prefix_nbytes, direct_address, direct_nbytes)

    set_device, synchronize, allocate_header, header = _put(client, tensor, plan)

    set_device.assert_called_once_with(tensor.device)
    synchronize.assert_called_once_with(tensor.device)
    allocate_header.assert_called_once_with(
        len(_METADATA),
        dtype=store_client_module.torch.uint8,
        device="cpu",
        pin_memory=True,
    )
    header.copy_.assert_called_once()
    header_address = store.register_buffer.call_args_list[0].args[0]
    store.batch_put_from_multi_buffers.assert_called_once_with(
        ["key"],
        [[header_address, *expected_addresses]],
        [[len(_METADATA), *expected_sizes]],
        "replicate",
    )
    store.register_buffer.assert_called_once_with(header_address, len(_METADATA))

    if prefix_nbytes:
        bounce_arena.acquire.assert_called_once_with(prefix_nbytes)
        bounce_arena.copy.assert_called_once_with(
            lease,
            [(tensor, 0, prefix_nbytes)],
        )
        bounce_arena.release.assert_called_once_with(lease)
    else:
        bounce_arena.acquire.assert_not_called()
        bounce_arena.copy.assert_not_called()
        bounce_arena.release.assert_not_called()

    if direct_address is not None:
        [registration] = (
            client._registration_transfer.acquire_registration_ranges.call_args.args[0]
        )
        assert registration.address == direct_address
        assert registration.nbytes == direct_nbytes
        assert registration.owners == (tensor,)
        client._registration_transfer.release_registration_ranges.assert_called_once_with(
            [direct_address]
        )


def test_safe_put_rejection_releases_temporary_resources():
    client, store, bounce_arena, _, tensor = _make_client()
    lease = bounce_arena.acquire.return_value
    store.batch_put_from_multi_buffers.return_value = [-200]

    with pytest.raises(
        store_client_module.EmbeddingStoreOperationError,
        match="failed to put",
    ):
        _put(client, tensor, _make_plan(tensor, 32, _DIRECT_ADDRESS, 96))

    assert client._poisoned is False
    client._registration_transfer.release_registration_ranges.assert_called_once_with(
        [_DIRECT_ADDRESS]
    )
    bounce_arena.release.assert_called_once_with(lease)


def test_unconfirmed_put_retains_direct_registration_and_bounce_lease():
    client, store, bounce_arena, _, tensor = _make_client()
    lease = bounce_arena.acquire.return_value
    store.batch_put_from_multi_buffers.return_value = [-1]

    with pytest.raises(
        EmbeddingStoreError,
        match="completion is unconfirmed",
    ):
        _put(client, tensor, _make_plan(tensor, 32, _DIRECT_ADDRESS, 96))

    assert client._poisoned is True
    assert bounce_arena in client._unsafe_owners
    assert tensor in client._unsafe_owners
    assert lease in client._unsafe_owners
    client._registration_transfer.release_registration_ranges.assert_not_called()
    bounce_arena.release.assert_not_called()


@pytest.mark.parametrize(
    ("plan_args", "header_status", "direct_failure", "lease_released"),
    [
        ((128, None, 0), -1, False, False),
        ((32, _DIRECT_ADDRESS, 96), 0, True, True),
    ],
    ids=["header-registration", "direct-registration"],
)
def test_registration_failure_releases_only_acquired_lease(
    plan_args,
    header_status,
    direct_failure,
    lease_released,
):
    client, store, bounce_arena, _, tensor = _make_client()
    lease = bounce_arena.acquire.return_value
    store.register_buffer.return_value = header_status
    if direct_failure:
        client._registration_transfer.acquire_registration_ranges.side_effect = (
            RuntimeError("registration failed")
        )

    with pytest.raises(
        store_client_module.EmbeddingStoreOperationError,
        match="Failed to register",
    ):
        _put(client, tensor, _make_plan(tensor, *plan_args))

    assert client._poisoned is False
    assert bounce_arena.release.call_args_list == (
        [call(lease)] if lease_released else []
    )


def test_put_tensor_reuses_pinned_header_registration():
    client, store, _, _, tensor = _make_client()
    plan = _make_plan(tensor, 0, _DIRECT_ADDRESS, 128)

    _, _, _, first_header = _put(client, tensor, plan)
    _, _, second_allocate, _ = _put(client, tensor, plan)

    assert client._put_header_tensor is first_header
    second_allocate.assert_not_called()
    assert first_header.copy_.call_count == 2
    store.register_buffer.assert_called_once_with(
        _HEADER_ADDRESS, len(_METADATA)
    )


def test_close_unregisters_pinned_header_before_closing_store():
    client, store, _, _, _ = _make_client()
    store.close.return_value = 0
    header = MagicMock()
    header.data_ptr.return_value = _HEADER_ADDRESS
    client._put_header_tensor = header

    client.close()

    assert store.mock_calls == [
        call.unregister_buffer(_HEADER_ADDRESS),
        call.close(),
    ]
    assert client._put_header_tensor is None
    client._registration_transfer.close.assert_not_called()


def test_close_does_not_close_shared_registration_transfer():
    client, store, _, _, _ = _make_client()
    store.close.return_value = 0

    client.close()

    assert store.mock_calls == [call.close()]
    client._registration_transfer.close.assert_not_called()


def test_factory_rejects_non_ascend_store_protocol():
    bounce_arena = MagicMock()
    registration_transfer = MagicMock()

    with (
        patch.object(
            store_client_module.MooncakeStoreConfig,
            "load_from_config",
            return_value=MagicMock(protocol="rdma"),
        ),
        pytest.raises(ValueError, match="protocol='ascend'"),
    ):
        store_client_module.create_ascend_mooncake_embedding_store_client(
            bounce_arena,
            registration_transfer,
        )

    registration_transfer._ensure_engine.assert_not_called()


def test_store_setup_reuses_supplied_transfer_engine():
    store = MagicMock()
    store.setup.return_value = 0
    engine = MagicMock()
    config = MagicMock(
        metadata_server="P2PHANDSHAKE",
        global_segment_size=1024,
        local_buffer_size=512,
        protocol="ascend",
        device_name="",
        master_server_address="127.0.0.1:50051",
        tenant_id=store_client_module.DEFAULT_TENANT_ID,
    )

    store_client_module._setup_store_with_engine(
        store,
        config,
        "local-host",
        engine,
    )

    store.setup.assert_called_once_with(
        "local-host",
        "P2PHANDSHAKE",
        1024,
        512,
        "ascend",
        "",
        "127.0.0.1:50051",
        engine=engine,
    )


def test_factory_passes_native_engine_to_store_setup():
    store = MagicMock()
    native_engine = MagicMock()
    engine_adapter = MagicMock()
    engine_adapter.get_engine.return_value = native_engine
    registration_transfer = MagicMock()
    registration_transfer._ensure_engine.return_value = engine_adapter
    config = MagicMock(
        protocol="ascend",
        enable_offload=False,
        tenant_id=store_client_module.DEFAULT_TENANT_ID,
    )

    mooncake = ModuleType("mooncake")
    mooncake.__path__ = []
    mooncake_store = ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = MagicMock(return_value=store)
    mooncake_store.ObjectDataType = SimpleNamespace(TENSOR="tensor")
    replicate_config = MagicMock()
    mooncake_store.ReplicateConfig = MagicMock(return_value=replicate_config)
    mooncake.store = mooncake_store

    with (
        patch.dict(
            sys.modules,
            {"mooncake": mooncake, "mooncake.store": mooncake_store},
        ),
        patch.object(
            store_client_module.MooncakeStoreConfig,
            "load_from_config",
            return_value=config,
        ),
        patch.object(store_client_module, "get_ip", return_value="127.0.0.1"),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "rdma_utils.get_requester_local_hostname",
            return_value="local-host",
        ),
        patch.object(store_client_module, "_setup_store_with_engine") as setup,
    ):
        client = store_client_module.create_ascend_mooncake_embedding_store_client(
            MagicMock(),
            registration_transfer,
        )

    registration_transfer._ensure_engine.assert_called_once_with()
    engine_adapter.get_engine.assert_called_once_with()
    setup.assert_called_once_with(store, config, "local-host", native_engine)
    assert client._registration_transfer is registration_transfer
