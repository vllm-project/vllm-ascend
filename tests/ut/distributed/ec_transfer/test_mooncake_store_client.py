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
    _RegistrationRangePlan,
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


def _make_client(*, use_fabric_mem=False, use_store_independent_te=False):
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
    registration_transfer.acquire_registration_ranges.side_effect = lambda ranges: [item.address for item in ranges]
    registration_transfer.release_registration_ranges.return_value = True
    client = AscendMooncakeEmbeddingStoreClient(
        store,
        bounce_arena,
        registration_transfer,
        replicate_config="replicate",
        use_fabric_mem=use_fabric_mem,
        use_store_independent_te=use_store_independent_te,
    )
    tensor = MagicMock(nbytes=128)
    tensor.device.type = "npu"
    tensor.is_contiguous.return_value = True
    return client, store, bounce_arena, bounce, tensor


def _make_store_config():
    return MagicMock(
        metadata_server="P2PHANDSHAKE",
        global_segment_size=1024,
        local_buffer_size=512,
        protocol="ascend",
        device_name="",
        master_server_address="127.0.0.1:50051",
        tenant_id=store_client_module.DEFAULT_TENANT_ID,
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


def test_independent_store_keeps_staging_registration_until_process_exit():
    client, store, _, _, tensor = _make_client(use_store_independent_te=True)
    tensor.data_ptr.return_value = _DIRECT_ADDRESS
    client.register_staging_buffer(tensor)
    client.register_staging_buffer(tensor)
    retained = store_client_module._PROCESS_LIFETIME_STORE_OWNERS
    before = len(retained)
    try:
        client.close()
        client.close()

        store.register_buffer.assert_called_once_with(_DIRECT_ADDRESS, 128)
        assert retained[before:] == [(store, tensor)]
        store.unregister_buffer.assert_not_called()
        store.close.assert_not_called()
    finally:
        del retained[before:]


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
    metadata_tensor = MagicMock(nbytes=len(_METADATA))
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
            "frombuffer",
            return_value=metadata_tensor,
        ),
        patch.object(
            store_client_module.torch,
            "empty",
            return_value=header,
        ) as allocate_header,
    ):
        client.put_tensor("key", tensor)
    return set_device, synchronize, allocate_header, header, metadata_tensor


@pytest.mark.parametrize(
    (
        "prefix_nbytes",
        "direct_address",
        "direct_nbytes",
        "expected_addresses",
        "expected_sizes",
    ),
    [
        (128, None, 0, [], []),
        (0, _DIRECT_ADDRESS, 128, [_DIRECT_ADDRESS], [128]),
        (32, _DIRECT_ADDRESS, 96, [_DIRECT_ADDRESS], [96]),
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

    set_device, synchronize, allocate_header, _, metadata_tensor = _put(client, tensor, plan)

    set_device.assert_called_once_with(tensor.device)
    synchronize.assert_called_once_with(tensor.device)
    allocate_header.assert_not_called()
    store.batch_put_from_multi_buffers.assert_called_once_with(
        ["key"],
        [[_BOUNCE_IO_ADDRESS, *expected_addresses]],
        [[len(_METADATA) + prefix_nbytes, *expected_sizes]],
        "replicate",
    )
    bounce_arena.acquire.assert_called_once_with(len(_METADATA) + prefix_nbytes)
    expected_copies = [(metadata_tensor, 0, len(_METADATA))]
    if prefix_nbytes:
        expected_copies.append((tensor, len(_METADATA), prefix_nbytes))
    bounce_arena.copy.assert_called_once_with(lease, expected_copies)
    bounce_arena.release.assert_called_once_with(lease)

    if direct_address is not None:
        client._registration_transfer.acquire_registration_ranges.assert_called_once_with(
            (
                _RegistrationRangePlan(
                    address=direct_address,
                    nbytes=direct_nbytes,
                    owners=(tensor,),
                ),
            )
        )
        client._registration_transfer.release_registration_ranges.assert_called_once_with([direct_address])
    else:
        client._registration_transfer.acquire_registration_ranges.assert_not_called()
    store.register_buffer.assert_not_called()


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
    client._registration_transfer.release_registration_ranges.assert_called_once_with([_DIRECT_ADDRESS])
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


def test_prepared_put_uses_exact_joint_wave_addresses_without_registration():
    client, store, _, _, tensor = _make_client()
    owners = [MagicMock(), tensor]

    client.put_prepared_tensor(
        "key",
        [_HEADER_ADDRESS, _BOUNCE_IO_ADDRESS, _DIRECT_ADDRESS],
        [len(_METADATA), 32, 96],
        owners,
    )

    store.batch_put_from_multi_buffers.assert_called_once_with(
        ["key"],
        [[_HEADER_ADDRESS, _BOUNCE_IO_ADDRESS, _DIRECT_ADDRESS]],
        [[len(_METADATA), 32, 96]],
        "replicate",
    )
    client._registration_transfer.acquire_registration_ranges.assert_not_called()
    client._registration_transfer.release_registration_ranges.assert_not_called()


def test_prepared_put_unknown_completion_poisons_all_wave_owners():
    client, store, bounce_arena, _, tensor = _make_client()
    lease = MagicMock()
    store.batch_put_from_multi_buffers.return_value = [-1]

    with pytest.raises(EmbeddingStoreError, match="completion is unconfirmed"):
        client.put_prepared_tensor(
            "key",
            [1],
            [128],
            [bounce_arena, lease, tensor],
        )

    assert client._poisoned is True
    assert bounce_arena in client._unsafe_owners
    assert lease in client._unsafe_owners
    assert tensor in client._unsafe_owners


def test_registration_failure_releases_bounce_lease():
    client, store, bounce_arena, _, tensor = _make_client()
    lease = bounce_arena.acquire.return_value
    client._registration_transfer.acquire_registration_ranges.side_effect = RuntimeError("registration failed")

    with pytest.raises(
        store_client_module.EmbeddingStoreOperationError,
        match="Failed to register",
    ):
        _put(client, tensor, _make_plan(tensor, 32, _DIRECT_ADDRESS, 96))

    assert client._poisoned is False
    bounce_arena.release.assert_called_once_with(lease)


def test_non_fabric_put_uses_no_persistent_header_registration():
    client, store, bounce_arena, _, tensor = _make_client()
    plan = _make_plan(tensor, 0, _DIRECT_ADDRESS, 128)

    _put(client, tensor, plan)
    _put(client, tensor, plan)

    assert client._put_header_tensor is None
    assert client._registration_transfer.acquire_registration_ranges.call_count == 2
    assert bounce_arena.acquire.call_count == 2


def test_fabric_put_uses_npu_header_without_application_registration():
    client, store, _, _, tensor = _make_client(use_fabric_mem=True)
    plan = _make_plan(tensor, 0, _DIRECT_ADDRESS, 128)

    _, synchronize, allocate_header, header, _ = _put(client, tensor, plan)

    allocate_header.assert_called_once_with(
        len(_METADATA),
        dtype=store_client_module.torch.uint8,
        device=tensor.device,
    )
    header.copy_.assert_called_once()
    assert synchronize.call_args_list == [call(tensor.device), call(tensor.device)]
    store.register_buffer.assert_not_called()
    client._registration_transfer.acquire_registration_ranges.assert_not_called()
    client._registration_transfer.release_registration_ranges.assert_not_called()


def test_fabric_get_uses_npu_read_buffer_without_registration():
    client, store, _, _, _ = _make_client(use_fabric_mem=True)
    device = SimpleNamespace(type="npu")
    buffer = MagicMock(nbytes=client._read_buffer_bytes)
    buffer.data_ptr.return_value = _HEADER_ADDRESS
    store.batch_get_into.return_value = [0]
    spec = MagicMock(nbytes=128)

    with (
        patch.object(store_client_module.torch.npu, "set_device") as set_device,
        patch.object(store_client_module.torch, "empty", return_value=buffer) as allocate,
    ):
        loaded = client._load_chunk({"key": spec}, device)

    assert loaded == {}
    set_device.assert_called_once_with(device)
    allocate.assert_called_once_with(
        client._read_buffer_bytes,
        dtype=store_client_module.torch.uint8,
        device=device,
    )
    store.register_buffer.assert_not_called()
    store.batch_get_into.assert_called_once()


def test_shared_roce_get_batches_objects_into_final_tensors():
    client, store, bounce_arena, bounce, _ = _make_client()
    device = SimpleNamespace(type="npu")
    specs = {
        "first": SimpleNamespace(
            shape=(4, 16),
            dtype="torch.bfloat16",
            nbytes=128,
        ),
        "second": SimpleNamespace(
            shape=(4, 16),
            dtype="torch.bfloat16",
            nbytes=128,
        ),
    }
    first = MagicMock()
    second = MagicMock()
    plans = [
        _make_plan(first, 32, _DIRECT_ADDRESS, 96),
        _make_plan(second, 32, _DIRECT_ADDRESS * 2, 96),
    ]
    lease = SimpleNamespace(offset=256)
    bounce_arena.acquire.return_value = lease
    header_size = store_client_module._MOONCAKE_TENSOR_HEADER.size
    store.batch_get_into_multi_buffers.return_value = [
        header_size + 128,
        header_size + 128,
    ]
    header_views = [MagicMock(), MagicMock()]
    for header in header_views:
        header.cpu.return_value.tolist.return_value = list(range(header_size))
    bounce.narrow.side_effect = [
        header_views[0],
        MagicMock(),
        header_views[1],
        MagicMock(),
    ]

    with (
        patch.object(store_client_module.torch.npu, "set_device") as set_device,
        patch.object(store_client_module.torch.npu, "synchronize") as synchronize,
        patch.object(
            store_client_module.torch,
            "empty",
            side_effect=[first, second],
        ),
        patch.object(
            store_client_module,
            "_plan_source",
            side_effect=plans,
        ),
        patch.object(
            store_client_module,
            "_validate_mooncake_tensor_metadata",
        ) as validate,
    ):
        loaded = client._load_chunk(specs, device)

    assert loaded == {"first": first, "second": second}
    set_device.assert_called_once_with(device)
    synchronize.assert_called_once_with(device)
    store.batch_get_into_multi_buffers.assert_called_once_with(
        ["first", "second"],
        [
            [
                _BOUNCE_ADDRESS + lease.offset,
                _BOUNCE_ADDRESS + lease.offset + header_size,
                _DIRECT_ADDRESS,
            ],
            [
                _BOUNCE_ADDRESS + lease.offset + header_size + 32,
                _BOUNCE_ADDRESS + lease.offset + 2 * header_size + 32,
                _DIRECT_ADDRESS * 2,
            ],
        ],
        [[header_size, 32, 96], [header_size, 32, 96]],
    )
    assert validate.call_count == 2
    client._registration_transfer.acquire_registration_ranges.assert_called_once()
    client._registration_transfer.release_registration_ranges.assert_called_once_with(
        [_DIRECT_ADDRESS, _DIRECT_ADDRESS * 2]
    )
    store.register_buffer.assert_not_called()
    bounce_arena.acquire.assert_called_once_with(2 * (header_size + 32))
    bounce_arena.release.assert_called_once_with(lease)
    store.batch_get_into.assert_not_called()
    assert client._read_buffer is None


def test_unconfirmed_non_fabric_get_retains_targets_and_bounce_lease():
    client, store, bounce_arena, _, target = _make_client()
    device = SimpleNamespace(type="npu")
    spec = SimpleNamespace(
        shape=(4, 16),
        dtype="torch.bfloat16",
        nbytes=128,
    )
    plan = _make_plan(target, 32, _DIRECT_ADDRESS, 96)
    lease = SimpleNamespace(offset=256)
    bounce_arena.acquire.return_value = lease
    store.batch_get_into_multi_buffers.return_value = [-1]

    with (
        patch.object(store_client_module.torch.npu, "set_device"),
        patch.object(store_client_module.torch, "empty", return_value=target),
        patch.object(store_client_module, "_plan_source", return_value=plan),
        pytest.raises(EmbeddingStoreError, match="completion is unconfirmed"),
    ):
        client._load_chunk({"key": spec}, device)

    assert client._poisoned is True
    assert target in client._unsafe_owners
    assert lease in client._unsafe_owners
    client._registration_transfer.release_registration_ranges.assert_not_called()
    bounce_arena.release.assert_not_called()


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


@pytest.mark.parametrize("value", ["not-json", "[]"])
def test_independent_store_requires_global_config_json_object(value):
    with (
        patch.dict(
            store_client_module.os.environ,
            {
                "ASCEND_ENABLE_USE_FABRIC_MEM": "0",
                "ASCEND_GLOBAL_RESOURCE_CONFIG": value,
            },
            clear=True,
        ),
        pytest.raises(ValueError, match="must be a JSON object"),
    ):
        store_client_module.use_store_independent_te()


def test_shared_roce_engine_enables_roce_before_engine_initialization():
    with patch.dict(
        store_client_module.os.environ,
        {
            "ASCEND_ENABLE_USE_FABRIC_MEM": "0",
            "HCCL_INTRA_PCIE_ENABLE": "1",
            "HCCL_INTRA_ROCE_ENABLE": "0",
        },
        clear=True,
    ):
        store_client_module._configure_shared_roce()
        assert "ASCEND_ENABLE_USE_FABRIC_MEM" not in store_client_module.os.environ
        assert store_client_module.os.environ["HCCL_INTRA_PCIE_ENABLE"] == "0"
        assert store_client_module.os.environ["HCCL_INTRA_ROCE_ENABLE"] == "1"


def test_shared_roce_engine_rejects_internal_buffer_pool():
    with (
        patch.dict(
            store_client_module.os.environ,
            {
                "HCCL_INTRA_ROCE_ENABLE": "1",
                "ASCEND_BUFFER_POOL": "4:8",
            },
            clear=True,
        ),
        pytest.raises(ValueError, match="ASCEND_BUFFER_POOL to be unset"),
    ):
        store_client_module._configure_shared_roce()


def test_factory_adopts_ec_engine_for_roce_store():
    store = MagicMock()
    store.setup.return_value = 0
    registration_transfer = MagicMock()
    transfer_engine = registration_transfer._ensure_engine.return_value
    transfer_engine.get_rpc_port.return_value = 12345
    native_engine = transfer_engine.get_engine.return_value
    bounce_arena = MagicMock()
    config = _make_store_config()
    mooncake, mooncake_store = _make_mooncake_modules(store)

    with (
        patch.dict(
            store_client_module.os.environ,
            {"ASCEND_ENABLE_USE_FABRIC_MEM": "0"},
        ),
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
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.rdma_utils.get_requester_local_hostname",
            return_value="local-host",
        ),
        patch.object(
            store_client_module,
            "_configure_shared_roce",
            return_value=True,
        ),
    ):
        client = store_client_module.create_ascend_mooncake_embedding_store_client(
            bounce_arena,
            registration_transfer,
        )

    registration_transfer._ensure_engine.assert_called_once_with()
    store.setup.assert_called_once_with(
        "local-host:12345",
        "P2PHANDSHAKE",
        1024,
        512,
        "ascend",
        "",
        "127.0.0.1:50051",
        engine=native_engine,
    )
    store.register_buffer.assert_not_called()
    assert client._bounce_arena is bounce_arena
    assert client._registration_transfer is registration_transfer
    assert client._use_fabric_mem is False


@pytest.mark.parametrize(
    ("environment", "use_fabric_mem", "use_independent_te"),
    [
        (
            {"ASCEND_ENABLE_USE_FABRIC_MEM": "1"},
            True,
            False,
        ),
        (
            {
                "ASCEND_ENABLE_USE_FABRIC_MEM": "0",
                "ASCEND_GLOBAL_RESOURCE_CONFIG": "{}",
            },
            False,
            True,
        ),
    ],
    ids=["fabric", "global-resource-config"],
)
def test_factory_lets_store_create_owned_engine(
    environment,
    use_fabric_mem,
    use_independent_te,
):
    store = MagicMock()
    store.setup.return_value = 0
    registration_transfer = MagicMock()
    config = _make_store_config()
    mooncake, mooncake_store = _make_mooncake_modules(store)

    with (
        patch.dict(
            store_client_module.os.environ,
            environment,
            clear=True,
        ),
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
    ):
        client = store_client_module.create_ascend_mooncake_embedding_store_client(
            MagicMock(),
            registration_transfer,
        )

    registration_transfer._ensure_engine.assert_not_called()
    store.setup.assert_called_once_with(
        "127.0.0.1",
        "P2PHANDSHAKE",
        1024,
        0,
        "ascend",
        "",
        "127.0.0.1:50051",
    )
    assert client._registration_transfer is registration_transfer
    assert client._use_fabric_mem is use_fabric_mem
    assert client.uses_store_independent_te is use_independent_te
