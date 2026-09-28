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
    client = AscendMooncakeEmbeddingStoreClient(
        store,
        bounce_arena,
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
    with (
        patch.object(
            store_client_module,
            "_encode_mooncake_tensor_metadata",
            return_value=_METADATA,
        ),
        patch.object(store_client_module, "_plan_source", return_value=plan),
    ):
        client.put_tensor("key", tensor)


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

    _put(client, tensor, plan)

    header_address = store.register_buffer.call_args_list[0].args[0]
    store.batch_put_from_multi_buffers.assert_called_once_with(
        ["key"],
        [[header_address, *expected_addresses]],
        [[len(_METADATA), *expected_sizes]],
        "replicate",
    )

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
        assert call(direct_address, direct_nbytes) in store.register_buffer.call_args_list
        store.unregister_buffer.assert_called_once_with(direct_address)


def test_store_registers_shared_bounce_arena_only_once():
    client, store, _, bounce, _ = _make_client()

    assert client._ensure_bounce_registered() is bounce
    assert client._ensure_bounce_registered() is bounce

    store.register_buffer.assert_called_once_with(
        _BOUNCE_ADDRESS,
        bounce.nbytes,
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
    store.unregister_buffer.assert_called_once_with(_DIRECT_ADDRESS)
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
    store.unregister_buffer.assert_not_called()
    bounce_arena.release.assert_not_called()


@pytest.mark.parametrize(
    ("plan_args", "register_results", "lease_released"),
    [
        ((128, None, 0), [0, -1], False),
        ((32, _DIRECT_ADDRESS, 96), [0, 0, -1], True),
    ],
    ids=["bounce-registration", "direct-registration"],
)
def test_registration_failure_releases_only_acquired_lease(
    plan_args,
    register_results,
    lease_released,
):
    client, store, bounce_arena, _, tensor = _make_client()
    lease = bounce_arena.acquire.return_value
    store.register_buffer.side_effect = register_results

    with pytest.raises(
        store_client_module.EmbeddingStoreOperationError,
        match="Failed to register",
    ):
        _put(client, tensor, _make_plan(tensor, *plan_args))

    assert client._poisoned is False
    assert bounce_arena.release.call_args_list == ([call(lease)] if lease_released else [])


def test_close_unregisters_bounce_before_closing_store():
    client, store, _, bounce, _ = _make_client()
    client._store_bounce_tensor = bounce
    store.close.return_value = 0

    client.close()

    assert store.mock_calls == [
        call.unregister_buffer(_BOUNCE_ADDRESS),
        call.close(),
    ]
    assert client._store_bounce_tensor is None


def test_factory_rejects_non_ascend_store_protocol():
    bounce_arena = MagicMock()

    with (
        patch.object(
            store_client_module.MooncakeStoreConfig,
            "load_from_config",
            return_value=MagicMock(protocol="rdma"),
        ),
        patch.object(
            store_client_module,
            "create_mooncake_embedding_store_client",
        ) as create_upstream_client,
        pytest.raises(ValueError, match="protocol='ascend'"),
    ):
        store_client_module.create_ascend_mooncake_embedding_store_client(bounce_arena)

    create_upstream_client.assert_not_called()


def test_factory_delegates_store_engine_creation_to_upstream():
    bounce_arena = MagicMock()
    upstream_client = MagicMock()

    with (
        patch.object(
            store_client_module.MooncakeStoreConfig,
            "load_from_config",
            return_value=MagicMock(protocol="ascend"),
        ),
        patch.object(
            store_client_module,
            "create_mooncake_embedding_store_client",
            return_value=upstream_client,
        ) as create_upstream_client,
    ):
        client = store_client_module.create_ascend_mooncake_embedding_store_client(
            bounce_arena,
            read_buffer_bytes=256,
        )

    create_upstream_client.assert_called_once_with(read_buffer_bytes=256)
    assert client.store is upstream_client.store
    assert client.replicate_config is upstream_client.replicate_config
    assert client._bounce_arena is bounce_arena
    assert client._read_buffer_bytes == 256
