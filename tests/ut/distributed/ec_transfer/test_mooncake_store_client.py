import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import EmbeddingStoreError

from tests.ut.distributed.ec_transfer.test_mooncake_memory import _make_pool
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import store_client as module
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_client import AscendMooncakeEmbeddingStoreClient


def _client(capacity=1024, **kwargs):
    pool = _make_pool(capacity)
    store = MagicMock()
    store.register_buffer.return_value = 0
    client = AscendMooncakeEmbeddingStoreClient(store, pool, MagicMock(), **kwargs)
    client.bind_staging_buffer(pool.tensor)
    return client, pool, store


def _get(client, objects):
    spec = SimpleNamespace(shape=(2, 4), dtype="torch.float32", nbytes=32)
    source = torch.arange(8, dtype=torch.float32).view(2, 4)
    metadata = module._encode_mooncake_tensor_metadata(source)
    blob = metadata + source.numpy().tobytes()

    def receive(keys, addresses, sizes):
        for address in addresses:
            ctypes.memmove(address, blob, len(blob))
        return [len(blob)] * len(keys)

    client.store.batch_get_into.side_effect = receive
    empty = torch.empty
    with patch.object(module.torch, "empty", side_effect=lambda *a, **kw: empty(*a, **{**kw, "device": "cpu"})):
        loaded = client._load_chunk(dict.fromkeys(objects, spec), SimpleNamespace(type="npu"))
    return loaded, source


@pytest.mark.parametrize("kwargs", [{}, {"use_store_independent_te": True}, {"use_fabric_mem": True}])
def test_get_uses_pool_and_next_p2p_reuses_completed_bytes(kwargs):
    client, pool, store = _client(**kwargs)
    loaded, source = _get(client, ["key"])
    assert torch.equal(loaded["key"], source)
    address = store.batch_get_into.call_args.args[1][0]
    assert pool.tensor.data_ptr() <= address < pool.tensor.data_ptr() + pool.capacity
    cached_count = len(pool._loaded)
    assert cached_count == 1
    pool._copy_staged.reset_mock()
    staged = pool.stage_wait([loaded["key"]])
    assert staged.tensors[0].data_ptr() == address + module._MOONCAKE_TENSOR_HEADER.size
    pool._copy_staged.assert_called_once_with([], [])
    pool.release(staged)
    assert pool._allocator._free == [(0, pool.capacity)]
    store.unregister_buffer.assert_not_called()
    client._memory.transfer.acquire_registration_ranges.assert_not_called()


def test_get_when_pool_busy_does_not_block_or_submit_io():
    client, pool, store = _client(256)
    held = pool.reserve_bytes([256], torch.device("cpu"))
    loaded, _ = _get(client, ["key"])
    assert loaded == {}
    store.batch_get_into.assert_not_called()
    pool.release(held)


@pytest.mark.parametrize("result", [[-704], [0]])
def test_safe_get_miss_releases_entire_lease(result):
    client, pool, store = _client(256)
    store.batch_get_into.return_value = result
    loaded = client._load_chunk({"key": SimpleNamespace(nbytes=32)}, SimpleNamespace(type="npu"))
    assert loaded == {}
    assert pool._allocator._free == [(0, 256)]
    assert not client._poisoned


@pytest.mark.parametrize("result", [[-1], None, [100000]])
def test_unknown_get_completion_keeps_lease_and_poisons_pool(result):
    client, pool, store = _client(256)
    store.batch_get_into.return_value = result
    with pytest.raises(EmbeddingStoreError, match="unconfirmed"):
        client._load_chunk({"key": SimpleNamespace(nbytes=32)}, SimpleNamespace(type="npu"))
    assert client._poisoned
    assert pool._allocator._free == []
    with pytest.raises(RuntimeError, match="poisoned"):
        pool.reserve_bytes([1], torch.device("cpu"))


def test_corrupt_get_header_releases_without_poison():
    client, pool, store = _client(256)
    with patch.object(
        module, "_validate_mooncake_tensor_metadata", side_effect=module.EmbeddingStoreOperationError("bad header")
    ):
        loaded, _ = _get(client, ["key"])
    assert loaded == {}
    assert not client._poisoned
    assert pool._allocator._free == [(0, 256)]


@pytest.mark.parametrize("result", [0, -705, -1])
def test_prepared_put_only_uses_fixed_pool_and_retains_unknown_completion(result):
    client, pool, store = _client(256)
    lease = pool.reserve_bytes([64], torch.device("cpu"))
    address = lease.tensors[0].data_ptr()
    store.batch_put_from_multi_buffers.return_value = [result]
    if result == 0:
        client.put_prepared_tensor("key", [address, address + 24], [24, 32], [lease])
    else:
        with pytest.raises(EmbeddingStoreError):
            client.put_prepared_tensor("key", [address, address + 24], [24, 32], [lease])
    assert client._poisoned is (result == -1)
    if result != -1:
        pool.release(lease)
        assert pool._allocator._free == [(0, 256)]
    store.unregister_buffer.assert_not_called()


def test_close_keeps_bound_store_and_pool_registered():
    client, pool, store = _client(use_store_independent_te=True)
    retained = len(module._PROCESS_LIFETIME_STORE_OWNERS)
    try:
        client.close()
        client.close()
        store.close.assert_not_called()
        store.unregister_buffer.assert_not_called()
        assert len(module._PROCESS_LIFETIME_STORE_OWNERS) == retained + 1
    finally:
        del module._PROCESS_LIFETIME_STORE_OWNERS[retained:]


def test_get_copy_sync_failure_retains_lease_and_target():
    client, pool, _ = _client(256)
    with (
        patch.object(
            module.torch.npu.current_stream.return_value, "synchronize", side_effect=RuntimeError("copy pending")
        ),
        pytest.raises(EmbeddingStoreError, match="unconfirmed"),
    ):
        _get(client, ["key"])
    assert client._poisoned
    assert pool._allocator._free == []
    assert client._unsafe_owners
