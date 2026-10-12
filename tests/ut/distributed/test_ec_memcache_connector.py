from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from vllm_ascend.distributed.ec_transfer import ec_memcache_connector as connector_module
from vllm_ascend.distributed.ec_transfer.ec_memcache_connector import (
    DEFAULT_RECV_BUFFER_TOKENS,
    MIN_RECV_BUFFER_TOKENS,
    ECMemcacheConnector,
    _get_recv_buffer_tokens,
)


def _make_connector(capacity_tokens: int = 8, hidden_dim: int = 4) -> ECMemcacheConnector:
    connector = ECMemcacheConnector.__new__(ECMemcacheConnector)
    connector._backend = MagicMock()
    connector._hidden_dim = hidden_dim
    connector._dtype = torch.float32
    connector._elem_size = torch.empty(0, dtype=connector._dtype).element_size()
    connector._recv_buffer_tokens = capacity_tokens
    connector._recv_buffer = torch.arange(
        capacity_tokens * hidden_dim,
        dtype=connector._dtype,
    ).reshape(capacity_tokens, hidden_dim)
    connector._recv_buffer_reuse_event = None
    connector._recv_buffer_registered = True
    connector._put_executor = None
    return connector


def _key_info(nbytes: int) -> MagicMock:
    info = MagicMock()
    info.size.return_value = nbytes
    return info


@contextmanager
def _mock_npu_allocations(npu: MagicMock) -> Iterator[None]:
    real_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    with (
        patch.object(connector_module.torch, "empty", side_effect=cpu_empty),
        patch.object(connector_module.torch, "npu", npu, create=True),
    ):
        yield


@pytest.mark.parametrize(
    ("extra_config", "max_num_batched_tokens", "expected"),
    [
        ({}, DEFAULT_RECV_BUFFER_TOKENS, DEFAULT_RECV_BUFFER_TOKENS),
        (
            {"recv_buffer_tokens": MIN_RECV_BUFFER_TOKENS},
            DEFAULT_RECV_BUFFER_TOKENS,
            MIN_RECV_BUFFER_TOKENS,
        ),
        ({"recv_buffer_tokens": 4096}, 4096, 4096),
    ],
)
def test_get_recv_buffer_tokens(extra_config: dict, max_num_batched_tokens: int, expected: int) -> None:
    vllm_config = MagicMock()
    vllm_config.ec_transfer_config.get_from_extra_config.side_effect = lambda key, default: extra_config.get(
        key, default
    )
    vllm_config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens

    assert _get_recv_buffer_tokens(vllm_config) == expected


@pytest.mark.parametrize("value", [True, 8192.0, "8192", None])
def test_get_recv_buffer_tokens_rejects_non_integer_values(value: object) -> None:
    vllm_config = MagicMock()
    vllm_config.ec_transfer_config.get_from_extra_config.return_value = value

    with pytest.raises(ValueError, match="recv_buffer_tokens must be an integer"):
        _get_recv_buffer_tokens(vllm_config)


@pytest.mark.parametrize(
    ("value", "max_num_batched_tokens"),
    [
        (MIN_RECV_BUFFER_TOKENS - 1, DEFAULT_RECV_BUFFER_TOKENS),
        (DEFAULT_RECV_BUFFER_TOKENS + 1, DEFAULT_RECV_BUFFER_TOKENS),
    ],
)
def test_get_recv_buffer_tokens_rejects_out_of_range_values(value: int, max_num_batched_tokens: int) -> None:
    vllm_config = MagicMock()
    vllm_config.ec_transfer_config.get_from_extra_config.return_value = value
    vllm_config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens

    with pytest.raises(ValueError, match="recv_buffer_tokens must be between"):
        _get_recv_buffer_tokens(vllm_config)


def test_get_recv_buffer_tokens_requires_ec_transfer_config() -> None:
    vllm_config = MagicMock()
    vllm_config.ec_transfer_config = None

    with pytest.raises(ValueError, match="requires ec_transfer_config"):
        _get_recv_buffer_tokens(vllm_config)


def test_ec_get_batch_uses_offsets_and_copies_out() -> None:
    connector = _make_connector()
    bytes_per_token = connector._hidden_dim * connector._elem_size
    # Batch_get_key_info returns two values, tokens lengths are 2 and 3.
    connector._backend.batch_get_key_info.return_value = [
        _key_info(2 * bytes_per_token),
        _key_info(3 * bytes_per_token),
    ]
    # Two embeddings successfully returned from memcache.
    connector._backend.batch_get_into_buffers.return_value = [0, 0]
    previous_copy = MagicMock()
    connector._recv_buffer_reuse_event = previous_copy
    copy_done = MagicMock()
    npu = MagicMock()
    npu.Event.return_value = copy_done

    expected_first = connector._recv_buffer[:2].clone()
    expected_second = connector._recv_buffer[2:5].clone()
    base_addr = connector._recv_buffer.data_ptr()

    with _mock_npu_allocations(npu):
        embeddings = connector._ec_get_batch(["key0", "key1"])

    # After two embeddings successfully returned from memcache, assert whether they are equal.
    assert embeddings is not None
    assert torch.equal(embeddings[0], expected_first)
    assert torch.equal(embeddings[1], expected_second)
    connector._recv_buffer.zero_()
    # Clear the buffer of connector, assert whether the copy operation has been executed.
    assert torch.equal(embeddings[0], expected_first)
    assert torch.equal(embeddings[1], expected_second)

    connector._backend.batch_get_key_info.assert_called_once_with(["key0", "key1"])
    connector._backend.batch_get_into_buffers.assert_called_once_with(
        ["key0", "key1"],
        [base_addr, base_addr + 2 * bytes_per_token],
        [2 * bytes_per_token, 3 * bytes_per_token],
        connector_module.MmcDirect.COPY_G2L.value,
    )
    # In method _wait_recv_buffer_reusable, this synchronization operation has certainly been called
    # In this test, capacity_tokens = 8, is larger than 2 + 3 = 5
    previous_copy.synchronize.assert_called_once_with()
    copy_done.record.assert_called_once_with(npu.current_stream.return_value)
    assert connector._recv_buffer_reuse_event is copy_done
    connector._backend.register_buffer.assert_not_called()


def test_ec_get_batch_splits_requests_by_buffer_capacity() -> None:
    connector = _make_connector(capacity_tokens=4)
    bytes_per_token = connector._hidden_dim * connector._elem_size
    connector._backend.batch_get_key_info.return_value = [
        _key_info(2 * bytes_per_token),
        _key_info(3 * bytes_per_token),
        _key_info(bytes_per_token),
    ]
    connector._backend.batch_get_into_buffers.side_effect = [[0], [0, 0]]
    first_copy_done = MagicMock()
    second_copy_done = MagicMock()
    npu = MagicMock()
    npu.Event.side_effect = [first_copy_done, second_copy_done]

    base_addr = connector._recv_buffer.data_ptr()
    with _mock_npu_allocations(npu):
        embeddings = connector._ec_get_batch(["key0", "key1", "key2"])

    assert embeddings is not None
    assert all(embedding is not None for embedding in embeddings)
    assert connector._backend.batch_get_into_buffers.call_args_list == [
        call(
            ["key0"],
            [base_addr],
            [2 * bytes_per_token],
            connector_module.MmcDirect.COPY_G2L.value,
        ),
        call(
            ["key1", "key2"],
            [base_addr, base_addr + 3 * bytes_per_token],
            [3 * bytes_per_token, bytes_per_token],
            connector_module.MmcDirect.COPY_G2L.value,
        ),
    ]
    first_copy_done.synchronize.assert_called_once_with()
    assert connector._recv_buffer_reuse_event is second_copy_done


def test_ec_get_batch_reuses_one_temporary_buffer_for_oversized_values() -> None:
    connector = _make_connector(capacity_tokens=4)
    bytes_per_token = connector._hidden_dim * connector._elem_size
    connector._backend.batch_get_key_info.return_value = [
        _key_info(5 * bytes_per_token),
        _key_info(5 * bytes_per_token),
        _key_info(10 * bytes_per_token),
    ]
    connector._backend.batch_get_into_buffers.side_effect = [[0, 0], [0]]
    first_copy_done = MagicMock()
    second_copy_done = MagicMock()
    lifecycle: list[str] = []
    first_copy_done.synchronize.side_effect = lambda: lifecycle.append("sync0")
    second_copy_done.synchronize.side_effect = lambda: lifecycle.append("sync1")
    connector._backend.unregister_buffer.side_effect = lambda _ptrs, _sizes: lifecycle.append("unregister")
    npu = MagicMock()
    npu.Event.side_effect = [first_copy_done, second_copy_done]

    with _mock_npu_allocations(npu):
        embeddings = connector._ec_get_batch(["key0", "key1", "key2"])

    assert embeddings is not None
    assert [embedding.shape[0] for embedding in embeddings if embedding is not None] == [
        5,
        5,
        10,
    ]
    # Register_buffer can only be called once.
    connector._backend.register_buffer.assert_called_once()
    register_args = connector._backend.register_buffer.call_args.args
    temporary_buffer_ptr = register_args[0][0]
    # Assert that the size of the registered buffer is the size of the largest embedding.
    assert register_args[1] == [10 * bytes_per_token]
    assert connector._backend.batch_get_into_buffers.call_args_list == [
        call(
            ["key0", "key1"],
            [temporary_buffer_ptr, temporary_buffer_ptr + 5 * bytes_per_token],
            [5 * bytes_per_token, 5 * bytes_per_token],
            connector_module.MmcDirect.COPY_G2L.value,
        ),
        call(
            ["key2"],
            [temporary_buffer_ptr],
            [10 * bytes_per_token],
            connector_module.MmcDirect.COPY_G2L.value,
        ),
    ]
    first_copy_done.synchronize.assert_called_once_with()
    second_copy_done.synchronize.assert_called_once_with()
    connector._backend.unregister_buffer.assert_called_once_with([temporary_buffer_ptr], [10 * bytes_per_token])
    assert lifecycle == ["sync0", "sync1", "unregister"]


def test_ec_get_batch_preserves_all_mixed_request_outputs() -> None:
    connector = _make_connector(capacity_tokens=4)
    bytes_per_token = connector._hidden_dim * connector._elem_size
    connector._backend.batch_get_key_info.return_value = [
        _key_info(2 * bytes_per_token),
        _key_info(5 * bytes_per_token),
        _key_info(2 * bytes_per_token),
    ]
    connector._backend.batch_get_into_buffers.side_effect = [[0, 0], [0]]
    regular_copy_done = MagicMock()
    oversized_copy_done = MagicMock()
    npu = MagicMock()
    npu.Event.side_effect = [regular_copy_done, oversized_copy_done]

    with _mock_npu_allocations(npu):
        embeddings = connector._ec_get_batch(["regular0", "oversized", "regular1"])

    assert embeddings is not None
    assert all(embedding is not None for embedding in embeddings)
    assert [embedding.shape[0] for embedding in embeddings if embedding is not None] == [
        2,
        5,
        2,
    ]
    assert connector._backend.batch_get_into_buffers.call_args_list[0].args[0] == [
        "regular0",
        "regular1",
    ]
    assert connector._backend.batch_get_into_buffers.call_args_list[1].args[0] == ["oversized"]
    regular_copy_done.synchronize.assert_called_once_with()
    oversized_copy_done.synchronize.assert_called_once_with()


def test_ec_get_batch_preserves_indices_when_key_metadata_is_skipped() -> None:
    """Keep outputs aligned with the original keys after metadata filtering.

    The first key has an empty value and the second has an unaligned value, so
    neither is submitted to Memcache. The third key is valid: it uses offset
    zero in the compact receive batch, but its output must still be written to
    original index 2 via ``src_idx`` rather than being assigned to index 0.
    """
    connector = _make_connector()
    bytes_per_token = connector._hidden_dim * connector._elem_size
    connector._backend.batch_get_key_info.return_value = [
        _key_info(0),
        _key_info(bytes_per_token + 1),
        _key_info(2 * bytes_per_token),
    ]
    connector._backend.batch_get_into_buffers.return_value = [0]
    copy_done = MagicMock()
    npu = MagicMock()
    npu.Event.return_value = copy_done
    base_addr = connector._recv_buffer.data_ptr()

    with _mock_npu_allocations(npu):
        embeddings = connector._ec_get_batch(["empty", "unaligned", "valid"])

    assert embeddings is not None
    assert embeddings[0] is None
    assert embeddings[1] is None
    assert embeddings[2] is not None
    assert embeddings[2].shape == (2, connector._hidden_dim)
    connector._backend.batch_get_into_buffers.assert_called_once_with(
        ["valid"],
        [base_addr],
        [2 * bytes_per_token],
        connector_module.MmcDirect.COPY_G2L.value,
    )


def test_ec_get_batch_keeps_failed_or_missing_results_as_none() -> None:
    """Preserve per-key results when a Memcache batch is only partly valid.

    Three valid requests receive only two result codes: the first succeeds,
    the second returns a nonzero error, and the third has no corresponding
    result. Only the successful request gets an output; the other original
    positions remain ``None``, while the successful copy-out still records a
    receive-buffer reuse event.
    """
    connector = _make_connector()
    bytes_per_token = connector._hidden_dim * connector._elem_size
    connector._backend.batch_get_key_info.return_value = [
        _key_info(bytes_per_token),
        _key_info(bytes_per_token),
        _key_info(bytes_per_token),
    ]
    connector._backend.batch_get_into_buffers.return_value = [0, 7]
    copy_done = MagicMock()
    npu = MagicMock()
    npu.Event.return_value = copy_done

    with _mock_npu_allocations(npu):
        embeddings = connector._ec_get_batch(["success", "failed", "missing"])

    assert embeddings is not None
    assert embeddings[0] is not None
    assert embeddings[1] is None
    assert embeddings[2] is None
    copy_done.record.assert_called_once_with(npu.current_stream.return_value)
    assert connector._recv_buffer_reuse_event is copy_done


def test_ec_get_batch_rejects_mismatched_key_info_count() -> None:
    connector = _make_connector()
    connector._backend.batch_get_key_info.return_value = [_key_info(16)]

    assert connector._ec_get_batch(["key0", "key1"]) is None
    connector._backend.batch_get_into_buffers.assert_not_called()


def test_shutdown_waits_unregisters_and_releases_receive_buffer() -> None:
    connector = _make_connector()
    copy_done = MagicMock()
    connector._recv_buffer_reuse_event = copy_done
    recv_buffer_ptr = connector._recv_buffer.data_ptr()
    recv_buffer_nbytes = connector._recv_buffer.nbytes

    connector.shutdown()

    copy_done.synchronize.assert_called_once_with()
    connector._backend.unregister_buffer.assert_called_once_with([recv_buffer_ptr], [recv_buffer_nbytes])
    assert connector._recv_buffer_registered is False
    assert connector._recv_buffer is None

    connector.shutdown()
    connector._backend.unregister_buffer.assert_called_once_with([recv_buffer_ptr], [recv_buffer_nbytes])
