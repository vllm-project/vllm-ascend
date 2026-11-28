"""Compatibility tests for the shared mooncake v1 transport helpers.

They lock the legacy import surface of the two V1 connector modules, the
deliberate divergence of the layerwise-only helpers, and the behaviour of the
helpers that moved to ``mooncake_v1.common.transport``.
"""

from unittest.mock import MagicMock, patch

import pytest
import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_connector as pull_connector
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_connector as layerwise_connector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common import transport

PULL_TRANSPORT_SYMBOLS = (
    "SizedDict",
    "ensure_zmq_recv",
    "ensure_zmq_send",
    "group_concurrent_contiguous",
    "split_if_not_byte_contiguous",
    "string_to_int64_hash",
    "zmq_ctx",
)
LAYERWISE_SHARED_SYMBOLS = ("SizedDict", "ensure_zmq_send", "string_to_int64_hash", "zmq_ctx")


def _same_definition(left, right):
    """True when both names resolve to the same definition.

    Identity is not asserted here: the kv_offload suite pops and re-imports the
    kv_transfer modules from sys.modules, so a module object captured at import
    time can be a different instance carrying the same definition.
    """
    return (getattr(left, "__module__", None), getattr(left, "__qualname__", None)) == (
        getattr(right, "__module__", None),
        getattr(right, "__qualname__", None),
    )


def test_legacy_import_surface():
    """The connector modules keep re-exporting the shared transport helpers."""
    for name in PULL_TRANSPORT_SYMBOLS:
        assert _same_definition(getattr(pull_connector, name), getattr(transport, name))
    for name in LAYERWISE_SHARED_SYMBOLS:
        assert _same_definition(getattr(layerwise_connector, name), getattr(transport, name))


def test_layerwise_helpers_kept_distinct():
    """The layerwise-specific helpers keep their own implementation and alias."""
    assert layerwise_connector.group_contiguous_ids is layerwise_connector.group_concurrent_contiguous
    assert not _same_definition(layerwise_connector.group_contiguous_ids, transport.group_concurrent_contiguous)
    assert layerwise_connector.ensure_zmq_recv_with_poller is layerwise_connector.ensure_zmq_recv
    assert not _same_definition(layerwise_connector.ensure_zmq_recv_with_poller, transport.ensure_zmq_recv)


def test_layerwise_group_ids_behavior():
    src_groups, dst_groups = layerwise_connector.group_contiguous_ids([1, 2, 3, 5, 6], [10, 11, 12, 14, 15])
    assert src_groups == [[1, 2, 3], [5, 6]]
    assert dst_groups == [[10, 11, 12], [14, 15]]


def test_zmq_ctx_rejects_unknown_type():
    with (
        pytest.raises(ValueError),
        transport.zmq_ctx("INVALID", "tcp://127.0.0.1:5555"),
    ):
        pass


def test_zmq_ctx_allows_dealer():
    """V1 accepts DEALER sockets, which is why its zmq_ctx is not the V2 one."""
    mock_socket = MagicMock()
    with (
        patch.object(transport, "make_zmq_socket", return_value=mock_socket) as mock_make_socket,
        transport.zmq_ctx(zmq.DEALER, "tcp://localhost:1234") as sock,  # type: ignore[attr-defined]  # pyzmq stubs omit DEALER
    ):
        assert sock is mock_socket
    mock_make_socket.assert_called_once()


def test_ensure_zmq_recv_retries_with_mock_socket():
    mock_socket = MagicMock()
    mock_socket.recv.side_effect = zmq.ZMQError("Receive timeout")  # type: ignore[attr-defined]  # pyzmq stubs omit ZMQError
    with (
        patch.object(transport, "logger") as mock_logger,
        patch.object(transport.time, "sleep") as mock_sleep,
        pytest.raises(RuntimeError, match="after 2 retries"),
    ):
        transport.ensure_zmq_recv(mock_socket, "tcp://localhost:1234", max_retries=2)
    assert mock_socket.recv.call_count == 2
    mock_logger.warning.assert_called_once()
    mock_logger.error.assert_called_once()
    mock_sleep.assert_called_once()


def test_socket_pool_returns_and_closes():
    mock_socket = MagicMock()
    mock_context = MagicMock()
    pool = transport.RemoteSocketPool(timeout=1.0)
    remote_path = transport.make_zmq_path("tcp", "host", 1234)
    failing_socket = MagicMock()
    failing_socket.close.side_effect = RuntimeError("close failed")
    with (
        patch.object(transport.zmq, "Context", return_value=mock_context) as mock_context_cls,
        patch.object(transport, "make_zmq_socket", return_value=mock_socket) as mock_make_socket,
    ):
        first_socket = pool.get("host", 1234)
        assert first_socket is mock_socket
        mock_context_cls.assert_called_once()
        mock_make_socket.assert_called_once()
        pool.return_socket(first_socket, "host", 1234)
        assert len(pool.sockets[remote_path]) == 1
        second_socket = pool.get("host", 1234)
        assert second_socket is mock_socket
        mock_make_socket.assert_called_once()
        pool.return_socket(second_socket, "host", 1234)
        # A failing socket must not abort the cleanup nor make close() raise.
        pool.sockets[remote_path].append(failing_socket)
        pool.close()
    mock_socket.close.assert_called_once_with(linger=0)
    mock_context.destroy.assert_called_once_with(linger=0)
    failing_socket.close.assert_called_once_with(linger=0)
    assert not pool.sockets
    pool.close()  # terminal: a second close must not touch the sockets again
    mock_socket.close.assert_called_once_with(linger=0)
    failing_socket.close.assert_called_once_with(linger=0)


def test_socket_pool_get_after_close_raises():
    mock_socket = MagicMock()
    pool = transport.RemoteSocketPool()
    with (
        patch.object(transport.zmq, "Context"),
        patch.object(transport, "make_zmq_socket", return_value=mock_socket),
    ):
        pool.return_socket(pool.get("host", 1234), "host", 1234)
        pool.close()
        with pytest.raises(RuntimeError, match="closed"):
            pool.get("host", 1234)


def test_socket_pool_return_after_close_closes_socket_and_context():
    mock_socket = MagicMock()
    mock_context = MagicMock()
    pool = transport.RemoteSocketPool()
    with (
        patch.object(transport.zmq, "Context", return_value=mock_context),
        patch.object(transport, "make_zmq_socket", return_value=mock_socket),
    ):
        checked_out = pool.get("host", 1234)
        assert pool._contexts[checked_out] is mock_context
        pool.close()
        # close() must not close a socket that is still checked out.
        mock_socket.close.assert_not_called()
        mock_context.destroy.assert_not_called()
        assert pool._contexts[checked_out] is mock_context
        pool.return_socket(checked_out, "host", 1234)
    mock_socket.close.assert_called_once_with(linger=0)
    mock_context.destroy.assert_called_once_with(linger=0)
    assert not pool.sockets
    assert not pool._contexts


def test_socket_pool_discard_releases_checked_out_socket_and_context():
    mock_socket = MagicMock()
    mock_context = MagicMock()
    pool = transport.RemoteSocketPool()
    with (
        patch.object(transport.zmq, "Context", return_value=mock_context),
        patch.object(transport, "make_zmq_socket", return_value=mock_socket),
    ):
        checked_out = pool.get("host", 1234)
    assert pool._contexts[checked_out] is mock_context

    pool.discard(checked_out)

    mock_socket.close.assert_called_once_with(linger=0)
    mock_context.destroy.assert_called_once_with(linger=0)
    assert checked_out not in pool._contexts
    assert not any(queue for queue in pool.sockets.values())

    pool.discard(checked_out)  # best effort: a repeated discard does not raise
    assert mock_context.destroy.call_count == 1
    pool.close()  # terminal close stays safe after a discard


def test_socket_pool_timeout_is_applied():
    mock_socket = MagicMock()
    pool = transport.RemoteSocketPool(timeout=2.5)
    with (
        patch.object(transport.zmq, "Context"),
        patch.object(transport, "make_zmq_socket", return_value=mock_socket),
    ):
        pool.get("host", 1234)
    expected_timeout = int(pool.timeout * 1000)
    mock_socket.setsockopt.assert_any_call(transport.zmq.SNDTIMEO, expected_timeout)
    mock_socket.setsockopt.assert_any_call(transport.zmq.RCVTIMEO, expected_timeout)
