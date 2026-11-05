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


def test_legacy_import_surface():
    """The connector modules keep re-exporting the shared transport helpers."""
    for name in PULL_TRANSPORT_SYMBOLS:
        assert getattr(pull_connector, name) is getattr(transport, name)
    for name in LAYERWISE_SHARED_SYMBOLS:
        assert getattr(layerwise_connector, name) is getattr(transport, name)


def test_layerwise_helpers_kept_distinct():
    """The layerwise-specific helpers keep their own implementation and alias."""
    assert layerwise_connector.group_contiguous_ids is layerwise_connector.group_concurrent_contiguous
    assert layerwise_connector.group_contiguous_ids is not transport.group_concurrent_contiguous
    assert layerwise_connector.ensure_zmq_recv_with_poller is layerwise_connector.ensure_zmq_recv
    assert layerwise_connector.ensure_zmq_recv_with_poller is not transport.ensure_zmq_recv


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
        transport.zmq_ctx(zmq.DEALER, "tcp://localhost:1234") as sock,
    ):
        assert sock is mock_socket
    mock_make_socket.assert_called_once()


def test_ensure_zmq_recv_retries_with_mock_socket():
    mock_socket = MagicMock()
    mock_socket.recv.side_effect = zmq.ZMQError("Receive timeout")
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
