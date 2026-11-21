"""Tests for the layerwise metaserver client.

The real ``httpx.Client`` is replaced with a mock so no network call is made;
the retry loop is exercised through a mock ``post``.
"""

import types
from unittest.mock import MagicMock, patch

import httpx
import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise import metaserver
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.metaserver import MetaserverClient


def _fake_config(tls_config):
    get_from_extra_config = MagicMock(return_value=tls_config)
    return types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(get_from_extra_config=get_from_extra_config)
    ), get_from_extra_config


def test_client_construction_without_tls():
    config, get_from_extra_config = _fake_config({})
    with patch.object(metaserver.httpx, "Client") as mock_client_cls:
        client = MetaserverClient(config)

    get_from_extra_config.assert_called_once_with("tls_config", {})
    kwargs = mock_client_cls.call_args.kwargs
    assert kwargs["limits"] == httpx.Limits(max_connections=100000)
    assert kwargs["timeout"] is None
    assert "cert" not in kwargs
    assert "verify" not in kwargs
    assert client.cert_path == (None, None, None)
    assert client.ssl_enable is False
    assert client.ca_path is False
    assert client._client is mock_client_cls.return_value
    assert client._closed is False


def test_client_construction_with_tls():
    tls_config = {
        "ssl_keyfile": "key.pem",
        "ssl_certfile": "cert.pem",
        "ssl_ca_certs": "ca.pem",
        "ssl_keyfile_password": "secret",
        "ssl_enable": True,
    }
    config, _ = _fake_config(tls_config)
    with patch.object(metaserver.httpx, "Client") as mock_client_cls:
        client = MetaserverClient(config)

    kwargs = mock_client_cls.call_args.kwargs
    assert kwargs["limits"] == httpx.Limits(max_connections=100000)
    assert kwargs["timeout"] is None
    assert kwargs["cert"] == ("cert.pem", "key.pem", "secret")
    assert kwargs["verify"] == "ca.pem"
    assert client.cert_path == ("cert.pem", "key.pem", "secret")
    assert client.ssl_enable is True
    assert client.ca_path == "ca.pem"


def _make_client(tls_config=None):
    config, _ = _fake_config(tls_config or {})
    with patch.object(metaserver.httpx, "Client") as mock_client_cls:
        client = MetaserverClient(config)
    return client, mock_client_cls.return_value


def test_access_posts_once():
    client, mock_http_client = _make_client()
    client.access("http://metaserver", {"request_id": "req-1"})
    mock_http_client.post.assert_called_once_with("http://metaserver", json={"request_id": "req-1"})


def test_access_retries_three_times_then_raises():
    client, mock_http_client = _make_client()
    errors = [RuntimeError("first"), RuntimeError("second"), RuntimeError("last")]
    mock_http_client.post.side_effect = errors

    with patch.object(metaserver, "logger") as mock_logger, pytest.raises(RuntimeError, match="last"):
        client.access("http://metaserver", {"request_id": "req-1"})

    assert mock_http_client.post.call_count == 3
    assert mock_logger.error.call_count == 3
    mock_logger.error.assert_called_with("Failed to connect to metaserver. url=%s, retry=%d. ", "http://metaserver", 3)


def test_access_retries_then_succeeds():
    client, mock_http_client = _make_client()
    mock_http_client.post.side_effect = [RuntimeError("first"), RuntimeError("second"), None]

    with patch.object(metaserver, "logger") as mock_logger:
        client.access("http://metaserver", {"request_id": "req-1"})

    assert mock_http_client.post.call_count == 3
    assert mock_logger.error.call_count == 2


def test_close_is_idempotent():
    client, mock_http_client = _make_client()
    client.close()
    client.close()
    mock_http_client.close.assert_called_once()
    assert client._closed is True


def test_close_swallows_errors():
    client, mock_http_client = _make_client()
    mock_http_client.close.side_effect = RuntimeError("close failed")

    with patch.object(metaserver, "logger") as mock_logger:
        client.close()
        client.close()

    assert mock_http_client.close.call_count == 1
    assert mock_logger.warning.call_count == 1
    assert client._closed is True
