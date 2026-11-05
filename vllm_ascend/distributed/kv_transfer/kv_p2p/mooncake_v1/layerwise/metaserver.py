# SPDX-License-Identifier: Apache-2.0
"""HTTP client for the layerwise connector metaserver.

The client bundles the TLS/plain ``httpx.Client`` construction and the retry
loop that used to live in ``MooncakeLayerwiseConnectorScheduler`` so the
scheduler only composes it. ``close()`` is implemented (idempotent, best
effort) but is intentionally not wired into the connector yet: shutdown wiring
is handled by the lifecycle node.

This module must not import ``vllm_ascend`` or the old V1 connector modules.
"""

from typing import TYPE_CHECKING, Any

import httpx
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class MetaserverClient:
    """Metaserver HTTP client with the connector's TLS and retry behaviour."""

    def __init__(self, vllm_config: "VllmConfig") -> None:
        tls_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("tls_config", {})
        ssl_keyfile = tls_config.get("ssl_keyfile")
        ssl_certfile = tls_config.get("ssl_certfile")
        ssl_ca_certs = tls_config.get("ssl_ca_certs", False)
        ssl_keyfile_password = tls_config.get("ssl_keyfile_password")
        self.cert_path = (ssl_certfile, ssl_keyfile, ssl_keyfile_password)
        self.ssl_enable = tls_config.get("ssl_enable", False)
        self.ca_path = ssl_ca_certs
        if self.ssl_enable:
            self._client = httpx.Client(
                limits=httpx.Limits(max_connections=100000), timeout=None, cert=self.cert_path, verify=self.ca_path
            )
        else:
            self._client = httpx.Client(limits=httpx.Limits(max_connections=100000), timeout=None)
        self._closed = False

    def access(self, url, message) -> None:
        success = False
        retry = 0
        while retry < 3 and success is False:
            retry += 1
            try:
                self._client.post(url, json=message)
                success = True
            except Exception as e:
                logger.error("Failed to connect to metaserver. url=%s, retry=%d. ", url, retry)
                if retry == 3:
                    raise e

    def close(self) -> None:
        """Close the underlying client once; idempotent and best effort."""
        if self._closed:
            return
        self._closed = True
        try:
            self._client.close()
        except Exception as e:
            logger.warning("Failed to close metaserver client. error=%s. ", e)


__all__ = ["MetaserverClient"]
