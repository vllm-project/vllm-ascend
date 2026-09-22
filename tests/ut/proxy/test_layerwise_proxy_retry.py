import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from examples.disaggregated_prefill_v1 import (
    load_balance_proxy_layerwise_server_example as proxy,
)


class TestLayerwiseProxyRetry(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.endpoint = "/chat/completions"
        self.request_id = "request-0"
        self.request = httpx.Request("POST", f"http://prefill{self.endpoint}")
        self.proxy_state = SimpleNamespace(
            acquire_aborted_prefiller_requests=MagicMock(),
            req_id_future={},
        )

    async def _send(self, client):
        with patch.object(proxy, "proxy_state", self.proxy_state):
            await proxy.send_request_to_service(
                client,
                prefiller_id=0,
                endpoint=self.endpoint,
                req_data={"model": "test"},
                request_id=self.request_id,
                max_retries=3,
                base_delay=0,
            )

    async def test_retries_connect_errors(self):
        client = SimpleNamespace(
            post=AsyncMock(
                side_effect=[
                    httpx.ConnectError("connection refused", request=self.request),
                    httpx.Response(200, request=self.request),
                ]
            )
        )

        with patch.object(proxy.asyncio, "sleep", new=AsyncMock()) as sleep:
            await self._send(client)

        self.assertEqual(client.post.await_count, 2)
        sleep.assert_awaited_once_with(0)

    async def test_retries_pool_timeouts(self):
        client = SimpleNamespace(
            post=AsyncMock(
                side_effect=[
                    httpx.PoolTimeout("connection pool exhausted", request=self.request),
                    httpx.Response(200, request=self.request),
                ]
            )
        )

        with patch.object(proxy.asyncio, "sleep", new=AsyncMock()) as sleep:
            await self._send(client)

        self.assertEqual(client.post.await_count, 2)
        sleep.assert_awaited_once_with(0)

    async def test_does_not_retry_http_errors(self):
        client = SimpleNamespace(
            post=AsyncMock(
                return_value=httpx.Response(
                    500,
                    request=self.request,
                )
            )
        )

        with self.assertRaises(httpx.HTTPStatusError):
            await self._send(client)

        client.post.assert_awaited_once()

    async def test_does_not_retry_transport_errors_after_connect(self):
        client = SimpleNamespace(
            post=AsyncMock(
                side_effect=httpx.ReadTimeout(
                    "read timed out",
                    request=self.request,
                )
            )
        )

        with self.assertRaises(httpx.ReadTimeout):
            await self._send(client)

        client.post.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
