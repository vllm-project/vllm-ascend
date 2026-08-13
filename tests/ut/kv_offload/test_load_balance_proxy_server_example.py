import importlib.util
import sys
import unittest
from pathlib import Path

import httpx


MODULE_PATH = (
    Path(__file__).parents[3]
    / "examples"
    / "disaggregated_prefill_v1"
    / "load_balance_proxy_server_example.py"
)
SPEC = importlib.util.spec_from_file_location("load_balance_proxy_server_example", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
proxy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = proxy
SPEC.loader.exec_module(proxy)


class TestBackendRetryPolicy(unittest.IsolatedAsyncioTestCase):
    async def test_prefill_retries_non_400_status(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(503 if attempts == 1 else 200, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await proxy.send_request_to_service(
                client, "/v1/completions", {}, "req", max_retries=2, base_delay=0
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(attempts, 2)

    async def test_prefill_does_not_retry_400(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(400, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(httpx.HTTPStatusError):
                await proxy.send_request_to_service(
                    client, "/v1/completions", {}, "req", max_retries=3, base_delay=0
                )

        self.assertEqual(attempts, 1)

    async def test_decode_retries_non_400_status(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(500 if attempts == 1 else 200, content=b"ok", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            chunks = [
                chunk
                async for chunk in proxy.stream_service_response(
                    client, "/v1/completions", {}, "req", max_retries=2, base_delay=0
                )
            ]

        self.assertEqual(chunks, [b"ok"])
        self.assertEqual(attempts, 2)

    async def test_decode_does_not_retry_400(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(400, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(httpx.HTTPStatusError):
                async for _ in proxy.stream_service_response(
                    client, "/v1/completions", {}, "req", max_retries=3, base_delay=0
                ):
                    pass

        self.assertEqual(attempts, 1)


if __name__ == "__main__":
    unittest.main()
