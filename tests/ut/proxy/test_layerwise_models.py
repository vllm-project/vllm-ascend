# SPDX-License-Identifier: Apache-2.0
"""Exercise the layerwise proxy's model listing through its HTTP route."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from examples.disaggregated_prefill_v1 import load_balance_proxy_layerwise_server_example as proxy


def request_models(monkeypatch, handler, roles=("prefiller", "decoder"), headers=None):
    async def run():
        async with httpx.AsyncClient(
            base_url="http://backend/v1", transport=httpx.MockTransport(handler), timeout=None
        ) as backend_client:
            backend = SimpleNamespace(client=backend_client, url="http://backend/v1")
            state = SimpleNamespace(
                prefillers=[backend] if "prefiller" in roles else [],
                decoders=[backend] if "decoder" in roles else [],
            )
            monkeypatch.setattr(proxy, "proxy_state", state)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy"
            ) as client:
                return await client.get("/v1/models", headers=headers)

    return asyncio.run(run())


@pytest.mark.parametrize("roles", [("prefiller",), ("decoder",)])
@pytest.mark.parametrize("authorization", [None, "Bearer client-key"])
def test_models_forwarding(monkeypatch, roles, authorization):
    payload = {"object": "list", "data": [{"id": "test-model", "object": "model"}]}
    calls = []

    def backend(request):
        calls.append(request)
        assert str(request.url) == "http://backend/v1/models"
        assert request.headers.get("Authorization") == authorization
        assert all(value is not None and 0 < value <= 3.0 for value in request.extensions["timeout"].values())
        return httpx.Response(200, json=payload)

    headers = {"Authorization": authorization} if authorization else {}
    response = request_models(monkeypatch, backend, roles, headers)
    assert response.status_code == 200
    assert response.json() == payload
    assert len(calls) == 1


def test_models_without_backend(monkeypatch):
    def unexpected_request(request):
        pytest.fail("No backend request should be sent")

    response = request_models(monkeypatch, unexpected_request, roles=())
    assert response.status_code == 503


@pytest.mark.parametrize("backend_status,expected_status", [(401, 401), (403, 403), (404, 502), (500, 502)])
def test_models_backend_status(monkeypatch, backend_status, expected_status):
    response = request_models(monkeypatch, lambda request: httpx.Response(backend_status, text="private error"))
    assert response.status_code == expected_status
    assert "private error" not in response.text


@pytest.mark.parametrize(
    "error,expected_status",
    [(httpx.ConnectError, 502), (httpx.ReadTimeout, 504), (httpx.ConnectTimeout, 504)],
)
def test_models_transport_error(monkeypatch, error, expected_status):
    def backend(request):
        raise error("private connection details", request=request)

    response = request_models(monkeypatch, backend)
    assert response.status_code == expected_status
    assert "private connection details" not in response.text


def test_models_invalid_json(monkeypatch):
    response = request_models(monkeypatch, lambda request: httpx.Response(200, text="not json"))
    assert response.status_code == 502


def test_models_prefers_prefiller(monkeypatch):
    async def run():
        def backend(request):
            return httpx.Response(200, json={"object": "list", "data": [{"id": request.url.host}]})

        async with (
            httpx.AsyncClient(base_url="http://prefiller/v1", transport=httpx.MockTransport(backend)) as prefill,
            httpx.AsyncClient(base_url="http://decoder/v1", transport=httpx.MockTransport(backend)) as decode,
        ):
            monkeypatch.setattr(
                proxy,
                "proxy_state",
                SimpleNamespace(
                    prefillers=[SimpleNamespace(client=prefill, url="http://prefiller/v1")],
                    decoders=[SimpleNamespace(client=decode, url="http://decoder/v1")],
                ),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy"
            ) as client:
                response = await client.get("/v1/models")
                assert response.status_code == 200
                assert response.json()["data"] == [{"id": "prefiller"}]

    asyncio.run(run())
