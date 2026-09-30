# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy


@pytest.mark.parametrize(
    ("response_kwargs", "expected_body"),
    [
        (
            {"json": {"error": {"message": "maximum context length is 32"}}},
            {"error": {"message": "maximum context length is 32"}},
        ),
        (
            {"text": "prefill unavailable"},
            {"error": {"message": "prefill unavailable"}},
        ),
    ],
)
def test_prefill_http_error_is_forwarded(
    monkeypatch,
    response_kwargs,
    expected_body,
):
    upstream_request = httpx.Request("POST", "http://prefill/v1/completions")
    upstream_response = httpx.Response(
        400,
        request=upstream_request,
        **response_kwargs,
    )
    error = httpx.HTTPStatusError(
        "upstream returned HTTP 400",
        request=upstream_request,
        response=upstream_response,
    )
    request = AsyncMock()
    request.json.return_value = {"prompt": "test"}
    request.body.return_value = b'{"prompt":"test"}'

    monkeypatch.setattr(proxy, "runtime", object())
    monkeypatch.setattr(proxy, "global_args", argparse.Namespace())
    assign_instances = AsyncMock(side_effect=error)
    monkeypatch.setattr(proxy, "assign_instances", assign_instances)

    response = asyncio.run(proxy.handle_completions_impl("completions", request))

    assert response.status_code == 400
    assert json.loads(response.body) == expected_body
    assign_instances.assert_awaited_once()
