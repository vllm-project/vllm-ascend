# SPDX-License-Identifier: Apache-2.0

import pytest

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, 0),
        ({"usage": {}}, 0),
        ({"usage": {"prompt_tokens_details": {}}}, 0),
        ({"usage": {"prompt_tokens_details": {"cached_tokens": 0}}}, 0),
        ({"usage": {"prompt_tokens_details": {"cached_tokens": 32}}}, 32),
    ],
)
def test_extract_cached_tokens_cold_and_warm(payload, expected):
    assert proxy.extract_cached_tokens(payload) == expected


def test_update_cached_tokens_in_nonstream_response():
    payload = {
        "choices": [{"text": "ok"}],
        "usage": {"prompt_tokens": 33, "prompt_tokens_details": {}},
    }

    assert proxy.update_cached_tokens_in_chunk(payload, 0)
    assert payload["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
