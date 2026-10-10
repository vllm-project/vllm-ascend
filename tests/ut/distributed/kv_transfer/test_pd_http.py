# SPDX-License-Identifier: Apache-2.0
"""Legacy reset replies must not hide explicit errors from newer servers."""

import pytest
from requests import HTTPError, Response

from tests.e2e.pull_request.pd_http import assert_prefix_reset_response


@pytest.mark.parametrize("body", [b"", b'{"success":true}'])
def test_success_formats(body):
    response = Response()
    response.status_code = 200
    response._content = body
    assert_prefix_reset_response(response)


@pytest.mark.parametrize(
    "status,body", [(500, b""), (204, b""), (200, b'{"success":false}'), (200, b"{}"), (200, b"invalid")]
)
def test_reset_errors_are_rejected(status, body):
    response = Response()
    response.status_code = status
    response._content = body
    with pytest.raises((AssertionError, ValueError, HTTPError)):
        assert_prefix_reset_response(response)
