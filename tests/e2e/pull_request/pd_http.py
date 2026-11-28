# SPDX-License-Identifier: Apache-2.0
"""HTTP compatibility for PD contracts across supported vLLM versions."""


def assert_prefix_reset_response(response) -> None:
    """Accept legacy empty HTTP 200 and newer explicit success responses.

    Legacy vLLM does not report reset success. The caller must additionally
    require a nonempty remote KV transfer, proving D did not serve a local hit.
    """
    response.raise_for_status()
    assert response.status_code == 200, response.text
    if response.content:
        assert response.json().get("success") is True, response.text
