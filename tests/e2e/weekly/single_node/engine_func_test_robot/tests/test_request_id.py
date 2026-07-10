import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
import regex as re

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion, completion_request

CONCURRENT_REQUESTS = 3


def _request_body(endpoint, stream):
    if endpoint == "chat":
        return completion_request.chat_request_body(
            messages=[{"role": "user", "content": "Hello."}],
            stream=stream,
            max_tokens=10,
        )
    return {
        "model": "auto",
        "prompt": "Hello.",
        "stream": stream,
        "max_tokens": 10,
    }


def _send_request_id_request(api_client, endpoint, stream, request_id):
    uri = completion_request.CHAT_URI if endpoint == "chat" else completion_request.COMPLETIONS_URI
    return completion_request.send_request(
        api_client,
        uri,
        _request_body(endpoint, stream),
        headers={"X-Request-ID": request_id},
    )


def _response_error_code(response):
    if "application/json" in response.headers.get("Content-Type", ""):
        body = response.json()
        return body.get("error", {}).get("code") or body.get("code")
    match = re.search(r'"code"\s*:\s*(\d+)', response.text, re.M)
    return int(match.group(1)) if match else None


def _assert_request_id_response(response, request_id, stream, allow_400):
    if allow_400 and response.status_code != 200:
        assertion.assert_validation_error_response(response)
        return
    if allow_400 and _response_error_code(response) == 400:
        return

    if stream:
        assertion.assert_status_code_200(response)
        assertion.assert_stream_has_done(response.text)
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
        assertion.assert_finish_reason_valid(finish_reason)
        id_match = re.search(r'"id"\s*:\s*"([^"]+)"', response.text)
        if id_match:
            assert id_match.group(1).endswith(request_id)
    else:
        assertion.assert_status_code_200(response)
        body = response.json()
        assertion.assert_finish_reason_valid(body["choices"][0]["finish_reason"])
        if body.get("id"):
            assert body["id"].endswith(request_id)


@pytest.mark.parametrize("endpoint", ["chat", "completions"])
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("same_request_id", [True, False], ids=["same_id", "different_ids"])
def test_request_id_is_propagated_under_concurrency(api_client, endpoint, stream, same_request_id):
    shared_request_id = f"test-same-id-{uuid.uuid4().hex[:8]}"
    request_ids = [
        shared_request_id if same_request_id else f"test-unique-id-{uuid.uuid4().hex[:8]}"
        for _ in range(CONCURRENT_REQUESTS)
    ]

    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = [
            executor.submit(_send_request_id_request, api_client, endpoint, stream, request_id)
            for request_id in request_ids
        ]

    for future, request_id in zip(futures, request_ids):
        response = future.result()
        _assert_request_id_response(response, request_id, stream, allow_400=same_request_id)
