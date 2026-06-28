import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def _assert_success_response(response, stream):
    assertion.assert_status_code_200(response)

    if stream:
        assertion.assert_stream_has_done(response.text)
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize(
    "top_k",
    [1, 10, 50, 100, 1000],
    ids=["k1", "k10", "k50", "k100", "k1000"],
)
def test_top_k_normal_values(api_client, stream, top_k):
    """top_k is within the normal range; the request should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Please answer briefly."}],
        "top_k": top_k,
        "stream": stream,
        "max_tokens": 100,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    _assert_success_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_with_temperature(api_client, stream):
    """top_k can be used together with temperature."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
        "top_k": 50,
        "temperature": 0.8,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    _assert_success_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_with_top_p(api_client, stream):
    """top_k can be used together with top_p."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
        "top_k": 50,
        "top_p": 0.9,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    _assert_success_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_disable_with_minus_one(api_client, stream):
    """top_k=-1 disables top-k filtering; the request should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Please answer briefly."}],
        "top_k": -1,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    _assert_success_response(response, stream)
