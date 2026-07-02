import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


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
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_k": top_k,
        "stream": stream,
        "max_tokens": 100,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


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
    assertion.assert_chat_completion_success(response, stream)


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
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_disable_with_minus_one(api_client, stream):
    """top_k=-1 disables top-k filtering; the request should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_k": -1,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)
