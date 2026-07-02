import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_boundary_one(api_client, stream):
    """top_k=1 is the lower positive boundary and should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
        "top_k": 1,
        "stream": stream,
        "max_tokens": 30,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_very_large(api_client, stream):
    """A very large top_k value should be accepted and clamped by the engine."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_k": 100000,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_minus_one_boundary(api_client, stream):
    """top_k=-1 is the disable-filtering boundary and should respond normally."""
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


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_without_setting(api_client, stream):
    """top_k is omitted; the default sampling behavior should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_zero_boundary(api_client, stream):
    """top_k=0 is accepted by the engine and should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_k": 0,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_minus_two_boundary(api_client, request, stream):
    """top_k=-2 is outside the supported range and should return error code 400."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_k": -2,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status behavior differs by architecture, but error code is 400
    assertion.assert_architecture_error_code_400_response(response, request, stream)
