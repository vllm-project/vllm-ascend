import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_greater_than_one(api_client, stream):
    """top_p greater than 1.0 should return error code 400."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_p": 1.5,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: error code is 400
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_negative(api_client, stream):
    """A negative top_p value should return error code 400."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_p": -0.5,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: error code is 400
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_integer_type(api_client, stream):
    """An integer top_p value of 1 should be accepted by the engine."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_p": 1,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_string(api_client, stream):
    """A numeric string top_p value should be accepted by the engine."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_p": "0.9",
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_null(api_client, stream):
    """top_p=null should be accepted by the engine."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_p": None,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_array(api_client, stream):
    """An array top_p value should return error code 400."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_p": [0.5, 0.9],
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: error code is 400
    assertion.assert_error_code_400(response)
