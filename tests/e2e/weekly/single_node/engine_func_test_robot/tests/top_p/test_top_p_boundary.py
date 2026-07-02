import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_boundary_zero(api_client, stream):
    """top_p=0.0 is outside the supported range and should return error code 400."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Name the capital of China."}],
        "top_p": 0.0,
        "stream": stream,
        "max_tokens": 10,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: error code is 400
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_boundary_one(api_client, stream):
    """top_p=1.0 is the upper boundary and should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe sunset in one sentence."}],
        "top_p": 1.0,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_boundary_negative_small(api_client, stream):
    """A small negative top_p value should return error code 400."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_p": -0.0001,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: error code is 400
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_boundary_just_above_one(api_client, stream):
    """A top_p value just above 1.0 should return error code 400."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_p": 1.0001,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: error code is 400
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_boundary_precision_decimal(api_client, stream):
    """A high-precision top_p value below 1.0 should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_p": 0.999999,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_without_explicit_setting(api_client, stream):
    """top_p is omitted; the default sampling behavior should respond normally."""
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
def test_top_p_boundary_very_small_positive(api_client, stream):
    """A very small positive top_p value should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_p": 0.0001,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)
