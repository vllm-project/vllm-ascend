import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_zero(api_client, stream):
    """top_k=0 is accepted by the engine and should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Please answer briefly."}],
        "top_k": 0,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_negative_not_minus_one(api_client, request, stream):
    """A negative top_k value other than -1 should return error code 400."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Please answer briefly."}],
        "top_k": -5,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status behavior differs by architecture, but error code is 400
    assertion.assert_architecture_error_code_400_response(response, request, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_exceed_vocab_size(api_client, stream):
    """A top_k value above vocab size should be accepted and clamped by the engine."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Please answer briefly."}],
        "top_k": 999999999,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_float(api_client, request, stream):
    """A float top_k value should return error code 400."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Please answer briefly."}],
        "top_k": 10.5,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status behavior differs by architecture, but error code is 400
    assertion.assert_architecture_error_code_400_response(response, request, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_string(api_client, stream):
    """A numeric string top_k value should be accepted by the engine."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Please answer briefly."}],
        "top_k": "50",
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_k_null(api_client, stream):
    """top_k=null should be accepted by the engine."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Please answer briefly."}],
        "top_k": None,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)
