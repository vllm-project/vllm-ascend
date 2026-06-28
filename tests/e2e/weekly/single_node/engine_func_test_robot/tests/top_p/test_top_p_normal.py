import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize(
    "top_p",
    [0.1, 0.5, 0.9, 0.99, 1.0],
    ids=["p0.1", "p0.5", "p0.9", "p0.99", "p1.0"],
)
def test_top_p_normal_values(api_client, stream, top_p):
    """top_p is within the normal range; the request should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe a summer night in one sentence."}],
        "top_p": top_p,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_with_temperature(api_client, stream):
    """top_p can be used together with temperature."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
        "top_p": 0.85,
        "temperature": 0.8,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_with_top_k(api_client, stream):
    """top_p can be used together with top_k."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe spring flowers in one sentence."}],
        "top_p": 0.9,
        "top_k": 50,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_small_conservative(api_client, stream):
    """top_p=0.1 should keep nucleus sampling conservative and respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Recommend one good book."}],
        "top_p": 0.1,
        "temperature": 0.7,
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_greedy_zero(api_client, stream):
    """top_p=0.0 is outside the supported range and should return error code 400."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Name the capital of China."}],
        "top_p": 0.0,
        "temperature": 0.0,
        "stream": stream,
        "max_tokens": 10,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: error code is 400
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_disable_with_one(api_client, stream):
    """top_p=1.0 disables nucleus filtering and should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe autumn in three words."}],
        "top_p": 1.0,
        "stream": stream,
        "max_tokens": 30,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_without_setting(api_client, stream):
    """top_p is omitted; the default sampling behavior should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Please answer briefly."}],
        "stream": stream,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: request succeeds and finish_reason is valid
    assertion.assert_chat_completion_success(response, stream)
