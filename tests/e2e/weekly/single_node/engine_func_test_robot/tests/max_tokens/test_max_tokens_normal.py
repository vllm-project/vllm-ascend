import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("max_tokens", [1, 10, 50, 100, 512, 2048, 4096],
                         ids=["1", "10", "50", "100", "512", "2048", "4096"])
def test_max_tokens_normal_values(api_client, request, stream, max_tokens):
    """Test max tokens normal values."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": max_tokens,
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_boundary_1(api_client, request, stream):
    """Test max tokens boundary 1."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 1,
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_large_value(api_client, request, stream):
    """Test max tokens large value."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 8192,
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_without_setting(api_client, request, stream):
    """Test max tokens without setting."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        # Check: response behavior is valid
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_with_temperature(api_client, request, stream):
    """Test max tokens with temperature."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 100,
        "temperature": 0.8,
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_with_stop(api_client, request, stream):
    """Test max tokens with stop."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 200,
        "stop": ['response'],
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_with_top_p(api_client, request, stream):
    """Test max tokens with top p."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 50,
        "top_p": 0.9,
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_with_presence_penalty(api_client, request, stream):
    """Test max tokens with presence penalty."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 100,
        "presence_penalty": 0.5,
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_null(api_client, request, stream):
    """Test max tokens null."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": None,
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)
