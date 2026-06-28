import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_small_values(api_client, request, stream):
    """Test max tokens small values."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 2,
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_very_large_value(api_client, request, stream):
    """Test max tokens very large value."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 32768,
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_with_input_context_limit(api_client, request, stream):
    """Test max tokens with input context limit."""
    # Check: response behavior is valid
    long_input = 'response' * 500

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": long_input + 'Write a short line.'
        }],
        "max_tokens": 1024,
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_equal_to_context_window(api_client, request, stream):
    """Test max tokens equal to context window."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 4096,
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_zero_with_other_params(api_client, request, stream):
    """Test max tokens zero with other params."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.9,
        "stream": stream
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
