import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def test_max_tokens_zero_non_stream(api_client, request):
    """Test max tokens zero non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 0,
        "stream": False
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_max_tokens_zero_stream(api_client, request):
    """Test max tokens zero stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 0,
        "stream": True
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_max_tokens_negative_non_stream(api_client, request):
    """Test max tokens negative non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": -10,
        "stream": False
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_max_tokens_negative_stream(api_client, request):
    """Test max tokens negative stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": -10,
        "stream": True
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_max_tokens_float_non_stream(api_client, request):
    """Test max tokens float non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 50.5,
        "stream": False
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_max_tokens_float_stream(api_client, request):
    """Test max tokens float stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": 50.5,
        "stream": True
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_max_tokens_string_non_stream(api_client, request):
    """Test max tokens string non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": "100",
        "stream": False
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    if assertion.has_error_code(response):
        # Check: status code and error code are 400
        assertion.assert_error_code_400(response)
    else:
        # Check: finish_reason is valid
        finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)


def test_max_tokens_string_stream(api_client, request):
    """Test max tokens string stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": "100",
        "stream": True
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    if assertion.has_error_code(response):
        # Check: status code and error code are 400
        assertion.assert_error_code_400(response)
    else:
        # Check: finish_reason is valid
        assertion.assert_stream_has_done(response.text)
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
        assertion.assert_finish_reason_valid(finish_reason)


def test_max_tokens_boolean_non_stream(api_client, request):
    """Test max tokens boolean non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": True,
        "stream": False
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    if assertion.has_error_code(response):
        # Check: status code and error code are 400
        assertion.assert_error_code_400(response)
    else:
        # Check: finish_reason is valid
        finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)


def test_max_tokens_boolean_stream(api_client, request):
    """Test max tokens boolean stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Write a short line.'
        }],
        "max_tokens": False,
        "stream": True
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    if assertion.has_error_code(response):
        # Check: status code and error code are 400
        assertion.assert_error_code_400(response)
    else:
        # Check: finish_reason is valid
        assertion.assert_stream_has_done(response.text)
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
        assertion.assert_finish_reason_valid(finish_reason)
