import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def test_repetition_penalty_less_than_one_non_stream(api_client, request):
    """Test repetition penalty less than one non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": 0.9,
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: finish_reason is valid
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_repetition_penalty_less_than_one_stream(api_client, request):
    """Test repetition penalty less than one stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": 0.9,
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    assertion.assert_stream_has_done(response.text)

    # Check: finish_reason is valid
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_repetition_penalty_zero_non_stream(api_client, request):
    """Test repetition penalty zero non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": 0.0,
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_repetition_penalty_zero_stream(api_client, request):
    """Test repetition penalty zero stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": 0.0,
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_repetition_penalty_negative_non_stream(api_client, request):
    """Test repetition penalty negative non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": -1.0,
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_repetition_penalty_negative_stream(api_client, request):
    """Test repetition penalty negative stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": -1.0,
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_repetition_penalty_string_non_stream(api_client, request):
    """Test repetition penalty string non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": "1",
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: finish_reason is valid
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_repetition_penalty_string_stream(api_client, request):
    """Test repetition penalty string stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": "1",
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    assertion.assert_stream_has_done(response.text)

    # Check: finish_reason is valid
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_repetition_penalty_null_non_stream(api_client, request):
    """Test repetition penalty null non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": None,
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: finish_reason is valid
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_repetition_penalty_null_stream(api_client, request):
    """Test repetition penalty null stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": None,
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    assertion.assert_stream_has_done(response.text)

    # Check: finish_reason is valid
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_repetition_penalty_array_non_stream(api_client, request):
    """Test repetition penalty array non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": [1.0, 1.2],
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_repetition_penalty_array_stream(api_client, request):
    """Test repetition penalty array stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": [1.0, 1.2],
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_repetition_penalty_object_non_stream(api_client, request):
    """Test repetition penalty object non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": {"value": 1.2},
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_repetition_penalty_object_stream(api_client, request):
    """Test repetition penalty object stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": {"value": 1.2},
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)
