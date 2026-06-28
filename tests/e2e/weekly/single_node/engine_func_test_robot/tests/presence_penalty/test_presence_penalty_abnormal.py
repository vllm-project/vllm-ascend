import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def test_presence_penalty_exceed_upper_limit_non_stream(api_client, request):
    """Test presence penalty exceed upper limit non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "presence_penalty": 2.1,
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_presence_penalty_exceed_upper_limit_stream(api_client, request):
    """Test presence penalty exceed upper limit stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "presence_penalty": 2.1,
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_presence_penalty_below_lower_limit_non_stream(api_client, request):
    """Test presence penalty below lower limit non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "presence_penalty": -2.1,
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_presence_penalty_below_lower_limit_stream(api_client, request):
    """Test presence penalty below lower limit stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "presence_penalty": -2.1,
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_presence_penalty_string_non_stream(api_client, request):
    """Test presence penalty string non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "presence_penalty": "1",
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: finish_reason is valid
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_presence_penalty_string_stream(api_client, request):
    """Test presence penalty string stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "presence_penalty": "1",
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


def test_presence_penalty_null_non_stream(api_client, request):
    """Test presence penalty null non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "presence_penalty": None,
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: finish_reason is valid
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_presence_penalty_null_stream(api_client, request):
    """Test presence penalty null stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "presence_penalty": None,
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


def test_presence_penalty_array_non_stream(api_client, request):
    """Test presence penalty array non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "presence_penalty": [0.5, 1.0],
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_presence_penalty_array_stream(api_client, request):
    """Test presence penalty array stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "presence_penalty": [0.5, 1.0],
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_presence_penalty_object_non_stream(api_client, request):
    """Test presence penalty object non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "presence_penalty": {"value": 1.0},
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_presence_penalty_object_stream(api_client, request):
    """Test presence penalty object stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "presence_penalty": {"value": 1.0},
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)