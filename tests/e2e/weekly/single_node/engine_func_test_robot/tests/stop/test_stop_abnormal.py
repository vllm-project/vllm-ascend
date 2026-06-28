from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def test_stop_integer_non_stream(api_client, request):
    """Test stop integer non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell a short story."}],
        "stop": 123,
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    if response.status_code == 200:
        finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)
    else:
        assertion.assert_error_code_400(response)


def test_stop_integer_stream(api_client, request):
    """Test stop integer stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell a short story."}],
        "stop": 123,
        "stream": True,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    if response.status_code == 200:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
        assertion.assert_finish_reason_valid(finish_reason)
    else:
        assertion.assert_error_code_400(response)


def test_stop_null_non_stream(api_client, request):
    """Test stop null non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell a short story."}],
        "stop": None,
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: finish_reason is valid
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_stop_null_stream(api_client, request):
    """Test stop null stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell a short story."}],
        "stop": None,
        "stream": True,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    assertion.assert_stream_has_done(response.text)

    # Check: finish_reason is valid
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_stop_nested_array_non_stream(api_client, request):
    """Test stop nested array non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell a short story."}],
        "stop": [["a", "b"]],
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_stop_nested_array_stream(api_client, request):
    """Test stop nested array stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell a short story."}],
        "stop": [["a", "b"]],
        "stream": True,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_stop_object_non_stream(api_client, request):
    """Test stop object non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell a short story."}],
        "stop": {"key": "value"},
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    if response.status_code == 200:
        finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)
    else:
        assertion.assert_error_code_400(response)


def test_stop_object_stream(api_client, request):
    """Test stop object stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell a short story."}],
        "stop": {"key": "value"},
        "stream": True,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    if response.status_code == 200:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
        assertion.assert_finish_reason_valid(finish_reason)
    else:
        assertion.assert_error_code_400(response)


def test_stop_exceed_max_limit_non_stream(api_client, request):
    """Test stop exceed max limit non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell a short story."}],
        "stop": ["a", "b", "c", "d", "e", "f"],
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_stop_exceed_max_limit_stream(api_client, request):
    """Test stop exceed max limit stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell a short story."}],
        "stop": ["a", "b", "c", "d", "e", "f"],
        "stream": True,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    assertion.assert_status_code_200(response)
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_stop_sequence_too_long_non_stream(api_client, request):
    """Test stop sequence too long non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell a short story."}],
        "stop": ["a" * 100],  # Check: response behavior is valid
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: finish_reason is valid
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_stop_sequence_too_long_stream(api_client, request):
    """Test stop sequence too long stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell a short story."}],
        "stop": ["a" * 100],
        "stream": True,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    assertion.assert_stream_has_done(response.text)

    # Check: finish_reason is valid
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)
