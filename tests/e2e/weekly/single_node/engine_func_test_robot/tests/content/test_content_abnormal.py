from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def test_content_integer_non_stream(api_client, request):
    """Non-streaming: content is integer type, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": 12345}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_integer_stream(api_client, request):
    """Streaming: content is integer type, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": 12345}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_object_non_stream(api_client, request):
    """Non-streaming: content is object type (non-standard multimodal format), should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": {"text": "hello", "extra": "data"}}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_object_stream(api_client, request):
    """Streaming: content is object type, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": {"text": "hello", "extra": "data"}}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_boolean_non_stream(api_client, request):
    """Non-streaming: content is boolean type, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": True}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_boolean_stream(api_client, request):
    """Streaming: content is boolean type, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": False}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_invalid_format_non_stream(api_client, request):
    """Non-streaming: content is array but format does not conform to OpenAI multimodal spec (string array),
    should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": ["invalid", "array", "format"]}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_invalid_format_stream(api_client, request):
    """Streaming: content is array but format does not conform to OpenAI multimodal spec, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": ["invalid", "array", "format"]}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


# ==================== Content Array Abnormal Tests ====================


def test_content_array_missing_type_non_stream(api_client, request):
    """Non-streaming: content array object missing type field, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"text": "你好"}]}],  # missing type field
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_missing_type_stream(api_client, request):
    """Streaming: content array object missing type field, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"text": "你好"}]}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_missing_text_non_stream(api_client, request):
    """Non-streaming: content array object type is text but missing text field, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text"}]}],  # missing text field
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_invalid_type_non_stream(api_client, request):
    """Non-streaming: content array object type is invalid, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "invalid_type", "text": "你好"}]}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_invalid_type_stream(api_client, request):
    """Streaming: content array object type is invalid, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "unknown", "text": "你好"}]}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_text_field_null_non_stream(api_client, request):
    """Non-streaming: content array object text field is null, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": None}]}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_text_field_integer_non_stream(api_client, request):
    """Non-streaming: content array object text field is integer type, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": 12345}]}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_text_field_integer_stream(api_client, request):
    """Streaming: content array object text field is integer type, should return 400 error"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": 12345}]}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code 400, error code 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)
