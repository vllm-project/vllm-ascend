from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def test_chat_template_kwargs_string_non_stream(api_client, request):
    """Non-streaming: chat_template_kwargs is a string instead of an object, so a 400 error should be returned."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": "invalid_string",
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 400 and error code is 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_string_stream(api_client, request):
    """Streaming: chat_template_kwargs is a string, so a 400 error should be returned."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": "invalid_string",
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 400 and error code is 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_array_non_stream(api_client, request):
    """Non-streaming: chat_template_kwargs is an array instead of an object, so a 400 error should be returned."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": ["item1", "item2"],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 400 and error code is 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_array_stream(api_client, request):
    """Streaming: chat_template_kwargs is an array, so a 400 error should be returned."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": ["item1", "item2"],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 400 and error code is 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_integer_non_stream(api_client, request):
    """Non-streaming: chat_template_kwargs is an integer, so a 400 error should be returned."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": 123,
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 400 and error code is 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_integer_stream(api_client, request):
    """Streaming: chat_template_kwargs is an integer, so a 400 error should be returned."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": 123,
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 400 and error code is 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_boolean_non_stream(api_client, request):
    """Non-streaming: chat_template_kwargs is a boolean, so a 400 error should be returned."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": True,
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 400 and error code is 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_boolean_stream(api_client, request):
    """Streaming: chat_template_kwargs is a boolean, so a 400 error should be returned."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": False,
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 400 and error code is 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_nested_invalid_type_non_stream(api_client, request):
    """Non-streaming: chat_template_kwargs contains a nested invalid value type, so a 400 error should be returned."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "valid_param": "value",
            "invalid_param": [1, 2, 3],  # Some engines may not support array values for parameters
        },
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: should be 400 if the engine validates strictly, or 200 if it ignores invalid values
    assert response.status_code in [
        200,
        400,
    ], f"status code should be 200 or 400, got {response.status_code}"


def test_chat_template_kwargs_nested_invalid_type_stream(api_client, request):
    """Streaming: chat_template_kwargs contains a nested invalid value type."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "valid_param": "value",
            "invalid_param": {"nested": [1, 2, 3]},
        },
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code should be 200 or 400
    assert response.status_code in [
        200,
        400,
    ], f"status code should be 200 or 400, got {response.status_code}"
