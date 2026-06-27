import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_very_long_value(api_client, request, stream):
    """chat_template_kwargs value is an extremely long string; boundary test."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"custom_param": "a" * 10000},  # Extremely long string value
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code should be 200 if the engine accepts it, or 400 if it exceeds the limit
    assert response.status_code in [
        200,
        400,
    ], f"status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_many_keys(api_client, request, stream):
    """chat_template_kwargs contains many key-value pairs; boundary test."""
    # Build an object with many keys
    kwargs = {f"param_{i}": f"value_{i}" for i in range(100)}

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": kwargs,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code should be 200 or 400
    assert response.status_code in [
        200,
        400,
    ], f"status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_unicode_keys(api_client, request, stream):
    """chat_template_kwargs contains Unicode key names; boundary test."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "中文参数": "value",
            "日本語パラメータ": "value",
            "emoji_参数": "value",
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code should be 200 or 400 depending on whether the engine supports non-ASCII key names
    assert response.status_code in [
        200,
        400,
    ], f"status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_special_chars_in_keys(api_client, request, stream):
    """chat_template_kwargs key names contain special characters; boundary test."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "param-with-dash": "value",
            "param_with_underscore": "value",
            "param.with.dot": "value",
            "param:with:colon": "value",
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code should be 200 or 400
    assert response.status_code in [
        200,
        400,
    ], f"status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_numeric_string_values(api_client, request, stream):
    """chat_template_kwargs values are numeric strings; boundary type-conversion test."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "number_as_string": "12345",
            "float_as_string": "3.14159",
            "bool_as_string": "true",
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code should be 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_mixed_types_values(api_client, request, stream):
    """chat_template_kwargs values have mixed types (number, boolean, null); boundary test."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "int_value": 42,
            "float_value": 3.14,
            "bool_value": True,
            "null_value": None,
            "string_value": "text",
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code should be 200 or 400 depending on how the engine handles non-string values
    assert response.status_code in [
        200,
        400,
    ], f"status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_reserved_words_keys(api_client, request, stream):
    """chat_template_kwargs uses reserved words or internal keywords as key names; boundary test."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "model": "overridden_model",  # Key that may conflict with request parameters
            "messages": "overridden",  # Key that may conflict with request parameters
            "stream": True,  # Key that may conflict with request parameters
            "temperature": 2.0,  # Key that may conflict with generation parameters
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code should be 200 if the engine isolates namespaces correctly, or 400 if there is a conflict
    assert response.status_code in [
        200,
        400,
    ], f"status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_empty_string_values(api_client, request, stream):
    """chat_template_kwargs values are empty strings; boundary test."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "empty_string": "",
            "whitespace_only": "   ",
            "null_string": "null",
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_deeply_nested_object(api_client, request, stream):
    """chat_template_kwargs is a deeply nested object; boundary test."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"level1": {"level2": {"level3": {"level4": {"level5": {"deep_value": "found"}}}}}},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code should be 200 if the engine flattens nested objects, or 400 if it rejects nesting
    assert response.status_code in [
        200,
        400,
    ], f"status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_case_sensitive_keys(api_client, request, stream):
    """chat_template_kwargs key names are case-sensitive; boundary test."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "Add_Generation_Prompt": True,  # Different from the standard snake_case form
            "ADD_GENERATION_PROMPT": True,  # All uppercase
            "add_generation_prompt": True,  # Standard lowercase
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code should be 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)
