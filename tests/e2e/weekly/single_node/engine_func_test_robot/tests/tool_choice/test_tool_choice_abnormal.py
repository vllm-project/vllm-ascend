import pytest
from ...utility import request_helper as helper
from ...utility import assertion


TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": 'Get weather information.',
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]


def test_tool_choice_invalid_string_non_stream(api_client, request):
    """Test tool choice invalid string non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "invalid_choice",
        "stream": False,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_invalid_string_stream(api_client, request):
    """Test tool choice invalid string stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "unknown_option",
        "stream": True,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_nonexistent_function_name_non_stream(api_client, request):
    """Test tool choice nonexistent function name non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "nonexistent_function",
        "stream": False,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_nonexistent_function_name_stream(api_client, request):
    """Test tool choice nonexistent function name stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "nonexistent_function",
        "stream": True,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_null_non_stream(api_client, request):
    """Test tool choice null non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": None,
        "stream": False,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: finish_reason is valid
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_tool_choice_null_stream(api_client, request):
    """Test tool choice null stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": None,
        "stream": True,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_tool_choice_integer_non_stream(api_client, request):
    """Test tool choice integer non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": 123,
        "stream": False,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_integer_stream(api_client, request):
    """Test tool choice integer stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": 0,
        "stream": True,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_empty_object_non_stream(api_client, request):
    """Test tool choice empty object non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {},
        "stream": False,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_empty_object_stream(api_client, request):
    """Test tool choice empty object stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {},
        "stream": True,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_malformed_function_object_non_stream(api_client, request):
    """Test tool choice malformed function object non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {
            "type": "function"
            # Check: response behavior is valid
        },
        "stream": False,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_malformed_function_object_stream(api_client, request):
    """Test tool choice malformed function object stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {
            "type": "function",
            "function": {}  # Check: response behavior is valid
        },
        "stream": True,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_wrong_type_value_non_stream(api_client, request):
    """Test tool choice wrong type value non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {
            "type": "invalid_type",
            "function": {"name": "get_weather"}
        },
        "stream": False,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_wrong_type_value_stream(api_client, request):
    """Test tool choice wrong type value stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {
            "type": "tool",
            "function": {"name": "get_weather"}
        },
        "stream": True,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_array_type_non_stream(api_client, request):
    """Test tool choice array type non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": ["get_weather"],
        "stream": False,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_array_type_stream(api_client, request):
    """Test tool choice array type stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": ["auto"],
        "stream": True,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_boolean_non_stream(api_client, request):
    """Test tool choice boolean non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": True,
        "stream": False,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_tool_choice_boolean_stream(api_client, request):
    """Test tool choice boolean stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": False,
        "stream": True,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)
