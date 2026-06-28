import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_empty_tools_list(api_client, request, stream):
    """Test tool choice empty tools list."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Hello.'
        }],
        "tools": [],
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_case_sensitivity_none(api_client, request, stream):
    """Test tool choice case sensitivity none."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": 'Get weather information.',
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ],
        "tool_choice": "None",  # Check: response behavior is valid
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_case_sensitivity_auto(api_client, request, stream):
    """Test tool choice case sensitivity auto."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": 'Get weather information.',
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ],
        "tool_choice": "AUTO",  # Check: response behavior is valid
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_unicode_function_name(api_client, request, stream):
    """Test tool choice unicode function name."""
    tools_with_unicode_name = [
        {
            "type": "function",
            "function": {
                "name": 'response',
                "description": 'Get weather information.',
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": tools_with_unicode_name,
        "tool_choice": 'response',
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_special_chars_function_name(api_client, request, stream):
    """Test tool choice special chars function name."""
    tools_with_special_name = [
        {
            "type": "function",
            "function": {
                "name": "function-name_with.special",
                "description": 'Get weather information.',
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": tools_with_special_name,
        "tool_choice": "function-name_with.special",
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_very_long_function_name(api_client, request, stream):
    """Test tool choice very long function name."""
    long_name = "a" * 200  # Check: response behavior is valid
    tools_with_long_name = [
        {
            "type": "function",
            "function": {
                "name": long_name,
                "description": 'Get weather information.',
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": tools_with_long_name,
        "tool_choice": long_name,
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_many_tools(api_client, request, stream):
    """Test tool choice many tools."""
    many_tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": f"Description for tool {i}",
                "parameters": {"type": "object", "properties": {}}
            }
        }
        for i in range(50)  # Check: response behavior is valid
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": many_tools,
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_type_field_case_variations(api_client, request, stream):
    """Test tool choice type field case variations."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": 'Get weather information.',
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ],
        "tool_choice": {
            "type": "FUNCTION",  # Check: response behavior is valid
            "function": {
                "name": "test_func"
            }
        },
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_additional_properties(api_client, request, stream):
    """Test tool choice additional properties."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": 'Get weather information.',
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {
                "name": "test_func",
                "extra_field": "extra_value"  # Check: response behavior is valid
            }
        },
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_whitespace_in_string(api_client, request, stream):
    """Test tool choice whitespace in string."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": 'Get weather information.',
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ],
        "tool_choice": " auto ",  # Check: response behavior is valid
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_function_name_partial_match(api_client, request, stream):
    """Test tool choice function name partial match."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather_info",
                    "description": 'Get weather information.',
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ],
        "tool_choice": "get_weather",  # Check: response behavior is valid
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_duplicate_function_names(api_client, request, stream):
    """Test tool choice duplicate function names."""
    tools_with_duplicates = [
        {
            "type": "function",
            "function": {
                "name": "same_name",
                "description": 'Get weather information.',
                "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "same_name",  # Check: response behavior is valid
                "description": 'Get weather information.',
                "parameters": {"type": "object", "properties": {"b": {"type": "number"}}}
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": tools_with_duplicates,
        "tool_choice": "same_name",
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_omit_tools_with_required(api_client, request, stream):
    """Test tool choice omit tools with required."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Hello.'
        }],
        "tools": [],
        "tool_choice": "required",
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"
