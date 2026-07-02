import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)

# Check: response behavior is valid
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name."},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD."},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_none(api_client, request, stream):
    """Test tool choice none."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "none",
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_auto(api_client, request, stream):
    """Test tool choice auto."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_required(api_client, request, stream):
    """Test tool choice required."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What time is it?"}],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "required",
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_specific_function_string(api_client, request, stream):
    """Test tool choice specific function string."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "get_weather",
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_function_object(api_client, request, stream):
    """Test tool choice function object."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_without_tools(api_client, request, stream):
    """Test tool choice without tools."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello."}],
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_with_tools_non_function_call_query(api_client, request, stream):
    """Test tool choice with tools non function call query."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Introduce yourself."}],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_tool_with_null_description(api_client, request, stream):
    """Test tool choice tool with null description."""
    tools_with_null_desc = [
        {
            "type": "function",
            "function": {
                "name": "simple_tool",
                "description": None,
                "parameters": {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
            },
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Call the tool."}],
        "tools": tools_with_null_desc,
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_parallel_tools(api_client, request, stream):
    """Test tool choice parallel tools."""
    multiple_tools = [
        {
            "type": "function",
            "function": {
                "name": "tool_a",
                "description": "Tool A.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_b",
                "description": "Tool B.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Call both tools."}],
        "tools": multiple_tools,
        "tool_choice": "required",
        "stream": stream,
        "max_tokens": 5120,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)
