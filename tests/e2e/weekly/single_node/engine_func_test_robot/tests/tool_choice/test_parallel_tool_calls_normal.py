import json

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
            "description": 'Get weather information.',
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": 'City name.'
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": 'Get current time.',
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": 'Search the web.',
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": 'Search query.'
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def _validate_tool_calls_response(response, stream, expected_tools=None):
    """Test  validate tool calls response."""
    if expected_tools is None:
        expected_tools = ["get_weather", "get_time"]

    if stream:
        # Check: response behavior is valid
        assertion.assert_stream_has_done(response.text)

        tool_calls_found = False
        finish_reason_tool_calls = False
        for line in response.text.strip().split("\n"):
            if line.startswith("data: ") and "[DONE]" not in line:
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("tool_calls"):
                        tool_calls_found = True
                        for tc in delta.get("tool_calls", []):
                            assert "index" in tc, 'response'
                            if tc.get("id"):
                                assert tc.get("type") == "function", 'response'
                            if tc.get("function"):
                                func = tc.get("function", {})
                                if func.get("name"):
                                    assert func.get("name") in expected_tools, \
                                        f"Function name should be one of {expected_tools}, got {func.get('name')}"
                    if choices[0].get("finish_reason") == "tool_calls":
                        finish_reason_tool_calls = True

        assert tool_calls_found, 'response'
        assert finish_reason_tool_calls, 'response'
    else:
        # Check: response behavior is valid
        response_json = response.json()
        choices = response_json.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})

            if message.get("tool_calls") or choice.get("finish_reason") == "tool_calls":
                assert choice.get("finish_reason") == "tool_calls", \
                    f"finish_reason should be tool_calls, got {choice.get('finish_reason')}"

                tool_calls = message.get("tool_calls", [])
                assert len(tool_calls) >= 1, 'response'

                for tc in tool_calls:
                    assert "id" in tc, 'response'
                    assert tc.get("type") == "function", 'response'
                    assert "function" in tc, 'response'
                    func = tc.get("function", {})
                    assert "name" in func, "function should contain name"
                    assert "arguments" in func, 'response'
                    assert func.get("name") in expected_tools, \
                        f"Function name should be one of {expected_tools}, got {func.get('name')}"
                    args = json.loads(func.get("arguments", "{}"))
                    assert isinstance(args, dict), 'response'


def _validate_tool_calls_count(response, stream, max_count):
    """Test  validate tool calls count."""
    if stream:
        assertion.assert_stream_has_done(response.text)
    else:
        response_json = response.json()
        choices = response_json.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            if message.get("tool_calls") or choice.get("finish_reason") == "tool_calls":
                tool_calls = message.get("tool_calls", [])
                assert len(tool_calls) <= max_count, \
                    f"tool_calls count should be <= {max_count}, got {len(tool_calls)}"


def _validate_no_tool_calls(response, stream):
    """Test  validate no tool calls."""
    if stream:
        # Check: response behavior is valid
        assertion.assert_stream_has_done(response.text)

        for line in response.text.strip().split("\n"):
            if line.startswith("data: ") and "[DONE]" not in line:
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    # Check: tool_calls structure is valid
                    if delta.get("tool_calls"):
                        assert False, (
                            "Streaming response should not contain tool_calls, "
                            f"got {delta.get('tool_calls')}"
                        )
                    # Check: finish_reason is valid
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        assert finish_reason != "tool_calls", (
                            f"finish_reason should not be tool_calls, got {finish_reason}"
                        )
    else:
        # Check: response behavior is valid
        response_json = response.json()
        choices = response_json.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])
            assert tool_calls is None or len(tool_calls) == 0, (
                f"Response should not contain tool_calls, got {tool_calls}"
            )
            finish_reason = choice.get("finish_reason")
            assert finish_reason != "tool_calls", f"finish_reason should not be tool_calls, got {finish_reason}"


def _validate_tool_calls_with_stop_finish_reason(response, stream, expected_tools=None):
    """Test  validate tool calls with stop finish reason."""
    if expected_tools is None:
        expected_tools = ["get_weather"]

    if stream:
        # Check: response behavior is valid
        assertion.assert_stream_has_done(response.text)

        tool_calls_found = False
        finish_reason_stop = False
        for line in response.text.strip().split("\n"):
            if line.startswith("data: ") and "[DONE]" not in line:
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("tool_calls"):
                        tool_calls_found = True
                        for tc in delta.get("tool_calls", []):
                            assert "index" in tc, 'response'
                            if tc.get("id"):
                                assert tc.get("type") == "function", 'response'
                            if tc.get("function"):
                                func = tc.get("function", {})
                                if func.get("name"):
                                    assert func.get("name") in expected_tools, \
                                        f"Function name should be one of {expected_tools}, got {func.get('name')}"
                    # Check: finish_reason is valid
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        assert finish_reason == "stop", \
                            f"Named-function tool_choice should finish with stop, got {finish_reason}"
                        finish_reason_stop = True

        assert tool_calls_found, 'response'
        assert finish_reason_stop, 'response'
    else:
        # Check: response behavior is valid
        response_json = response.json()
        choices = response_json.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})

            if message.get("tool_calls"):
                tool_calls = message.get("tool_calls", [])
                assert len(tool_calls) >= 1, 'response'

                for tc in tool_calls:
                    assert "id" in tc, 'response'
                    assert tc.get("type") == "function", 'response'
                    assert "function" in tc, 'response'
                    func = tc.get("function", {})
                    assert "name" in func, "function should contain name"
                    assert "arguments" in func, 'response'
                    assert func.get("name") in expected_tools, \
                        f"Function name should be one of {expected_tools}, got {func.get('name')}"
                    args = json.loads(func.get("arguments", "{}"))
                    assert isinstance(args, dict), 'response'

                # Check: finish_reason is valid
                finish_reason = choice.get("finish_reason")
                assert finish_reason == "stop", \
                    f"Named-function tool_choice should finish with stop, got {finish_reason}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_true(api_client, request, stream):
    """Test parallel tool calls true."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: response behavior is valid
    _validate_tool_calls_response(response, stream, expected_tools=["get_weather"])


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_false(api_client, request, stream):
    """Test parallel tool calls false."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: tool_calls structure is valid
    _validate_tool_calls_count(response, stream, max_count=1)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_default_value(api_client, request, stream):
    """Test parallel tool calls default value."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        # Check: tool_calls structure is valid
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: response behavior is valid
    _validate_tool_calls_response(response, stream, expected_tools=["get_weather", "get_time"])


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_tool_choice_none(api_client, request, stream):
    """Test parallel tool calls with tool choice none."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "none",
        "parallel_tool_calls": True,  # Check: response behavior is valid
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: tool_calls structure is valid
    _validate_no_tool_calls(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_tool_choice_required(api_client, request, stream):
    """Test parallel tool calls with tool choice required."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: tool_calls structure is valid
    _validate_tool_calls_response(response, stream, expected_tools=["get_weather", "get_time", "search_web"])


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_specific_function(api_client, request, stream):
    """Test parallel tool calls with specific function."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {
            "type": "function",
            "function": {
                "name": "get_weather"
            }
        },
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: tool_calls structure is valid
    # Check: finish_reason is valid
    _validate_tool_calls_with_stop_finish_reason(response, stream, expected_tools=["get_weather"])


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_false_with_specific_function(api_client, request, stream):
    """Test parallel tool calls false with specific function."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {
            "type": "function",
            "function": {
                "name": "get_weather"
            }
        },
        "parallel_tool_calls": False,
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: tool_calls structure is valid
    _validate_tool_calls_count(response, stream, max_count=1)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_without_tools(api_client, request, stream):
    """Test parallel tool calls without tools."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Hello.'
        }],
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: tool_calls structure is valid
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_single_tool(api_client, request, stream):
    """Test parallel tool calls single tool."""
    single_tool = [
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

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'What is the weather?'
        }],
        "tools": single_tool,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: tool_calls structure is valid
    _validate_tool_calls_response(response, stream, expected_tools=["get_weather"])


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_non_function_call_query(api_client, request, stream):
    """Test parallel tool calls non function call query."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Introduce yourself.'
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: tool_calls structure is valid
    _validate_no_tool_calls(response, stream)
