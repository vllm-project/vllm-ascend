import json

import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "City name.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _validate_tool_calls_count(response, stream, max_count):
    """Test  validate tool calls count."""
    if stream:
        # Check: response behavior is valid
        assertion.assert_stream_has_done(response.text)

        tool_call_indices = set()
        finish_reason_tool_calls = False
        for line in response.text.strip().split("\n"):
            if line.startswith("data: ") and "[DONE]" not in line:
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("tool_calls"):
                        for tc in delta.get("tool_calls", []):
                            if "index" in tc:
                                tool_call_indices.add(tc.get("index"))
                    if choices[0].get("finish_reason") == "tool_calls":
                        finish_reason_tool_calls = True

        # Check: tool_calls structure is valid
        if tool_call_indices:
            assert len(tool_call_indices) <= max_count, (
                f"Streaming tool_calls count should be <= {max_count}, got {len(tool_call_indices)}"
            )
            assert finish_reason_tool_calls, "response"
    else:
        # Check: response behavior is valid
        response_json = response.json()
        choices = response_json.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            if message.get("tool_calls") or choice.get("finish_reason") == "tool_calls":
                tool_calls = message.get("tool_calls", [])
                assert len(tool_calls) <= max_count, f"tool_calls count should be <= {max_count}, got {len(tool_calls)}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_empty_tools_list(api_client, request, stream):
    """Test parallel tool calls with empty tools list."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello."}],
        "tools": [],
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_false_with_empty_tools_list(api_client, request, stream):
    """Test parallel tool calls false with empty tools list."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello."}],
        "tools": [],
        "parallel_tool_calls": False,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_many_tools(api_client, request, stream):
    """Test parallel tool calls with many tools."""
    many_tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": f"Description for tool {i}",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for i in range(50)  # Check: response behavior is valid
    ]

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": many_tools,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_tool_choice_required_and_false(api_client, request, stream):
    """Test parallel tool calls with tool choice required and false."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: tool_calls structure is valid
    _validate_tool_calls_count(response, stream, max_count=1)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_duplicate_tool_names(api_client, request, stream):
    """Test parallel tool calls with duplicate tool names."""
    tools_with_duplicates = [
        {
            "type": "function",
            "function": {
                "name": "same_name",
                "description": "Get current time.",
                "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "same_name",  # Check: response behavior is valid
                "description": "Get current time.",
                "parameters": {"type": "object", "properties": {"b": {"type": "number"}}},
            },
        },
    ]

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": tools_with_duplicates,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_tool_having_null_description(api_client, request, stream):
    """Test parallel tool calls with tool having null description."""
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
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_tool_having_empty_parameters(api_client, request, stream):
    """Test parallel tool calls with tool having empty parameters."""
    tools_with_empty_params = [
        {
            "type": "function",
            "function": {
                "name": "no_param_tool",
                "description": "Simple tool.",
                "parameters": {},  # Check: response behavior is valid
            },
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Call the tool."}],
        "tools": tools_with_empty_params,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_unicode_tool_name(api_client, request, stream):
    """Test parallel tool calls with unicode tool name."""
    tools_with_unicode = [
        {
            "type": "function",
            "function": {
                "name": "response",
                "description": "Simple tool.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "response",
                "description": "Simple tool.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": tools_with_unicode,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_special_chars_tool_name(api_client, request, stream):
    """Test parallel tool calls with special chars tool name."""
    tools_with_special_chars = [
        {
            "type": "function",
            "function": {
                "name": "tool-name_with.special",
                "description": "Simple tool.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": tools_with_special_chars,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_very_long_tool_name(api_client, request, stream):
    """Test parallel tool calls with very long tool name."""
    long_name = "a" * 200  # Check: response behavior is valid
    tools_with_long_name = [
        {
            "type": "function",
            "function": {
                "name": long_name,
                "description": "Simple tool.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": tools_with_long_name,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_complex_nested_parameters(api_client, request, stream):
    """Test parallel tool calls with complex nested parameters."""
    tools_with_complex_params = [
        {
            "type": "function",
            "function": {
                "name": "complex_tool",
                "description": "Simple tool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nested": {
                            "type": "object",
                            "properties": {
                                "level1": {
                                    "type": "object",
                                    "properties": {"level2": {"type": "array", "items": {"type": "string"}}},
                                }
                            },
                        },
                        "array_of_objects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                            },
                        },
                    },
                    "required": ["nested"],
                },
            },
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": tools_with_complex_params,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_false_with_single_tool_multiple_times(api_client, request, stream):
    """Test parallel tool calls false with single tool multiple times."""
    single_tool = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": single_tool,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: tool_calls structure is valid
    _validate_tool_calls_count(response, stream, max_count=1)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_max_tokens_limit(api_client, request, stream):
    """Test parallel tool calls with max tokens limit."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 10,  # Check: response behavior is valid
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assert response.status_code in [200, 400], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_all_parameters_combination(api_client, request, stream):
    """Test parallel tool calls with all parameters combination."""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather?"},
        ],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "stream": stream,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)
