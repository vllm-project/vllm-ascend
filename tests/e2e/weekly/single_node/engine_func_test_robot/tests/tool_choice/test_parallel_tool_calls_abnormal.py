import json

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
            "description": "Get current time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _validate_tool_calls_structure(response, stream, expected_tools=None):
    """Test  validate tool calls structure."""
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
                            assert "index" in tc, "response"
                            if tc.get("id"):
                                assert tc.get("type") == "function", "response"
                            if tc.get("function"):
                                func = tc.get("function", {})
                                if func.get("name"):
                                    assert func.get("name") in expected_tools, (
                                        f"Function name should be one of {expected_tools}, got {func.get('name')}"
                                    )
                    if choices[0].get("finish_reason") == "tool_calls":
                        finish_reason_tool_calls = True

        # Check: finish_reason is valid
        if tool_calls_found:
            assert finish_reason_tool_calls, "response"
    else:
        # Check: response behavior is valid
        response_json = response.json()
        choices = response_json.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})

            if message.get("tool_calls") or choice.get("finish_reason") == "tool_calls":
                assert choice.get("finish_reason") == "tool_calls", (
                    f"finish_reason should be tool_calls, got {choice.get('finish_reason')}"
                )

                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        assert "id" in tc, "response"
                        assert tc.get("type") == "function", "response"
                        assert "function" in tc, "response"
                        func = tc.get("function", {})
                        assert "name" in func, "function should contain name"
                        assert "arguments" in func, "response"
                        assert func.get("name") in expected_tools, (
                            f"Function name should be one of {expected_tools}, got {func.get('name')}"
                        )
                        args = json.loads(func.get("arguments", "{}"))
                        assert isinstance(args, dict), "response"


def _assert_invalid_parallel_tool_calls_response(response):
    """Validate engines that either reject or coerce invalid boolean values."""
    if assertion.has_error_code(response):
        assertion.assert_error_code_400(response)
        return

    assertion.assert_status_code_200(response)
    response_json = response.json()
    choices = response_json.get("choices", [])
    if choices:
        finish_reason = choices[0].get("finish_reason")
        if finish_reason == "tool_calls":
            _validate_tool_calls_structure(response, stream=False)
        else:
            assertion.assert_finish_reason_valid(finish_reason)


def test_parallel_tool_calls_string_value_non_stream(api_client, request):
    """Test parallel tool calls string value non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": "true",  # Check: response behavior is valid
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: invalid value is rejected or safely coerced
    _assert_invalid_parallel_tool_calls_response(response)


def test_parallel_tool_calls_string_value_stream(api_client, request):
    """Test parallel tool calls string value stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": "false",  # Check: response behavior is valid
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_parallel_tool_calls_integer_value_non_stream(api_client, request):
    """Test parallel tool calls integer value non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": 1,  # Check: response behavior is valid
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: invalid value is rejected or safely coerced
    _assert_invalid_parallel_tool_calls_response(response)


def test_parallel_tool_calls_integer_value_stream(api_client, request):
    """Test parallel tool calls integer value stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": 0,  # Check: response behavior is valid
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_parallel_tool_calls_null_value_non_stream(api_client, request):
    """Test parallel tool calls null value non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": None,
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: finish_reason is valid
    response_json = response.json()
    choices = response_json.get("choices", [])
    if choices:
        finish_reason = choices[0].get("finish_reason")
        if finish_reason == "tool_calls":
            # Check: tool_calls structure is valid
            _validate_tool_calls_structure(response, stream=False)
        else:
            assertion.assert_finish_reason_valid(finish_reason)


def test_parallel_tool_calls_null_value_stream(api_client, request):
    """Test parallel tool calls null value stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": None,
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    if finish_reason == "tool_calls":
        # Check: tool_calls structure is valid
        _validate_tool_calls_structure(response, stream=True)
    else:
        assertion.assert_finish_reason_valid(finish_reason)


def test_parallel_tool_calls_array_value_non_stream(api_client, request):
    """Test parallel tool calls array value non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": [True],  # Check: response behavior is valid
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_parallel_tool_calls_array_value_stream(api_client, request):
    """Test parallel tool calls array value stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": [],  # Check: response behavior is valid
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_parallel_tool_calls_object_value_non_stream(api_client, request):
    """Test parallel tool calls object value non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": {"enabled": True},  # Check: response behavior is valid
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_parallel_tool_calls_object_value_stream(api_client, request):
    """Test parallel tool calls object value stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": {},  # Check: response behavior is valid
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_parallel_tool_calls_float_value_non_stream(api_client, request):
    """Test parallel tool calls float value non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": 1.0,  # Check: response behavior is valid
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: invalid value is rejected or safely coerced
    _assert_invalid_parallel_tool_calls_response(response)


def test_parallel_tool_calls_float_value_stream(api_client, request):
    """Test parallel tool calls float value stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": 0.5,  # Check: response behavior is valid
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_parallel_tool_calls_empty_string_non_stream(api_client, request):
    """Test parallel tool calls empty string non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": "",  # Check: response behavior is valid
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_parallel_tool_calls_empty_string_stream(api_client, request):
    """Test parallel tool calls empty string stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": "",  # Check: response behavior is valid
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)
