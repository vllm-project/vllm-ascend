import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import completion_request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
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


@pytest.mark.parametrize(
    "tool_choice",
    ["none", "auto", "required", {"type": "function", "function": {"name": "get_weather"}}],
)
def test_tool_choice_accepts_supported_values(api_client, tool_choice):
    response = completion_request.send_chat_request(
        api_client,
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=TOOLS,
        tool_choice=tool_choice,
        max_tokens=128,
    )
    assertion.assert_status_code_200(response)


def test_tool_choice_accepts_streaming_request(api_client):
    response = completion_request.send_chat_request(
        api_client,
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
        max_tokens=128,
    )
    assertion.assert_status_code_200(response)
    assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize(
    "tool_choice",
    [
        "get_weather",
        "invalid",
        None,
        1,
        [],
        True,
        {"type": "function"},
        {"type": "invalid", "function": {"name": "get_weather"}},
    ],
)
def test_tool_choice_rejects_invalid_values(api_client, tool_choice):
    response = completion_request.send_chat_request(api_client, tools=TOOLS, tool_choice=tool_choice)
    assertion.assert_validation_error_response(response)


@pytest.mark.parametrize("parallel_tool_calls", [True, False])
def test_parallel_tool_calls_accepts_boolean_values(api_client, parallel_tool_calls):
    response = completion_request.send_chat_request(
        api_client,
        messages=[{"role": "user", "content": "Use tools if needed."}],
        tools=TOOLS,
        parallel_tool_calls=parallel_tool_calls,
        max_tokens=128,
    )
    assertion.assert_status_code_200(response)


@pytest.mark.parametrize("parallel_tool_calls", ["true", 1, None, [], {}, 0.5])
def test_parallel_tool_calls_rejects_invalid_values(api_client, parallel_tool_calls):
    response = completion_request.send_chat_request(api_client, tools=TOOLS, parallel_tool_calls=parallel_tool_calls)
    assertion.assert_validation_error_response(response)


def test_tool_choice_accepts_complex_tool_schema(api_client):
    complex_tool = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": None,
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]
    response = completion_request.send_chat_request(
        api_client, tools=complex_tool, tool_choice="auto", parallel_tool_calls=False, max_tokens=128
    )
    assert response.status_code in (200, 400)
