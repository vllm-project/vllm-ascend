# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion, completion_request

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get weather info",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    }
]


def test_messages_api_basic_chat(api_client):
    """Test case 1: Basic text chat via /v1/messages endpoint."""
    response = completion_request.send_messages_request(
        api_client,
        prompt="Hello!",
        max_tokens=100,
    )
    assertion.assert_messages_response_has_text(response)


def test_messages_api_tool_calling(api_client):
    """Test case 2: Tool calling via /v1/messages endpoint."""
    response = completion_request.send_messages_request(
        api_client,
        prompt="What is the weather in Shanghai?",
        max_tokens=100,
        tools=TOOLS,
    )
    assertion.assert_messages_response_has_text_or_tool_use(response)


def test_messages_api_streaming(api_client):
    """Test case 3: Streaming response via /v1/messages endpoint."""
    response = completion_request.send_messages_request_stream(
        api_client,
        prompt="Count to 5.",
        max_tokens=50,
    )
    assertion.assert_messages_stream_response_success(response)


@pytest.mark.parametrize(
    "prompt,max_tokens",
    [
        ("Hello!", 50),
        ("What is AI?", 100),
        ("Tell me a joke", 150),
    ],
)
def test_messages_api_various_prompts(api_client, prompt, max_tokens):
    """Test messages API with various prompts and max_tokens."""
    response = completion_request.send_messages_request(
        api_client,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    assertion.assert_messages_response_has_text(response)
