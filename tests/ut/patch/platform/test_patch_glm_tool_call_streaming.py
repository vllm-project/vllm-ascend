# SPDX-License-Identifier: Apache-2.0

import json

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)

from vllm_ascend.patch.platform import (
    patch_glm_tool_call_streaming as glm_streaming_patch,
)


def test_remaining_args_delta_preserves_metadata_by_default():
    original_delta = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                id="call_current",
                type="function",
                function=DeltaFunctionCall(
                    name="current_name",
                    arguments='{"files":[',
                ),
            )
        ]
    )

    result = OpenAIServingChat._create_remaining_args_delta(
        original_delta,
        "]}",
        0,
    )

    tc = result.tool_calls[0]
    assert tc.index == 0
    assert tc.id == "call_current"
    assert tc.type == "function"
    assert tc.function.name == "current_name"
    assert tc.function.arguments == "]}"
    serialized = tc.model_dump(exclude_unset=True)
    assert serialized["id"] == "call_current"
    assert serialized["type"] == "function"
    assert serialized["function"]["name"] == "current_name"


def test_empty_remaining_args_delta_keeps_original_delta():
    original_delta = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                id="call_current",
                type="function",
                function=DeltaFunctionCall(
                    name="current_name",
                    arguments="",
                ),
            ),
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments="{}"),
            ),
        ]
    )

    result = OpenAIServingChat._create_remaining_args_delta(
        original_delta,
        "",
        0,
    )

    assert result is original_delta
    assert result.tool_calls[0].function.name == "current_name"
    assert result.tool_calls[1].function.arguments == "{}"


def test_remaining_args_delta_uses_explicit_fallback_metadata():
    result = OpenAIServingChat._create_remaining_args_delta(
        DeltaMessage(),
        '{"filepath":"pong.py"}',
        0,
        fallback_tool_call_id="call_files",
        fallback_tool_call_type="function",
        fallback_tool_call_name="builtin_read_many_files",
    )

    tc = result.tool_calls[0]
    assert tc.index == 0
    assert tc.id == "call_files"
    assert tc.type == "function"
    assert tc.function.name == "builtin_read_many_files"
    assert tc.function.arguments == '{"filepath":"pong.py"}'


def test_terminal_argument_chunk_is_split_before_finish_chunk():
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "GLM-5",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": '"pong.py"}',
                            },
                        }
                    ]
                },
                "finish_reason": "tool_calls",
                "stop_reason": None,
            }
        ],
    }

    chunks = glm_streaming_patch._split_terminal_tool_arg_chunk(f"data: {json.dumps(chunk)}\n\n")

    assert len(chunks) == 2
    arg_payload = json.loads(chunks[0].removeprefix("data: ").removesuffix("\n\n"))
    finish_payload = json.loads(chunks[1].removeprefix("data: ").removesuffix("\n\n"))

    arg_choice = arg_payload["choices"][0]
    assert arg_choice["finish_reason"] is None
    assert arg_choice["stop_reason"] is None
    assert arg_choice["delta"]["tool_calls"][0]["function"]["arguments"] == '"pong.py"}'

    finish_choice = finish_payload["choices"][0]
    assert finish_choice["finish_reason"] == "tool_calls"
    assert finish_choice["delta"] == {}


def test_non_terminal_and_done_chunks_are_not_split():
    content = 'data: {"choices":[]}\n\n'
    done = "data: [DONE]\n\n"

    assert glm_streaming_patch._split_terminal_tool_arg_chunk(content) == [content]
    assert glm_streaming_patch._split_terminal_tool_arg_chunk(done) == [done]
