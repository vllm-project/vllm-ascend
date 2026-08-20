# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser

from vllm_ascend.patch.platform import (
    patch_glm_tool_call_streaming as glm_streaming_patch,
)


class _Tokenizer:
    def get_vocab(self):
        return {"<tool_call>": 1, "</tool_call>": 2}


def _stream_tool_arguments(schema, value_chunks, tool_name="run", argument_name="value"):
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name=tool_name,
                parameters={
                    "type": "object",
                    "properties": {argument_name: schema},
                },
            ),
        ),
    ]
    request = ChatCompletionRequest(model="test", messages=[], tools=tools)
    parser = Glm4MoeModelToolParser(_Tokenizer(), tools=tools)
    chunks = [
        "<tool_call>",
        tool_name,
        "<arg_key>",
        argument_name,
        "</arg_key>",
        "<arg_value>",
        *value_chunks,
        "</arg_value>",
        "</tool_call>",
    ]

    current_text = ""
    argument_fragments = []
    for chunk in chunks:
        previous_text = current_text
        current_text += chunk
        result = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if result is not None and result.tool_calls:
            for tool_call in result.tool_calls:
                function = tool_call.function
                if isinstance(function, dict):
                    arguments = function.get("arguments")
                else:
                    arguments = function.arguments
                if arguments:
                    argument_fragments.append(arguments)

    return "".join(argument_fragments)


def test_issue_12261_nullable_string_schema_streams_valid_json():
    schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}

    arguments = _stream_tool_arguments(
        schema,
        ["P", "ura", "90"],
        tool_name="start_app",
        argument_name="hvd",
    )

    assert json.loads(arguments) == {"hvd": "Pura90"}


@pytest.mark.parametrize(
    "schema",
    [
        {},
        True,
        {"type": ["string", "null"]},
        {"oneOf": [{"type": "string"}, {"type": "null"}]},
    ],
)
def test_unknown_and_nullable_string_schemas_stream_as_strings(schema):
    arguments = _stream_tool_arguments(schema, ["P", "ura", "90"])

    assert json.loads(arguments) == {"value": "Pura90"}


@pytest.mark.parametrize(
    ("schema", "value_chunks", "expected"),
    [
        ({"type": "object"}, ["{", '"x"', ":", "1", "}"], {"x": 1}),
        ({"type": "number"}, ["1", "e", "3"], 1000.0),
        (
            {"oneOf": [{"type": "integer"}, {"type": "null"}]},
            ["1", "2", "3"],
            123,
        ),
    ],
)
def test_confirmed_non_string_values_wait_for_complete_serialization(schema, value_chunks, expected):
    arguments = _stream_tool_arguments(schema, value_chunks)

    assert json.loads(arguments) == {"value": expected}


def test_remaining_args_delta_omits_metadata_by_default():
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
    assert tc.id is None
    assert tc.type is None
    assert tc.function.name is None
    assert tc.function.arguments == "]}"
    serialized = tc.model_dump(exclude_unset=True)
    assert "id" not in serialized
    assert "type" not in serialized
    assert "name" not in serialized["function"]


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
