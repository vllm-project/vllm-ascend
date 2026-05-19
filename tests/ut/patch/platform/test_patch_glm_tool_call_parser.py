# SPDX-License-Identifier: Apache-2.0

import json

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

from vllm_ascend.patch.platform import (
    patch_glm_tool_call_parser as glm_tool_call_patch,
)


class FakeTokenizer:
    def get_vocab(self):
        return {
            "<tool_call>": 1,
            "</tool_call>": 2,
            "<arg_key>": 3,
            "</arg_key>": 4,
            "<arg_value>": 5,
            "</arg_value>": 6,
        }


def _reset_streaming_state(parser):
    parser._buffer = ""
    parser._in_tool_call = False
    parser.current_tool_name_sent = False
    parser._current_tool_name = None
    parser._pending_key = None
    parser._streaming_string_value = False
    parser.prev_tool_call_arr = []
    parser.current_tool_id = -1
    parser.streamed_args_for_tool = []
    parser._tool_call_ids = []
    parser._args_started = []
    parser._args_closed = []
    parser._seen_keys = []


def test_create_remaining_args_delta_uses_fallback_metadata_for_args_only_delta():
    original_delta = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments='{"files":['),
            )
        ]
    )

    result = OpenAIServingChat._create_remaining_args_delta(
        original_delta,
        '{"files":[{"filepath":"HumanEval-X/README.md"}]}',
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
    assert tc.function.arguments == ('{"files":[{"filepath":"HumanEval-X/README.md"}]}')


def test_create_remaining_args_delta_uses_fallback_over_original_delta():
    # _create_remaining_args_delta ignores original_delta metadata and uses
    # the explicit fallback_* parameters instead.  The caller is responsible
    # for passing non-None fallback values only for the first chunk of a
    # tool call (when the header has not yet been streamed).
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
        fallback_tool_call_id="call_fallback",
        fallback_tool_call_type="function",
        fallback_tool_call_name="fallback_name",
    )

    tc = result.tool_calls[0]
    assert tc.id == "call_fallback"
    assert tc.type == "function"
    assert tc.function.name == "fallback_name"
    assert tc.function.arguments == "]}"


def test_glm_streaming_final_chunk_emits_inline_string_value():
    parser = Glm4MoeModelToolParser(FakeTokenizer())
    _reset_streaming_state(parser)

    request = ChatCompletionRequest(
        model="zai-org/GLM-4.7",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "builtin_get_problems",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {"type": "string"},
                        },
                    },
                },
            }
        ],
    )

    chunks = [
        "<tool_call>",
        "builtin_get_problems\n",
        "<arg_key>filepath</arg_key>",
        "<arg_value>pong.py</arg_value></tool_call>",
    ]

    last_tool_delta = None
    for chunk in chunks:
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if result is not None and result.tool_calls:
            last_tool_delta = result

    assert last_tool_delta is not None
    assert last_tool_delta.tool_calls[0].function.arguments == '{"filepath":"pong.py"}'
    assert parser.streamed_args_for_tool == ['{"filepath":"pong.py"}']
    assert parser.prev_tool_call_arr == [
        {
            "name": "builtin_get_problems",
            "arguments": {"filepath": "pong.py"},
        }
    ]


def test_glm47_streaming_delta_serializes_tool_call_fields():
    parser = Glm47MoeModelToolParser(FakeTokenizer())
    _reset_streaming_state(parser)

    request = ChatCompletionRequest(
        model="GLM-5",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "builtin_get_problems",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {"type": "string"},
                        },
                    },
                },
            }
        ],
    )

    chunks = [
        "<tool_call>",
        "builtin_get_problems\n",
        "<arg_key>filepath</arg_key>",
        "<arg_value>pong.py</arg_value></tool_call>",
    ]

    serialized_deltas = []
    for chunk in chunks:
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if result is None:
            continue

        choice = ChatCompletionResponseStreamChoice(
            index=0,
            delta=result,
            logprobs=None,
            finish_reason=None,
        )
        response = ChatCompletionStreamResponse(
            id="chatcmpl-test",
            created=0,
            model="GLM-5",
            choices=[choice],
        )
        serialized_deltas.append(response.model_dump(exclude_unset=True)["choices"][0]["delta"])

    assert len(serialized_deltas) == 2
    assert serialized_deltas[0]["tool_calls"][0]["type"] == "function"
    assert serialized_deltas[0]["tool_calls"][0]["function"]["name"] == "builtin_get_problems"
    assert serialized_deltas[-1] != {}
    assert serialized_deltas[-1]["tool_calls"][0]["index"] == 0
    assert serialized_deltas[-1]["tool_calls"][0]["function"]["arguments"] == '{"filepath":"pong.py"}'


def test_terminal_argument_chunk_is_split_before_finish_chunk():
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "GLM-4.7",
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

    chunks = glm_tool_call_patch._split_terminal_tool_arg_chunk(f"data: {json.dumps(chunk)}\n\n")

    assert len(chunks) == 2
    arg_payload = json.loads(chunks[0].removeprefix("data: ").removesuffix("\n\n"))
    finish_payload = json.loads(chunks[1].removeprefix("data: ").removesuffix("\n\n"))

    assert arg_payload["choices"][0]["finish_reason"] is None
    assert arg_payload["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '"pong.py"}'
    assert finish_payload["choices"][0]["finish_reason"] == "tool_calls"
    assert finish_payload["choices"][0]["delta"] == {}
