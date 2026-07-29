#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import asyncio
import json
from types import SimpleNamespace

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage, ErrorResponse
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.parser import ParserManager
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

from vllm_ascend.patch.platform import patch_kimi_k3_parsers as parser_patch
from vllm_ascend.patch.platform.kimi_k3_parser import KimiK3Parser
from vllm_ascend.patch.platform.kimi_k3_protocol import (
    CLOSE_TOKEN,
    OPEN_TOKEN,
    RESPONSE_END,
    RESPONSE_START,
    SEP_TOKEN,
    THINK_END,
    THINK_START,
    TOOLS_END,
    TOOLS_START,
)
from vllm_ascend.patch.platform.kimi_k3_reasoning_parser import (
    KimiK3ReasoningParser,
)
from vllm_ascend.patch.platform.kimi_k3_tool_parser import (
    KimiK3ToolParser,
)


class DummyTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del add_special_tokens
        if text == THINK_START:
            return [1, 2, 3]
        if text == THINK_END:
            return [4, 2, 3]
        return [ord(char) for char in text]


TOKENIZER = DummyTokenizer()


def _tools_definition():
    return [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "flag": {"type": "boolean"},
                        "text": {"type": "string"},
                    },
                },
            },
        }
    ]


def _request(
    *,
    tool_choice="auto",
    tools=None,
    **kwargs,
) -> ChatCompletionRequest:
    if tools is None:
        tools = _tools_definition()
    return ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "calculate"}],
        tools=tools,
        tool_choice=tool_choice,
        **kwargs,
    )


def _argument(key: str, value_type: str, value: str) -> str:
    return f'{OPEN_TOKEN}argument key="{key}" type="{value_type}"{SEP_TOKEN}{value}{CLOSE_TOKEN}argument{SEP_TOKEN}'


def _call(*arguments: str) -> str:
    return f'{OPEN_TOKEN}call tool="calc" index="1"{SEP_TOKEN}{"".join(arguments)}{CLOSE_TOKEN}call{SEP_TOKEN}'


def _response(content: str) -> str:
    return f"{RESPONSE_START}{content}{RESPONSE_END}"


def _tools(*calls: str, close: bool = True) -> str:
    suffix = TOOLS_END if close else ""
    return f"{TOOLS_START}{''.join(calls)}{suffix}"


def _parser_cls(
    *,
    reasoning: bool = True,
    tools: bool = True,
) -> type[KimiK3Parser]:
    parser_cls = ParserManager.get_parser(
        tool_parser_name="kimi_k3" if tools else None,
        reasoning_parser_name="kimi_k3" if reasoning else None,
        enable_auto_tools=tools,
    )
    assert parser_cls is not None
    assert issubclass(parser_cls, KimiK3Parser)
    return parser_cls


def _parser(*, thinking: bool = False) -> KimiK3Parser:
    return _parser_cls()(
        TOKENIZER,
        _tools_definition(),
        chat_template_kwargs={"thinking": thinking},
    )


def test_kimi_k3_parsers_are_registered_and_composed_like_upstream():
    assert ReasoningParserManager.get_reasoning_parser("kimi_k3") is KimiK3ReasoningParser
    assert ToolParserManager.get_tool_parser("kimi_k3") is KimiK3ToolParser

    combined = _parser_cls()
    assert combined.reasoning_parser_cls is KimiK3ReasoningParser
    assert combined.tool_parser_cls is KimiK3ToolParser

    reasoning_only = _parser_cls(reasoning=True, tools=False)
    assert reasoning_only.reasoning_parser_cls is KimiK3ReasoningParser
    assert reasoning_only.tool_parser_cls is None


def test_nonstream_accepts_response_without_message_close():
    reasoning, content, calls = _parser().parse(
        _response("answer"),
        _request(),
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content == "answer"
    assert calls is None


def test_nonstream_accepts_tools_close_followed_directly_by_eos():
    output = RESPONSE_END + _tools(
        _call(_argument("x", "number", "1")),
    )

    reasoning, content, calls = _parser().parse(
        output,
        _request(),
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content is None
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "calc"
    assert json.loads(calls[0].arguments) == {"x": 1}


def test_nonstream_preserves_completed_call_when_tools_close_is_truncated():
    output = RESPONSE_END + _tools(
        _call(_argument("x", "number", "1")),
        close=False,
    )

    _, content, calls = _parser().parse(
        output,
        _request(),
        enable_auto_tools=True,
    )

    assert content is None
    assert calls is not None and len(calls) == 1
    assert json.loads(calls[0].arguments) == {"x": 1}


def test_nonstream_truncated_call_does_not_leak_xtml():
    output = RESPONSE_END + TOOLS_START + f'{OPEN_TOKEN}call tool="calc" index="1"'

    reasoning, content, calls = _parser().parse(
        output,
        _request(tool_choice="required"),
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content is None
    assert calls is None


def test_tool_choice_none_strips_xtml_and_suppresses_calls():
    parser = _parser()
    request = _request(tool_choice="none")
    output = _response("answer") + _tools(
        _call(_argument("x", "number", "1")),
    )

    reasoning, content, calls = parser.parse(
        output,
        request,
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content == "answer"
    assert calls == []

    parser = _parser()
    messages: list[DeltaMessage] = []
    chunks = [
        OPEN_TOKEN,
        "response",
        f"{SEP_TOKEN}answer",
        RESPONSE_END,
        _tools(_call(_argument("x", "number", "1"))),
    ]
    for index, chunk in enumerate(chunks, start=1):
        delta = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=[index],
            request=request,
            prompt_token_ids=[99],
            finished=index == len(chunks),
        )
        if delta is not None:
            messages.append(delta)

    streamed_content = "".join(message.content or "" for message in messages)
    assert streamed_content == "answer"
    assert all(not message.tool_calls for message in messages)
    assert "<|" not in streamed_content


def test_typed_arguments_and_whitespace_tolerant_markers():
    parser = KimiK3ToolParser(TOKENIZER)
    output = f"{OPEN_TOKEN} response {SEP_TOKEN}answer{CLOSE_TOKEN} response {SEP_TOKEN}" + _tools(
        _call(
            _argument("x", "number", "1"),
            _argument("flag", "boolean", "true"),
            _argument("text", "string", "raw"),
        )
    )

    extracted = parser.extract_tool_calls(output, _request())

    assert extracted.tools_called is True
    assert extracted.content == "answer"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "x": 1,
        "flag": True,
        "text": "raw",
    }


def test_streaming_split_markers_do_not_leak():
    parser = KimiK3ToolParser(TOKENIZER)
    request = _request()
    previous_text = ""
    previous_ids: list[int] = []
    messages: list[DeltaMessage] = []
    chunks = [
        OPEN_TOKEN,
        "response",
        f"{SEP_TOKEN}Hi",
        OPEN_TOKEN,
        "tools",
        SEP_TOKEN,
        f'{OPEN_TOKEN}call tool="calc" index="1"{SEP_TOKEN}',
        _argument("x", "number", "1"),
        f"{CLOSE_TOKEN}call",
        SEP_TOKEN,
    ]

    for index, chunk in enumerate(chunks, start=1):
        current_text = previous_text + chunk
        current_ids = previous_ids + [index]
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_ids,
            current_token_ids=current_ids,
            delta_token_ids=[index],
            request=request,
        )
        if delta is not None:
            messages.append(delta)
        previous_text = current_text
        previous_ids = current_ids

    content = "".join(message.content or "" for message in messages)
    tool_calls = [tool_call for message in messages for tool_call in (message.tool_calls or [])]
    assert content == "Hi"
    assert "<|" not in content
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "calc"
    assert json.loads(tool_calls[0].function.arguments) == {"x": 1}


def test_auto_tool_choice_streams_through_delegating_parser():
    parser = _parser()
    request = _request()
    output = RESPONSE_END + _tools(
        _call(_argument("x", "number", "1")),
    )
    messages: list[DeltaMessage] = []

    for index, chunk in enumerate(
        (output[:11], output[11:37], output[37:]),
        start=1,
    ):
        delta = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=[index],
            request=request,
            prompt_token_ids=[99],
            finished=index == 3,
        )
        if delta is not None:
            messages.append(delta)

    tool_calls = [tool_call for message in messages for tool_call in (message.tool_calls or [])]
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "calc"
    assert json.loads(tool_calls[0].function.arguments) == {"x": 1}


def test_reasoning_parser_handles_consumed_prefix_and_stale_close():
    parser = KimiK3ReasoningParser(TOKENIZER)
    request = ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "calculate"}],
    )

    reasoning, content = parser.extract_reasoning(
        f"step{THINK_END}{RESPONSE_START}answer",
        request,
    )

    assert reasoning == "step"
    assert content == "answer"
    stale_close = [4, 2, 3]
    current_open = [1, 2, 3]
    assert not parser.is_reasoning_end([*stale_close, *current_open])
    assert parser.is_reasoning_end([*stale_close, *current_open, *stale_close])


def test_remote_decode_d02_glue_bypasses_parser():
    serving = SimpleNamespace()
    captured = {}

    async def original(
        request,
        result_generator,
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        parser,
        mm_token_counts,
    ):
        del (
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            mm_token_counts,
        )
        captured["parser"] = parser
        return "ok"

    setattr(serving, parser_patch._ORIGINAL_CHAT_FULL_ATTR, original)
    request = _request(
        kv_transfer_params={
            "do_remote_decode": True,
            "do_remote_prefill": False,
        }
    )
    parser = _parser()

    result = asyncio.run(
        parser_patch._wrapped_chat_completion_full_generator(
            serving,
            request,
            SimpleNamespace(),
            "request-id",
            "kimi-k3",
            [],
            TOKENIZER,
            SimpleNamespace(),
            parser,
            {},
        )
    )

    assert result == "ok"
    assert captured["parser"] is None


def test_kimi_k3_responses_are_rejected_on_pinned_core():
    serving = object.__new__(OpenAIServingResponses)
    serving.parser = _parser_cls()
    request = ResponsesRequest.model_validate(
        {
            "model": "kimi-k3",
            "input": "test",
        }
    )

    response = asyncio.run(serving._create_responses(request))

    assert isinstance(response, ErrorResponse)
    assert "Chat Completions" in response.error.message
