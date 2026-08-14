# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.qwen3_engine_reasoning_parser import (
    Qwen3ParserReasoningAdapter,
)
from vllm.tool_parsers import ToolParserManager

from vllm_ascend.patch.dots_tool_parser import DotsToolParser

MOCK_TOKENIZER = MagicMock()
MOCK_TOKENIZER.get_vocab.return_value = {"<think>": 1, "</think>": 2}


def _make_tool(name: str) -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "description": "test tool",
            "parameters": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
            },
        },
    )


def _make_request(tools=None) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[], model="test", tools=tools)


def test_dots_parsers_are_registered():
    assert ReasoningParserManager.get_reasoning_parser("dots") is Qwen3ParserReasoningAdapter
    assert ToolParserManager.get_tool_parser("dots") is DotsToolParser


def test_dots_tool_parser_converts_schema_types():
    tool = _make_tool("search")
    parser = DotsToolParser(MOCK_TOKENIZER, tools=[tool])
    text = (
        "visible<dots_function_call>"
        '<invoke name="search"><parameter name="count">3</parameter></invoke>'
        "</dots_function_call>"
    )

    result = parser.extract_tool_calls(text, _make_request([tool]))

    assert result.content == "visible"
    assert result.tools_called
    assert result.tool_calls[0].function.name == "search"
    assert json.loads(result.tool_calls[0].function.arguments) == {"count": 3}


def test_dots_tool_parser_buffers_partial_stream_marker():
    tool = _make_tool("search")
    parser = DotsToolParser(MOCK_TOKENIZER, tools=[tool])
    request = _make_request([tool])
    chunks = [
        "visible<dots_func",
        ('tion_call><invoke name="search"><parameter name="count">3</parameter></invoke></dots_function_call>'),
    ]

    results = []
    previous_text = ""
    for chunk in chunks:
        current_text = previous_text + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=request,
        )
        previous_text = current_text
        if delta is not None:
            results.append(delta)

    assert "".join(result.content or "" for result in results) == "visible"
    calls = [call for result in results for call in result.tool_calls]
    assert len(calls) == 1
    assert json.loads(calls[0].function.arguments) == {"count": 3}
