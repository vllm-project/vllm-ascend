# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import Mock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.abstract_parser import DelegatingParser

from vllm_ascend.models.iquest_q1.reasoning_parser import (
    IQuestQ1ReasoningParser,
)
from vllm_ascend.models.iquest_q1.tool_parser import IQuestQ1ToolParser

CALL_START = "<iquest_tool_call>"
CALL_END = "</iquest_tool_call>"
PARAMETERS = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "count": {"type": "integer"},
        "options": {"type": "object"},
    },
}


@pytest.fixture
def tokenizer():
    return Mock(
        get_vocab=lambda: {
            "<think>": 100,
            "</think>": 101,
            "<|iquest_assistant|>": 102,
            CALL_START: 103,
            CALL_END: 104,
        }
    )


def _request(tool_choice="auto"):
    return ChatCompletionRequest(
        model="m1",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {"name": "run", "parameters": PARAMETERS},
            }
        ],
        tool_choice=tool_choice,
    )


def _call(name="run", **arguments):
    body = name + "\n"
    for key, value in arguments.items():
        body += f"<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n"
    return f"{CALL_START}{body}{CALL_END}"


def _stream(parser, chunks, request):
    text = ""
    for chunk in chunks:
        result = parser.extract_tool_calls_streaming(text, text + chunk, chunk, [], [], [], request)
        if result is not None:
            yield result
        text += chunk


@pytest.mark.parametrize("request_kind", ["chat", "responses", "namespace"])
def test_schema_string_values_preserved_across_request_types(tokenizer, request_kind):
    request = _request()
    name = "run"
    if request_kind != "chat":
        tool = {
            "type": "function",
            "name": name,
            "parameters": PARAMETERS,
            "strict": False,
        }
        if request_kind == "namespace":
            tool = {
                "type": "namespace",
                "name": "agent",
                "description": "Agent tools",
                "tools": [tool],
            }
            name = "agent__run"
        request = ResponsesRequest(model="m1", input="test", tools=[tool])
    result = IQuestQ1ToolParser(tokenizer).extract_tool_calls(
        _call(name, text=" 123 \n", count="3", options='{"enabled":true}'), request
    )
    assert result.tools_called
    assert result.content is None
    assert result.tool_calls[0].function.name == name
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "text": " 123 \n",
        "count": 3,
        "options": {"enabled": True},
    }


@pytest.mark.parametrize("chunk_size", [1, 3, 23, 4096])
def test_streaming_split_tags_preserve_parallel_calls_and_surrounding_content(tokenizer, chunk_size):
    parser = IQuestQ1ToolParser(tokenizer)
    text = "before" + _call(text="123") + "between" + _call() + "after"
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    messages = list(_stream(parser, chunks, _request()))
    assert "".join(message.content or "" for message in messages) == "beforebetweenafter"
    calls = [call for message in messages for call in message.tool_calls]
    assert [call.index for call in calls] == [0, 1]
    assert len({call.id for call in calls}) == 2
    assert [call.function.name for call in calls] == ["run", "run"]
    assert [json.loads(call.function.arguments) for call in calls] == [
        {"text": "123"},
        {},
    ]
    assert parser.get_remaining_unstreamed_args() == ""
    result = parser.extract_tool_calls(text, _request())
    assert result.content == "beforebetweenafter"
    assert [call.function.arguments for call in result.tool_calls] == [call.function.arguments for call in calls]


@pytest.mark.parametrize("streaming", [False, True])
def test_malformed_call_preserved_as_content(tokenizer, streaming):
    text = CALL_START + "run<arg_key>text</arg_key><arg_value>broken" + CALL_END
    parser = IQuestQ1ToolParser(tokenizer)
    if streaming:
        results = list(_stream(parser, list(text), _request()))
        assert "".join(result.content or "" for result in results) == text
        assert not any(result.tool_calls for result in results)
    else:
        result = parser.extract_tool_calls(text, _request())
        assert not result.tools_called
        assert result.content == text


@pytest.mark.parametrize("without_tools", [False, True])
def test_disabled_tool_parsing_passes_xml_through(tokenizer, without_tools):
    request = _request("none")
    if without_tools:
        request.tools = None
    parser = IQuestQ1ToolParser(tokenizer)
    assert parser.adjust_request(request).skip_special_tokens
    text = _call(text="123")
    assert parser.extract_tool_calls(text, request).content == text
    assert "".join(result.content or "" for result in _stream(parser, list(text), request)) == text


class _IquestParser(DelegatingParser):
    reasoning_parser_cls = IQuestQ1ReasoningParser
    tool_parser_cls = IQuestQ1ToolParser


@pytest.mark.parametrize(
    "tool_choice",
    ["auto", "required", {"type": "function", "function": {"name": "run"}}],
)
def test_serving_parses_xml_for_auto_required_and_named_tools(tokenizer, tool_choice):
    parser = _IquestParser(tokenizer)
    request = parser.adjust_request(_request(tool_choice))
    assert request.skip_special_tokens is False
    assert request.structured_outputs is None
    reasoning, content, calls = parser.parse(
        "reasoning</think>" + _call(text="123") + "after",
        request,
        enable_auto_tools=True,
    )
    assert reasoning == "reasoning"
    assert content == "after"
    assert len(calls) == 1
    assert calls[0].name == "run"
    assert json.loads(calls[0].arguments) == {"text": "123"}


@pytest.mark.parametrize(
    "tool_choice",
    ["auto", "required", {"type": "function", "function": {"name": "run"}}],
)
def test_serving_mtp_delta_crosses_reasoning_end_into_tool_call(tokenizer, tool_choice):
    parser = _IquestParser(tokenizer)
    request = parser.adjust_request(_request(tool_choice))
    chunks = [
        ("reasoning", [1]),
        (" tail</think>before" + CALL_START, [2, 101, 3, 103]),
        ("run<arg_key>text</arg_key><arg_value>123</arg_value>", [4, 5]),
        (CALL_END + "after", [104, 6]),
    ]
    results = []
    for index, (text, ids) in enumerate(chunks):
        result = parser.parse_delta(
            text,
            ids,
            request,
            prompt_token_ids=[101, 7, 102, 100],
            finished=index == len(chunks) - 1,
        )
        if result is not None:
            results.append(result)
    assert "".join(result.reasoning or "" for result in results) == "reasoning tail"
    assert "".join(result.content or "" for result in results) == "beforeafter"
    calls = [call for result in results for call in result.tool_calls]
    assert len(calls) == 1
    assert calls[0].function.name == "run"
    assert json.loads(calls[0].function.arguments) == {"text": "123"}
