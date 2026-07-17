# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest

from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.23.0"):
    pytest.skip(
        "upstream vLLM renamed _extract_tool_call_regions",
        allow_module_level=True,
    )

from vllm.entrypoints.openai.chat_completion.protocol import (  # noqa: E402
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat  # noqa: E402

# vLLM main removed the ``_WrappedParser`` helper; the base ``Parser``
# already instantiates from ``reasoning_parser_cls`` / ``tool_parser_cls``
# class attributes, so a thin ``DelegatingParser`` subclass is equivalent.
from vllm.parser.abstract_parser import DelegatingParser  # type: ignore[import-not-found]  # noqa: E402
from vllm.reasoning.deepseek_v3_reasoning_parser import (  # noqa: E402
    DeepSeekV3ReasoningWithThinkingParser,
)
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser  # noqa: E402

from vllm_ascend.patch.platform import patch_glm47_tool_call_parser  # noqa: F401, E402


class _WrappedParser(DelegatingParser):
    pass


MOCK_TOKENIZER = MagicMock()
MOCK_TOKENIZER.get_vocab.return_value = {
    "<think>": 154841,
    "</think>": 154842,
    "<tool_call>": 154843,
    "</tool_call>": 154844,
    "<arg_key>": 154847,
    "</arg_key>": 154848,
    "<arg_value>": 154849,
    "</arg_value>": 154850,
}


def _request():
    return ChatCompletionRequest(
        model="glm5",
        messages=[{"role": "user", "content": "What time is it?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current date and time",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
        ],
        tool_choice="auto",
    )


def _collect_tool_args(tool_calls):
    return "".join(tc.function.arguments for tc in tool_calls if tc.function.arguments)


def _stream_tool_arguments(properties, value_chunks):
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="run",
                parameters={
                    "type": "object",
                    "properties": properties,
                },
            ),
        ),
    ]
    request = ChatCompletionRequest(model="test", messages=[], tools=tools)
    parser = Glm47MoeModelToolParser(MOCK_TOKENIZER, tools=tools)
    chunks = [
        "<tool_call>",
        "run",
        "<arg_key>",
        "value",
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
            argument_fragments.extend(
                tool_call.function.arguments for tool_call in result.tool_calls if tool_call.function.arguments
            )

    return "".join(argument_fragments)


def _parse_delta(parser, *args, finished=False, **kwargs):
    return parser.parse_delta(*args, finished=finished, **kwargs)


def test_glm47_nullable_string_schema_streams_valid_json():
    properties = {
        "value": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        }
    }

    arguments = _stream_tool_arguments(properties, ["P", "ura", "90"])

    assert json.loads(arguments) == {"value": "Pura90"}


def test_glm47_ambiguous_string_schema_streams_as_string():
    properties = {
        "value": {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
            ]
        }
    }

    arguments = _stream_tool_arguments(properties, ["P", "ura", "90"])

    assert json.loads(arguments) == {"value": "Pura90"}


@pytest.mark.parametrize("properties", [None, True])
def test_glm47_unknown_properties_stream_as_strings(properties):
    arguments = _stream_tool_arguments(properties, ["P", "ura", "90"])

    assert json.loads(arguments) == {"value": "Pura90"}


def test_glm47_confirmed_number_waits_for_complete_serialization():
    arguments = _stream_tool_arguments(
        {"value": {"type": "number"}},
        ["1", "e", "3"],
    )

    assert json.loads(arguments) == {"value": 1000.0}


def test_glm47_streaming_inline_zero_arg_tool_call_waits_until_complete():
    request = _request()
    parser = Glm47MoeModelToolParser(MOCK_TOKENIZER, request.tools)

    first = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="<tool_call>get",
        delta_text="<tool_call>get",
        previous_token_ids=[],
        current_token_ids=[154843, 455],
        delta_token_ids=[154843, 455],
        request=request,
    )
    assert first is None

    second = parser.extract_tool_calls_streaming(
        previous_text="<tool_call>get",
        current_text="<tool_call>get_current_time</tool_call>",
        delta_text="_current_time</tool_call>",
        previous_token_ids=[154843, 455],
        current_token_ids=[154843, 455, 11075, 3009, 154844],
        delta_token_ids=[11075, 3009, 154844],
        request=request,
    )

    assert second is not None
    assert second.tool_calls
    assert second.tool_calls[0].function.name == "get_current_time"
    assert json.loads(_collect_tool_args(second.tool_calls)) == {}

    finished = OpenAIServingChat._create_remaining_args_delta(second, "", 0)
    assert finished.tool_calls[0].function.name == "get_current_time"
    assert json.loads(_collect_tool_args(finished.tool_calls)) == {}


def test_glm45_reasoning_glm47_streaming_inline_zero_arg_tool_call():
    request = _request()
    _WrappedParser.reasoning_parser_cls = DeepSeekV3ReasoningWithThinkingParser
    _WrappedParser.tool_parser_cls = Glm47MoeModelToolParser
    parser = _WrappedParser(MOCK_TOKENIZER, request.tools)

    first = _parse_delta(
        parser,
        "Need current time.",
        [2001, 2002],
        request,
        prompt_token_ids=[],
        finished=False,
    )
    second = _parse_delta(
        parser,
        "</think><tool_call>get_current_time</tool_call>",
        [154842, 154843, 455, 11075, 3009, 154844],
        request,
        finished=True,
    )

    assert first is not None
    assert first.reasoning == "Need current time."
    assert second is not None
    assert second.tool_calls
    assert second.tool_calls[0].function.name == "get_current_time"
    assert json.loads(_collect_tool_args(second.tool_calls)) == {}
