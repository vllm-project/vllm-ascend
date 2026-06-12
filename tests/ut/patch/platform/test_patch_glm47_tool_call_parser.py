# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.deepseek_v3_reasoning_parser import (
    DeepSeekV3ReasoningWithThinkingParser,
)
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

from vllm_ascend.patch.platform import patch_glm47_tool_call_parser  # noqa: F401

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

    first = parser.parse_delta(
        "Need current time.",
        [2001, 2002],
        request,
        prompt_token_ids=[],
    )
    second = parser.parse_delta(
        "</think><tool_call>get_current_time</tool_call>",
        [154842, 154843, 455, 11075, 3009, 154844],
        request,
    )

    assert first is not None
    assert first.reasoning == "Need current time."
    assert second is not None
    assert second.tool_calls
    assert second.tool_calls[0].function.name == "get_current_time"
    assert json.loads(_collect_tool_args(second.tool_calls)) == {}
