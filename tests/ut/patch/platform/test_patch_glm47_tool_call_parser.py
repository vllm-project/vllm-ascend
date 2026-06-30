# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest

from vllm_ascend.utils import vllm_version_is

pytestmark = pytest.mark.skipif(
    not vllm_version_is("0.23.0"),
    reason="upstream vLLM renamed _extract_tool_call_regions",
)

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: E402
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


def _parse_delta(parser, *args, finished=False, **kwargs):
    return parser.parse_delta(*args, finished=finished, **kwargs)


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
