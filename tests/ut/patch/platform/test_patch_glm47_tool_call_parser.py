# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

from vllm_ascend.patch.platform import patch_glm47_tool_call_parser  # noqa: F401

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
ARG_KEY_START = "<arg_key>"
ARG_KEY_END = "</arg_key>"
ARG_VALUE_START = "<arg_value>"
ARG_VALUE_END = "</arg_value>"

_VOCAB = {
    THINK_START: 154841,
    THINK_END: 154842,
    TOOL_CALL_START: 154843,
    TOOL_CALL_END: 154844,
    ARG_KEY_START: 154847,
    ARG_KEY_END: 154848,
    ARG_VALUE_START: 154849,
    ARG_VALUE_END: 154850,
}
_VOCAB_R = {v: k for k, v in _VOCAB.items()}


class _FakeTokenizer:
    eos_token_id = None
    bos_token_id = None
    pad_token_id = None

    def decode(self, token_ids, **kwargs):
        return "".join(_VOCAB_R.get(tid, "") for tid in token_ids)

    def get_vocab(self):
        return dict(_VOCAB)


MOCK_TOKENIZER = _FakeTokenizer()


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


_LEGACY_PARSER = hasattr(Glm47MoeModelToolParser, "_extract_tool_call_regions")


@pytest.mark.skipif(
    not _LEGACY_PARSER,
    reason="Legacy regex-based parser no longer present; "
    "the new state-machine engine handles zero-arg tool calls natively.",
)
def test_glm47_streaming_inline_zero_arg_tool_call_waits_until_complete():
    request = _request()
    parser = Glm47MoeModelToolParser(MOCK_TOKENIZER, request.tools)

    first = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=TOOL_CALL_START + "get",
        delta_text=TOOL_CALL_START + "get",
        previous_token_ids=[],
        current_token_ids=[154843, 455],
        delta_token_ids=[154843, 455],
        request=request,
    )
    assert first is None

    second = parser.extract_tool_calls_streaming(
        previous_text=TOOL_CALL_START + "get",
        current_text=TOOL_CALL_START + "get_current_time" + TOOL_CALL_END,
        delta_text="_current_time" + TOOL_CALL_END,
        previous_token_ids=[154843, 455],
        current_token_ids=[154843, 455, 11075, 3009, 154844],
        delta_token_ids=[11075, 3009, 154844],
        request=request,
    )

    assert second is not None
    assert second.tool_calls
    assert second.tool_calls[0].function.name == "get_current_time"
    assert json.loads(_collect_tool_args(second.tool_calls)) == {}


@pytest.mark.skipif(
    _LEGACY_PARSER,
    reason="Only applicable for the new state-machine engine parser.",
)
def test_glm47_engine_streaming_inline_zero_arg_tool_call():
    request = _request()
    parser = Glm47MoeModelToolParser(MOCK_TOKENIZER, request.tools)

    first = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=TOOL_CALL_START,
        delta_text=TOOL_CALL_START,
        previous_token_ids=[],
        current_token_ids=[154843],
        delta_token_ids=[154843],
        request=request,
    )
    assert first is None

    second = parser.extract_tool_calls_streaming(
        previous_text=TOOL_CALL_START,
        current_text=TOOL_CALL_START + "get_current_time" + TOOL_CALL_END,
        delta_text="get_current_time" + TOOL_CALL_END,
        previous_token_ids=[154843],
        current_token_ids=[154843, 11075, 3009, 154844],
        delta_token_ids=[11075, 3009, 154844],
        request=request,
    )

    assert second is not None
    assert second.tool_calls
    assert second.tool_calls[0].function.name == "get_current_time"
