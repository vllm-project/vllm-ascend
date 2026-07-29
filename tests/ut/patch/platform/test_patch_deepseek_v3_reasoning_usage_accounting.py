# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.deepseek_v3_reasoning_parser import (
    DeepSeekV3ReasoningParser,
    DeepSeekV3ReasoningWithThinkingParser,
)
from vllm.reasoning.nemotron_v3_reasoning_parser import NemotronV3ReasoningParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser

from vllm_ascend.patch.platform import (
    patch_deepseek_v3_reasoning_usage_accounting as reasoning_usage_patch,  # noqa: F401
)
from vllm_ascend.patch.platform import (
    patch_minimax_usage_accounting as usage_patch,
)


class FakeTokenizer:
    def get_vocab(self):
        return {
            "<think>": 1,
            "</think>": 2,
        }


PARSER_CASES = [
    pytest.param("deepseek_v3", {"thinking": True}, id="deepseek-v3"),
    pytest.param("deepseek_v4", {"thinking": True}, id="deepseek-v4"),
    pytest.param("glm45", None, id="glm45"),
    pytest.param("holo2", None, id="holo2"),
]


def _reasoning_parser(parser_name, chat_template_kwargs=None):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    kwargs = {}
    if chat_template_kwargs is not None:
        kwargs["chat_template_kwargs"] = chat_template_kwargs
    return parser_cls(FakeTokenizer(), **kwargs)


def test_reasoning_parser_registration_is_unchanged():
    assert ReasoningParserManager.get_reasoning_parser("deepseek_v3") is DeepSeekV3ReasoningParser
    assert ReasoningParserManager.get_reasoning_parser("deepseek_v4") is DeepSeekV3ReasoningParser
    assert ReasoningParserManager.get_reasoning_parser("glm45") is DeepSeekV3ReasoningWithThinkingParser
    assert ReasoningParserManager.get_reasoning_parser("holo2") is DeepSeekV3ReasoningWithThinkingParser


@pytest.mark.parametrize(("parser_name", "chat_template_kwargs"), PARSER_CASES)
@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        pytest.param([1, 10, 11, 2, 20], 2, id="explicit-start"),
        pytest.param([99, 1, 10, 11, 2, 20], 2, id="prefix-before-start"),
        pytest.param([10, 11, 2, 20], 2, id="implicit-start"),
        pytest.param([10, 11], 2, id="truncated-reasoning"),
        pytest.param([2, 20], 0, id="end-token-first"),
        pytest.param([], 0, id="empty-output"),
    ],
)
def test_reasoning_token_count(
    parser_name,
    chat_template_kwargs,
    token_ids,
    expected,
):
    parser = _reasoning_parser(parser_name, chat_template_kwargs)

    assert parser.count_reasoning_tokens(token_ids) == expected


@pytest.mark.parametrize("parser_name", ["deepseek_v3", "deepseek_v4"])
@pytest.mark.parametrize("chat_template_kwargs", [{"thinking": False}, {"enable_thinking": False}])
def test_reasoning_tokens_when_thinking_is_disabled(parser_name, chat_template_kwargs):
    parser = _reasoning_parser(parser_name, chat_template_kwargs)

    assert parser.count_reasoning_tokens([10, 11, 2, 20]) == 0


@pytest.mark.parametrize("parser_name", ["glm45", "holo2"])
@pytest.mark.parametrize(
    "chat_template_kwargs",
    [
        {"thinking": False, "enable_thinking": False},
        {"thinking": False, "enable_thinking": None},
        {"thinking": None, "enable_thinking": False},
    ],
)
def test_reasoning_tokens_when_default_thinking_is_disabled(parser_name, chat_template_kwargs):
    parser = _reasoning_parser(parser_name, chat_template_kwargs)

    assert parser.count_reasoning_tokens([10, 11, 2, 20]) == 0


def test_non_wrapper_reasoning_parsers_are_unchanged():
    assert DeepSeekR1ReasoningParser(FakeTokenizer()).count_reasoning_tokens([10, 11, 2, 20]) == 0
    assert NemotronV3ReasoningParser(FakeTokenizer()).count_reasoning_tokens([10, 11, 2, 20]) == 0
    assert ReasoningParserManager.get_reasoning_parser("qwen3") is Qwen3ReasoningParser
    assert Qwen3ReasoningParser(FakeTokenizer()).count_reasoning_tokens([10, 11, 2, 20]) == 0


def test_full_response_usage_reports_reasoning_tokens():
    class FakeServing:
        enable_prompt_tokens_details = False

        def _make_usage_info(self, **kwargs):
            return usage_patch._make_usage_info(self, **kwargs)

    state = usage_patch._create_usage_tracking_state(
        num_choices=1,
        reasoning_parser=_reasoning_parser("glm45"),
    )
    state.num_prompt_tokens = 3
    state.final_res = SimpleNamespace(num_cached_tokens=None)
    state.completion_tokens = [4]
    state.raw_output_token_ids = [[10, 11, 2, 20]]

    usage = usage_patch._make_full_response_usage(FakeServing(), state)

    assert usage.completion_tokens_details.reasoning_tokens == 2


def test_stream_usage_reports_reasoning_tokens():
    state = usage_patch._create_usage_tracking_state(
        num_choices=1,
        reasoning_parser=_reasoning_parser("deepseek_v4", {"thinking": True}),
    )
    state.raw_output_token_ids = [[10, 11, 2, 20]]
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        },
    }

    data = usage_patch._inject_stream_usage_details(
        f"data: {json.dumps(chunk)}\n\n",
        state,
    )
    payload = json.loads(data.removeprefix("data: ").removesuffix("\n\n"))

    assert payload["usage"]["completion_tokens_details"] == {
        "reasoning_tokens": 2,
    }
