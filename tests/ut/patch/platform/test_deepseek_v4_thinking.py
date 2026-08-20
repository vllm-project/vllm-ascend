# SPDX-License-Identifier: Apache-2.0

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.tokenizers import deepseek_v4

from vllm_ascend.utils import vllm_version_is


class FakeTokenizer:
    vocab_size = 1

    def get_added_vocab(self):
        return {}

    def get_vocab(self):
        return {"<think>": 1, "</think>": 2}

    def encode(self, text, add_special_tokens=False, **kwargs):
        return text


def test_deepseek_v4_reasoning_effort_accepts_latest_values():
    for reasoning_effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        request = ChatCompletionRequest(
            model="deepseek-v4",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort=reasoning_effort,
        )
        assert request.reasoning_effort == reasoning_effort


def test_reasoning_effort_enables_thinking_unless_user_overrides():
    request = ChatCompletionRequest(
        model="deepseek-v4",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="high",
    )
    params = request.build_chat_params(None, "auto")
    assert params.chat_template_kwargs["enable_thinking"] is True

    request = ChatCompletionRequest(
        model="deepseek-v4",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="none",
    )
    params = request.build_chat_params(None, "auto")
    assert params.chat_template_kwargs["enable_thinking"] is False

    request = ChatCompletionRequest(
        model="deepseek-v4",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="high",
        chat_template_kwargs={"enable_thinking": False},
    )
    params = request.build_chat_params(None, "auto")
    assert params.chat_template_kwargs["enable_thinking"] is False


def test_deepseek_v4_tokenizer_maps_latest_reasoning_effort_values(monkeypatch):
    captured_kwargs = []

    def fake_encode_messages(messages, **kwargs):
        captured_kwargs.append(kwargs)
        return "prompt"

    monkeypatch.setattr(deepseek_v4, "encode_messages", fake_encode_messages)
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())

    if vllm_version_is("0.27.1"):
        cases = [
            ("none", "chat", None),
            ("minimal", "thinking", "high"),
            ("low", "thinking", "high"),
            ("medium", "thinking", "high"),
            ("high", "thinking", "high"),
            ("xhigh", "thinking", "max"),
            ("max", "thinking", "max"),
            ("unexpected", "thinking", "high"),
        ]
    else:
        cases = [
            ("none", "chat", None),
            ("minimal", "thinking", "low"),
            ("low", "thinking", "low"),
            ("medium", "thinking", "low"),
            ("high", "thinking", "high"),
            ("xhigh", "thinking", "high"),
            ("max", "thinking", "max"),
            ("unexpected", "thinking", "high"),
        ]
    for reasoning_effort, expected_mode, expected_effort in cases:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            enable_thinking=True,
            reasoning_effort=reasoning_effort,
        )
        assert captured_kwargs[-1]["thinking_mode"] == expected_mode
        assert captured_kwargs[-1]["reasoning_effort"] == expected_effort


def test_deepseek_v4_tokenizer_attaches_tools_to_existing_system(monkeypatch):
    captured_messages = []

    def fake_encode_messages(messages, **kwargs):
        captured_messages.append(messages)
        return "prompt"

    monkeypatch.setattr(deepseek_v4, "encode_messages", fake_encode_messages)
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hi"},
    ]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    original_messages = [message.copy() for message in messages]

    tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)

    assert captured_messages[-1] == [
        {"role": "system", "content": "system prompt", "tools": tools},
        {"role": "user", "content": "hi"},
    ]
    assert messages == original_messages


def test_deepseek_v4_tokenizer_adds_system_for_tools_when_missing(monkeypatch):
    captured_messages = []

    def fake_encode_messages(messages, **kwargs):
        captured_messages.append(messages)
        return "prompt"

    monkeypatch.setattr(deepseek_v4, "encode_messages", fake_encode_messages)
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    original_messages = [message.copy() for message in messages]

    tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)

    assert captured_messages[-1] == [
        {"role": "system", "tools": tools},
        {"role": "user", "content": "hi"},
    ]
    assert messages == original_messages


@pytest.mark.parametrize(
    ("chat_template_kwargs", "expected_state"),
    [
        ({}, "REASONING"),
        ({"thinking": True}, "REASONING"),
        ({"enable_thinking": True}, "REASONING"),
        ({"reasoning_effort": "high"}, "REASONING"),
        ({"thinking": False}, "CONTENT"),
        ({"enable_thinking": False}, "CONTENT"),
        ({"enable_thinking": True, "reasoning_effort": "none"}, "CONTENT"),
    ],
)
def test_parser_thinking_mode_matches_tokenizer_default(
    chat_template_kwargs,
    expected_state,
):
    parser = DeepSeekV4Parser(
        FakeTokenizer(),
        chat_template_kwargs=chat_template_kwargs,
    )

    assert parser.parser_engine_config.initial_state.name == expected_state
