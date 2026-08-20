# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser
from vllm.tokenizers import deepseek_v4

from vllm_ascend.patch.platform import patch_deepseek_v4_thinking_defaults


class FakeTokenizer:
    vocab_size = 1

    def get_added_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False, **kwargs):
        return text


class ReasoningTokenizer(FakeTokenizer):
    def get_vocab(self):
        return {"<think>": 1, "</think>": 2}


@pytest.mark.parametrize(
    ("provided_kwargs", "expected_kwargs"),
    [
        ({}, {"enable_thinking": True, "reasoning_effort": "high"}),
        (
            {"reasoning_effort": "low"},
            {"enable_thinking": True, "reasoning_effort": "low"},
        ),
        (
            {"reasoning_effort": "none"},
            {"enable_thinking": False, "reasoning_effort": "none"},
        ),
        (
            {"enable_thinking": False},
            {"enable_thinking": False, "reasoning_effort": "high"},
        ),
        (
            {"thinking": False},
            {"thinking": False, "reasoning_effort": "high"},
        ),
    ],
)
def test_deepseek_v4_thinking_defaults_preserve_explicit_controls(
    provided_kwargs,
    expected_kwargs,
):
    original_kwargs = dict(provided_kwargs)

    assert patch_deepseek_v4_thinking_defaults._apply_deepseek_v4_thinking_defaults(provided_kwargs) == expected_kwargs
    assert provided_kwargs == original_kwargs


def test_renderer_and_reasoning_parser_receive_matching_defaults(monkeypatch):
    class FakeDeepseekV4Renderer:
        pass

    captured_renderer_kwargs = []

    def fake_apply_chat_template(self, *args, **kwargs):
        captured_renderer_kwargs.append(kwargs)
        return "prompt"

    monkeypatch.setattr(
        patch_deepseek_v4_thinking_defaults,
        "DeepseekV4Renderer",
        FakeDeepseekV4Renderer,
    )
    monkeypatch.setattr(
        patch_deepseek_v4_thinking_defaults,
        "_original_apply_chat_template",
        fake_apply_chat_template,
    )
    monkeypatch.setattr(
        patch_deepseek_v4_thinking_defaults,
        "_original_effective_chat_template_kwargs",
        lambda self, request: {},
    )
    serving = SimpleNamespace(renderer=FakeDeepseekV4Renderer())
    request = ChatCompletionRequest(
        model="custom-served-alias",
        messages=[{"role": "user", "content": "hi"}],
    )

    effective_kwargs = patch_deepseek_v4_thinking_defaults._effective_chat_template_kwargs(
        serving,
        request,
    )
    patch_deepseek_v4_thinking_defaults._apply_chat_template(
        serving.renderer,
        messages=request.messages,
    )

    assert effective_kwargs == {key: captured_renderer_kwargs[-1][key] for key in effective_kwargs}
    reasoning_parser = DeepSeekV3ReasoningParser(
        ReasoningTokenizer(),
        chat_template_kwargs=effective_kwargs,
    )
    assert isinstance(reasoning_parser._parser, DeepSeekR1ReasoningParser)
    reasoning, content = reasoning_parser.extract_reasoning(
        "internal reasoning</think>final answer",
        request,
    )
    assert reasoning == "internal reasoning"
    assert content == "final answer"


def test_effective_defaults_only_apply_to_deepseek_v4_renderer(monkeypatch):
    monkeypatch.setattr(
        patch_deepseek_v4_thinking_defaults,
        "_original_effective_chat_template_kwargs",
        lambda self, request: {"custom": "value"},
    )

    kwargs = patch_deepseek_v4_thinking_defaults._effective_chat_template_kwargs(
        SimpleNamespace(renderer=object()),
        ChatCompletionRequest(
            model="deepseek-v4",
            messages=[{"role": "user", "content": "hi"}],
        ),
    )

    assert kwargs == {"custom": "value"}


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
    for reasoning_effort, expected_mode, expected_effort in cases:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            enable_thinking=True,
            reasoning_effort=reasoning_effort,
        )
        assert captured_kwargs[-1]["thinking_mode"] == expected_mode
        assert captured_kwargs[-1]["reasoning_effort"] == expected_effort
