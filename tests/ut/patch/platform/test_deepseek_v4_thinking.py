# SPDX-License-Identifier: Apache-2.0

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import deepseek_v4, deepseek_v4_encoding


class FakeTokenizer:
    vocab_size = 1

    def get_added_vocab(self):
        return {}

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


@pytest.mark.parametrize(
    ("kwargs", "expected_mode", "expected_effort"),
    [
        ({}, "thinking", "high"),
        ({"enable_thinking": True}, "thinking", "high"),
        ({"enable_thinking": False}, "chat", None),
        ({"thinking": False}, "chat", None),
        ({"reasoning_effort": "none"}, "chat", None),
        ({"reasoning_effort": "minimal"}, "thinking", "low"),
        ({"reasoning_effort": "low"}, "thinking", "low"),
        ({"reasoning_effort": "medium"}, "thinking", "low"),
        ({"reasoning_effort": "high"}, "thinking", "high"),
        ({"reasoning_effort": "xhigh"}, "thinking", "high"),
        ({"reasoning_effort": "max"}, "thinking", "max"),
        ({"reasoning_effort": "unexpected"}, "thinking", "high"),
        (
            {"enable_thinking": False, "reasoning_effort": "max"},
            "chat",
            "max",
        ),
    ],
)
def test_deepseek_v4_tokenizer_maps_latest_reasoning_effort_values(
    monkeypatch,
    kwargs,
    expected_mode,
    expected_effort,
):
    captured_kwargs = []

    def fake_encode_messages(messages, **kwargs):
        captured_kwargs.append(kwargs)
        return "prompt"

    monkeypatch.setattr(deepseek_v4, "encode_messages", fake_encode_messages)
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())

    tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        **kwargs,
    )
    assert captured_kwargs[-1]["thinking_mode"] == expected_mode
    assert captured_kwargs[-1]["reasoning_effort"] == expected_effort


def test_deepseek_v4_defaults_to_thinking_with_high_effort():
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum")
    assert prompt.endswith("<｜Assistant｜><think>")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reasoning_effort": "none"},
        {"enable_thinking": False, "reasoning_effort": "max"},
    ],
)
def test_deepseek_v4_explicit_disable_overrides_reasoning_effort(kwargs):
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        **kwargs,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>")


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_prefix"),
    [
        ("low", "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>hi"),
        (
            "high",
            "<\uff5cbegin\u2581of\u2581sentence\uff5c>Reasoning Effort: Absolute maximum with no shortcuts permitted.",
        ),
        (
            "max",
            "<\uff5cbegin\u2581of\u2581sentence\uff5c>Reasoning Effort: "
            "Beyond maximum \u2014 exhaustive, relentless, and uncompromising.",
        ),
    ],
)
def test_deepseek_v4_renders_0731_reasoning_effort_prompts(
    reasoning_effort,
    expected_prefix,
):
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )

    assert prompt.startswith(expected_prefix)
    assert prompt.endswith("<\uff5cAssistant\uff5c><think>")


def test_deepseek_v4_rejects_invalid_renderer_reasoning_effort():
    with pytest.raises(ValueError, match="Invalid reasoning effort: unexpected"):
        deepseek_v4_encoding.render_message(
            0,
            [{"role": "user", "content": "hi"}],
            thinking_mode="thinking",
            reasoning_effort="unexpected",
        )
