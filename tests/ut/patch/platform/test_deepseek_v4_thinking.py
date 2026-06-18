# SPDX-License-Identifier: Apache-2.0

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import deepseek_v4


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
