# SPDX-License-Identifier: Apache-2.0

from vllm.tokenizers import deepseek_v4

from vllm_ascend.patch.platform import patch_deepseek_v4_frontend  # noqa: F401


class FakeTokenizer:
    def get_added_vocab(self):
        return {}

    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False, **kwargs):
        return text


def _tokenizer(monkeypatch):
    captured_messages = []

    def fake_encode_messages(messages, **kwargs):
        captured_messages.append(messages)
        return "prompt"

    monkeypatch.setattr(deepseek_v4, "encode_messages", fake_encode_messages)
    return deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer()), captured_messages


def test_request_tools_attach_to_existing_system_without_mutation(monkeypatch):
    tokenizer, captured_messages = _tokenizer(monkeypatch)
    messages = [
        {"role": "system", "content": "system prompt", "tools": ["old"]},
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


def test_request_tools_insert_system_when_missing(monkeypatch):
    tokenizer, captured_messages = _tokenizer(monkeypatch)
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)

    assert captured_messages[-1] == [
        {"role": "system", "tools": tools},
        {"role": "user", "content": "hi"},
    ]
    assert messages == [{"role": "user", "content": "hi"}]


def test_request_tools_use_conversation_keyword(monkeypatch):
    tokenizer, captured_messages = _tokenizer(monkeypatch)
    messages = [{"role": "user", "content": "ignored"}]
    conversation = [{"role": "user", "content": "used"}]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    tokenizer.apply_chat_template(
        messages,
        conversation=conversation,
        tools=tools,
        tokenize=False,
    )

    assert captured_messages[-1] == [
        {"role": "system", "tools": tools},
        {"role": "user", "content": "used"},
    ]
