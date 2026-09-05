# SPDX-License-Identifier: Apache-2.0

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.tokenizers import deepseek_v4, deepseek_v4_encoding

from vllm_ascend.patch.platform import patch_deepseek_v4_frontend


class FakeTokenizer:
    vocab_size = 1
    name_or_path = "deepseek-v4"

    def get_added_vocab(self):
        return {}

    def get_vocab(self):
        return {"<think>": 1, "</think>": 2}

    def encode(self, text, add_special_tokens=False, **kwargs):
        return text


def _mock_checkpoint_config(monkeypatch, *, post_preview: bool) -> None:
    config: dict[str, object] = {"dspark_block_size": 5} if post_preview else {"model_type": "deepseek_v4"}
    monkeypatch.setattr(
        patch_deepseek_v4_frontend,
        "get_hf_file_to_dict",
        lambda *args, **kwargs: config,
    )


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
        ({"enable_thinking": False, "reasoning_effort": "max"}, "chat", "max"),
    ],
)
def test_post_preview_reasoning_effort_mapping(
    monkeypatch,
    kwargs,
    expected_mode,
    expected_effort,
):
    captured_kwargs = []

    def fake_encode_messages(messages, **kwargs):
        captured_kwargs.append(kwargs)
        return "prompt"

    _mock_checkpoint_config(monkeypatch, post_preview=True)
    monkeypatch.setattr(deepseek_v4, "encode_messages", fake_encode_messages)
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        **kwargs,
    )

    assert captured_kwargs[-1]["thinking_mode"] == expected_mode
    assert captured_kwargs[-1]["reasoning_effort"] == expected_effort


@pytest.mark.parametrize(
    ("kwargs", "expected_mode", "expected_effort"),
    [
        ({}, "thinking", "low"),
        ({"enable_thinking": True}, "thinking", "low"),
        ({"enable_thinking": False}, "chat", None),
        ({"reasoning_effort": "none"}, "chat", None),
        ({"reasoning_effort": "minimal"}, "thinking", "low"),
        ({"reasoning_effort": "low"}, "thinking", "low"),
        ({"reasoning_effort": "medium"}, "thinking", "low"),
        ({"reasoning_effort": "high"}, "thinking", "low"),
        ({"reasoning_effort": "xhigh"}, "thinking", "high"),
        ({"reasoning_effort": "max"}, "thinking", "high"),
        ({"reasoning_effort": "unexpected"}, "thinking", "low"),
        ({"enable_thinking": False, "reasoning_effort": "max"}, "chat", "high"),
    ],
)
def test_preview_reasoning_effort_mapping(
    monkeypatch,
    kwargs,
    expected_mode,
    expected_effort,
):
    captured_kwargs = []

    def fake_encode_messages(messages, **kwargs):
        captured_kwargs.append(kwargs)
        return "prompt"

    _mock_checkpoint_config(monkeypatch, post_preview=False)
    monkeypatch.setattr(deepseek_v4, "encode_messages", fake_encode_messages)
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        **kwargs,
    )

    assert captured_kwargs[-1]["thinking_mode"] == expected_mode
    assert captured_kwargs[-1]["reasoning_effort"] == expected_effort


def test_tools_are_attached_to_existing_system_without_mutation(monkeypatch):
    captured_messages = []

    def fake_encode_messages(messages, **kwargs):
        captured_messages.append(messages)
        return "prompt"

    _mock_checkpoint_config(monkeypatch, post_preview=True)
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


def test_tools_add_system_when_missing(monkeypatch):
    captured_messages = []

    def fake_encode_messages(messages, **kwargs):
        captured_messages.append(messages)
        return "prompt"

    _mock_checkpoint_config(monkeypatch, post_preview=True)
    monkeypatch.setattr(deepseek_v4, "encode_messages", fake_encode_messages)
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)

    assert captured_messages[-1] == [
        {"role": "system", "tools": tools},
        {"role": "user", "content": "hi"},
    ]
    assert messages == [{"role": "user", "content": "hi"}]


def test_conversation_keyword_is_bound_once(monkeypatch):
    captured_messages = []

    def fake_encode_messages(messages, **kwargs):
        captured_messages.append(messages)
        return "prompt"

    _mock_checkpoint_config(monkeypatch, post_preview=True)
    monkeypatch.setattr(deepseek_v4, "encode_messages", fake_encode_messages)
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
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


def test_tokenizer_instances_do_not_modify_generated_class(monkeypatch):
    _mock_checkpoint_config(monkeypatch, post_preview=True)
    first = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    first_class_apply = type(first).apply_chat_template
    second = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())

    assert type(first).apply_chat_template is first_class_apply
    assert "apply_chat_template" in first.__dict__
    assert "apply_chat_template" in second.__dict__


def test_rewrapping_tokenizer_binds_method_to_new_instance(monkeypatch):
    _mock_checkpoint_config(monkeypatch, post_preview=True)
    first = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    second = deepseek_v4.get_deepseek_v4_tokenizer(first)

    assert first.apply_chat_template.__self__ is first
    assert second.apply_chat_template.__self__ is second


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


@pytest.mark.parametrize(
    ("post_preview", "reasoning_effort", "expected_prefix"),
    [
        (False, None, "<｜begin▁of▁sentence｜><｜User｜>hi"),
        (False, "high", "<｜begin▁of▁sentence｜><｜User｜>hi"),
        (
            False,
            "max",
            "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum",
        ),
        (
            True,
            "high",
            "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum",
        ),
        (
            True,
            "max",
            "<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum",
        ),
    ],
)
def test_checkpoint_specific_reasoning_prompts(
    monkeypatch,
    post_preview,
    reasoning_effort,
    expected_prefix,
):
    _mock_checkpoint_config(monkeypatch, post_preview=post_preview)
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    kwargs = {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        **kwargs,
    )
    assert prompt.startswith(expected_prefix)
    assert prompt.endswith("<｜Assistant｜><think>")


def test_checkpoint_config_failure_uses_preview_mapping(monkeypatch):
    def unavailable_config(*args, **kwargs):
        raise OSError("config unavailable")

    monkeypatch.setattr(
        patch_deepseek_v4_frontend,
        "get_hf_file_to_dict",
        unavailable_config,
    )
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜><｜User｜>hi")


def test_deepseek_v4_rejects_invalid_renderer_reasoning_effort():
    with pytest.raises((AssertionError, ValueError), match="Invalid reasoning effort"):
        deepseek_v4_encoding.render_message(
            0,
            [{"role": "user", "content": "hi"}],
            thinking_mode="thinking",
            reasoning_effort="unexpected",
        )
