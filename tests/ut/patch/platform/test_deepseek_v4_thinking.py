# SPDX-License-Identifier: Apache-2.0

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.tokenizers import deepseek_v4, deepseek_v4_encoding

from vllm_ascend.patch.platform import patch_deepseek_v4_thinking


class FakeTokenizer:
    vocab_size = 1
    name_or_path = "deepseek-v4"

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
def test_deepseek_v4_tokenizer_maps_post_preview_reasoning_effort_values(
    monkeypatch,
    kwargs,
    expected_mode,
    expected_effort,
):
    captured_kwargs = []

    def fake_encode_messages(messages, **kwargs):
        captured_kwargs.append(kwargs)
        return "prompt"

    monkeypatch.setattr(
        patch_deepseek_v4_thinking,
        "get_hf_file_to_dict",
        lambda *args, **kwargs: {"dspark_block_size": 5},
    )
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
        (
            {"enable_thinking": False, "reasoning_effort": "max"},
            "chat",
            "high",
        ),
    ],
)
def test_deepseek_v4_tokenizer_maps_preview_reasoning_effort_values(
    monkeypatch,
    kwargs,
    expected_mode,
    expected_effort,
):
    captured_kwargs = []

    def fake_encode_messages(messages, **kwargs):
        captured_kwargs.append(kwargs)
        return "prompt"

    monkeypatch.setattr(
        patch_deepseek_v4_thinking,
        "get_hf_file_to_dict",
        lambda *args, **kwargs: {"model_type": "deepseek_v4"},
    )
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
    ("reasoning_effort", "expected_effort"),
    [
        (None, "high"),
        ("high", "high"),
        ("low", "low"),
    ],
)
def test_deepseek_v4_tokenizer_attaches_tools_to_existing_system(
    monkeypatch,
    reasoning_effort,
    expected_effort,
):
    captured_messages = []
    captured_kwargs = []

    def fake_encode_messages(messages, **kwargs):
        captured_messages.append(messages)
        captured_kwargs.append(kwargs)
        return "prompt"

    monkeypatch.setattr(
        patch_deepseek_v4_thinking,
        "get_hf_file_to_dict",
        lambda *args, **kwargs: {"dspark_block_size": 5},
    )
    monkeypatch.setattr(deepseek_v4, "encode_messages", fake_encode_messages)
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hi"},
    ]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    original_messages = [message.copy() for message in messages]
    kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}

    tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, **kwargs)

    assert captured_messages[-1] == [
        {"role": "system", "content": "system prompt", "tools": tools},
        {"role": "user", "content": "hi"},
    ]
    assert captured_kwargs[-1]["reasoning_effort"] == expected_effort
    assert messages == original_messages


def test_deepseek_v4_tokenizer_adds_system_for_tools_when_missing(monkeypatch):
    captured_messages = []

    def fake_encode_messages(messages, **kwargs):
        captured_messages.append(messages)
        return "prompt"

    monkeypatch.setattr(
        patch_deepseek_v4_thinking,
        "get_hf_file_to_dict",
        lambda *args, **kwargs: {"dspark_block_size": 5},
    )
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


def test_deepseek_v4_defaults_to_thinking_with_high_effort(monkeypatch):
    monkeypatch.setattr(
        patch_deepseek_v4_thinking,
        "get_hf_file_to_dict",
        lambda *args, **kwargs: {"dspark_block_size": 5},
    )
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum")
    assert prompt.endswith("<｜Assistant｜><think>")


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_prefix"),
    [
        (None, "<｜begin▁of▁sentence｜><｜User｜>hi"),
        ("high", "<｜begin▁of▁sentence｜><｜User｜>hi"),
        (
            "xhigh",
            "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum with no shortcuts permitted.",
        ),
        (
            "max",
            "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum with no shortcuts permitted.",
        ),
    ],
)
def test_deepseek_v4_preview_checkpoint_uses_preview_prompts(
    monkeypatch,
    reasoning_effort,
    expected_prefix,
):
    monkeypatch.setattr(
        patch_deepseek_v4_thinking,
        "get_hf_file_to_dict",
        lambda *args, **kwargs: {"model_type": "deepseek_v4"},
    )
    tokenizer = deepseek_v4.get_deepseek_v4_tokenizer(FakeTokenizer())
    kwargs = {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        **kwargs,
    )

    assert prompt.startswith(expected_prefix)
    assert "Reasoning Effort: Beyond maximum" not in prompt
    assert prompt.endswith("<｜Assistant｜><think>")


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


@pytest.mark.parametrize("request_kwargs", [{}, {"reasoning_effort": "high"}])
def test_parser_splits_implicit_start_reasoning(request_kwargs):
    request = ChatCompletionRequest(
        model="deepseek-v4",
        messages=[{"role": "user", "content": "hi"}],
        **request_kwargs,
    )
    params = request.build_chat_params(None, "auto")
    parser = DeepSeekV4Parser(
        FakeTokenizer(),
        chat_template_kwargs=params.chat_template_kwargs,
    )

    reasoning, content = parser.extract_reasoning(
        "reasoning text</think>answer text",
        request,
    )

    assert reasoning == "reasoning text"
    assert content == "answer text"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reasoning_effort": "none"},
        {"enable_thinking": False, "reasoning_effort": "max"},
    ],
)
def test_deepseek_v4_explicit_disable_overrides_reasoning_effort(monkeypatch, kwargs):
    monkeypatch.setattr(
        patch_deepseek_v4_thinking,
        "get_hf_file_to_dict",
        lambda *args, **kwargs: {"dspark_block_size": 5},
    )
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
def test_deepseek_v4_renders_post_preview_reasoning_effort_prompts(
    monkeypatch,
    reasoning_effort,
    expected_prefix,
):
    monkeypatch.setattr(
        patch_deepseek_v4_thinking,
        "get_hf_file_to_dict",
        lambda *args, **kwargs: {"dspark_block_size": 5},
    )
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
