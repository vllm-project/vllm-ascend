# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm_ascend.models.iquest_q1.reasoning_parser import (
    IQuestQ1ReasoningParser,
)

START = 100
END = 101
ASSISTANT = 102
TOOL = 103


@pytest.fixture
def tokenizer():
    return Mock(
        get_vocab=lambda: {
            "<think>": START,
            "</think>": END,
            "<|iquest_assistant|>": ASSISTANT,
            "<iquest_tool_call>": TOOL,
        }
    )


@pytest.mark.parametrize(
    "output,expected",
    [
        ("reasoning</think>answer", ("reasoning", "answer")),
        ("<think>reasoning</think>answer", ("reasoning", "answer")),
        ("<think>line one\nline two</think>", ("line one\nline two", None)),
        ("<think>unfinished", ("unfinished", None)),
        ("</think>answer", (None, "answer")),
        ("answer", ("answer", None)),
        ("", (None, None)),
    ],
)
def test_implicit_or_explicit_reasoning_start(tokenizer, output, expected):
    parser = IQuestQ1ReasoningParser(tokenizer)
    assert parser.extract_reasoning(output, request=None) == expected


@pytest.mark.parametrize(
    "chat_kwargs,enabled",
    [
        ({}, True),
        ({"enable_thinking": True}, True),
        ({"enable_thinking": False}, False),
        ({"thinking": False}, False),
        ({"thinking": True}, True),
    ],
)
def test_thinking_flag_matches_checkpoint_template(tokenizer, chat_kwargs, enabled):
    """Either chat-template kwarg turns thinking off."""
    parser = IQuestQ1ReasoningParser(tokenizer, chat_template_kwargs=chat_kwargs)
    output = "reasoning</think>answer"
    expected = ("reasoning", "answer") if enabled else (None, output)
    assert parser.extract_reasoning(output, request=None) == expected


@pytest.mark.parametrize(
    "tokens,ended,content",
    [
        ([START, 1, END, 2, 3], True, [2, 3]),
        ([END, 1, ASSISTANT, START, 2], False, []),
        ([END, 1, ASSISTANT, 2], False, []),
        ([END, 1, START, 2], False, []),
        ([1, 2], False, []),
        ([TOOL, 1, ASSISTANT, 2], False, []),
        ([START, 1, END, TOOL, 2], True, [TOOL, 2]),
    ],
)
def test_previous_turn_cannot_end_current_reasoning(tokenizer, tokens, ended, content):
    parser = IQuestQ1ReasoningParser(tokenizer)
    assert parser.is_reasoning_end(tokens) is ended
    assert parser.extract_content_ids(tokens) == content


def test_mtp_delta_contains_reasoning_end_and_content(tokenizer):
    parser = IQuestQ1ReasoningParser(tokenizer)
    chunks = [
        ("<think>", [START]),
        ("first ", [1]),
        ("second</think>answer", [2, END, 3]),
        (" tail", [4]),
    ]
    text = ""
    token_ids: list[int] = []
    messages = []
    for chunk, chunk_ids in chunks:
        result = parser.extract_reasoning_streaming(
            text, text + chunk, chunk, token_ids, token_ids + chunk_ids, chunk_ids
        )
        if result is not None:
            messages.append(result)
        text += chunk
        token_ids += chunk_ids
    assert "".join(message.reasoning or "" for message in messages) == "first second"
    assert "".join(message.content or "" for message in messages) == "answer tail"
    assert parser.is_reasoning_end_streaming(token_ids, iter([2, END, 3]))
    assert parser.count_reasoning_tokens(token_ids) == 2


def test_disabled_thinking_emits_content_from_first_token(tokenizer):
    parser = IQuestQ1ReasoningParser(tokenizer, chat_template_kwargs={"enable_thinking": False})
    result = parser.extract_reasoning_streaming("", "answer", "answer", [], [1], [1])
    assert result.content == "answer"
    assert result.reasoning is None
    assert parser.is_reasoning_end([ASSISTANT, START])
    assert parser.is_reasoning_end_streaming([1], iter([1]))
    assert parser.extract_content_ids([1, 2]) == [1, 2]
    assert parser.count_reasoning_tokens([1, 2]) == 0


@pytest.mark.parametrize("missing", ["<think>", "</think>", "<|iquest_assistant|>"])
def test_missing_template_token_rejected(tokenizer, missing):
    vocab = tokenizer.get_vocab()
    del vocab[missing]
    with pytest.raises(ValueError, match="missing required"):
        IQuestQ1ReasoningParser(Mock(get_vocab=lambda: vocab))
