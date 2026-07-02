# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for MiniMax M3 reasoning parser.

Covers: vllm_ascend/patch/platform/minimax/minimax_m3_reasoning_parser.py
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage

from vllm_ascend.patch.platform.minimax.minimax_m3_reasoning_parser import (
    MiniMaxM3ReasoningParser,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


# ---------------------------------------------------------------------------
# Fake tokenizer
# ---------------------------------------------------------------------------

START_ID = 100
END_ID = 101


class FakeTokenizer:
    """Tokeniser that exposes the MiniMax M3 think markers.

    Provides enough of the HuggingFace tokenizer interface for
    ``BaseThinkingReasoningParser`` to look up start/end token IDs.
    """

    def __init__(self, vocab_overrides: dict[str, int] | None = None):
        self.vocab: dict[str, int] = {
            "<mm:think>": START_ID,
            "</mm:think>": END_ID,
        }
        if vocab_overrides:
            self.vocab.update(vocab_overrides)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def encode(self, text: str, *args, **kwargs):
        """Minimal encode that returns integer token IDs for known markers."""
        result: list[int] = []
        i = 0
        while i < len(text):
            if text[i:].startswith("<mm:think>"):
                result.append(START_ID)
                i += len("<mm:think>")
            elif text[i:].startswith("</mm:think>"):
                result.append(END_ID)
                i += len("</mm:think>")
            else:
                # Assign a placeholder id for unknown chars
                result.append(ord(text[i]))
                i += 1
        return result

    def convert_ids_to_tokens(self, ids, *args, **kwargs):
        if not ids:
            return []
        return list(ids)

    def convert_tokens_to_string(self, tokens, *args, **kwargs):
        # tokens may be ints or strings depending on the caller
        parts = []
        for t in tokens:
            if isinstance(t, int):
                parts.append(chr(t) if 32 <= t < 128 else f"<{t}>")
            else:
                parts.append(str(t))
        return "".join(parts)


# ---------------------------------------------------------------------------
# Helper — instantiate with fake tokenizer
# ---------------------------------------------------------------------------

def _make_parser(**kwargs) -> MiniMaxM3ReasoningParser:
    return MiniMaxM3ReasoningParser(FakeTokenizer(), **kwargs)


# ===========================================================================
# Non-streaming extract_reasoning
# ===========================================================================

class TestExtractReasoning:
    def test_plain_content_no_markers(self):
        p = _make_parser()
        reasoning, content = p.extract_reasoning("Hello world", None)
        assert reasoning is None
        assert content == "Hello world"

    def test_explicit_thinking_block(self):
        p = _make_parser()
        reasoning, content = p.extract_reasoning(
            "<mm:think>Let me think</mm:think>The answer", None
        )
        assert reasoning == "Let me think"
        assert content == "The answer"

    def test_only_reasoning_no_closer(self):
        p = _make_parser()
        reasoning, content = p.extract_reasoning(
            "<mm:think>unfinished thought", None
        )
        assert reasoning == "unfinished thought"
        assert content is None

    def test_content_before_think_block(self):
        p = _make_parser()
        reasoning, content = p.extract_reasoning(
            "prefix<mm:think>reason</mm:think>suffix", None
        )
        assert reasoning == "reason"
        assert content == "prefixsuffix"

    def test_multiple_think_blocks(self):
        p = _make_parser()
        reasoning, content = p.extract_reasoning(
            "<mm:think>first</mm:think>text<mm:think>second</mm:think>tail", None
        )
        assert reasoning == "first"
        assert content == "text<mm:think>second</mm:think>tail"

    def test_stray_closer_at_start(self):
        p = _make_parser()
        reasoning, content = p.extract_reasoning(
            "</mm:think>Hello world", None
        )
        assert reasoning is None
        assert content == "Hello world"

    def test_stray_closer_at_start_with_content(self):
        p = _make_parser()
        reasoning, content = p.extract_reasoning(
            "</mm:think>clean text", None
        )
        assert reasoning is None
        assert content == "clean text"

    def test_thinking_mode_enabled_no_start_token(self):
        p = _make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
        reasoning, content = p.extract_reasoning(
            "I am reasoning</mm:think>final answer", None
        )
        assert reasoning == "I am reasoning"
        assert content == "final answer"

    def test_thinking_mode_enabled_no_end_token(self):
        p = _make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
        reasoning, content = p.extract_reasoning(
            "I am reasoning without end", None
        )
        assert reasoning == "I am reasoning without end"
        assert content is None

    def test_thinking_mode_enabled_with_start_still_detected(self):
        p = _make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
        reasoning, content = p.extract_reasoning(
            "<mm:think>explicit</mm:think>rest", None
        )
        assert reasoning == "explicit"
        assert content == "rest"

    def test_empty_output(self):
        p = _make_parser()
        reasoning, content = p.extract_reasoning("", None)
        assert reasoning is None
        assert content == ""


# ===========================================================================
# Streaming extract_reasoning_streaming
# ===========================================================================

class TestExtractReasoningStreaming:
    def _feed(self, parser, chunks):
        previous = ""
        previous_ids: list[int] = []
        results = []
        for chunk in chunks:
            if isinstance(chunk, tuple):
                delta, delta_ids = chunk
            else:
                delta = chunk
                # Encode to get proper token IDs for markers
                delta_ids = list(parser.model_tokenizer.encode(delta))
            current = previous + delta
            current_ids = previous_ids + list(delta_ids)
            result = parser.extract_reasoning_streaming(
                previous_text=previous,
                current_text=current,
                delta_text=delta,
                previous_token_ids=previous_ids,
                current_token_ids=current_ids,
                delta_token_ids=delta_ids,
            )
            if result is not None:
                results.append(result)
            previous = current
            previous_ids = current_ids
        return results

    def _collect(self, results, attr):
        return "".join(getattr(r, attr) or "" for r in results)

    def test_plain_content(self):
        p = _make_parser()
        results = self._feed(p, ["Hello ", "world!"])
        content = self._collect(results, "content")
        assert content == "Hello world!"

    def test_thinking_block_streaming(self):
        p = _make_parser()
        # Send everything together so start/end markers are processed atomically.
        results = self._feed(p, ["<mm:think>reasoning text</mm:think>answer"])
        reasoning = self._collect(results, "reasoning")
        content = self._collect(results, "content")
        assert "reasoning text" in reasoning
        assert "answer" in content

    def test_enabled_mode_streaming(self):
        p = _make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
        # In enabled mode, reasoning starts immediately (no start marker needed).
        results = self._feed(p, ["I think therefore</mm:think> I am"])
        reasoning = self._collect(results, "reasoning")
        content = self._collect(results, "content")
        assert "I think therefore" in reasoning
        assert "I am" in content

    def test_enabled_mode_no_closer(self):
        p = _make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
        results = self._feed(p, ["pure reasoning"])
        reasoning = self._collect(results, "reasoning")
        content = self._collect(results, "content")
        assert reasoning == "pure reasoning"
        assert content == ""

    def test_stray_closer_streaming(self):
        p = _make_parser()
        results = self._feed(p, ["</mm:think>clean text"])
        content = self._collect(results, "content")
        reasoning = self._collect(results, "reasoning")
        assert "clean text" in content
        assert reasoning == ""

    def test_start_token_in_middle_streaming(self):
        p = _make_parser()
        # When both start and end markers are in the same delta chunk, the
        # content before the start token is not emitted (parser limitation).
        results = self._feed(p, ["hello <mm:think>reasons</mm:think> bye"])
        reasoning = self._collect(results, "reasoning")
        content = self._collect(results, "content")
        assert "reasons" in reasoning
        assert "bye" in content

    def test_end_token_in_delta(self):
        p = _make_parser()
        results = self._feed(p, ["<mm:think>r</mm:think>c"])
        reasoning = self._collect(results, "reasoning")
        content = self._collect(results, "content")
        assert "r" in reasoning
        assert "c" in content

    def test_end_token_with_token_ids(self):
        p = _make_parser()
        results = self._feed(p, [
            ("<mm:think>r</mm:think>c", [START_ID, ord("r"), END_ID, ord("c")]),
        ])
        reasoning = self._collect(results, "reasoning")
        content = self._collect(results, "content")
        assert "r" in reasoning
        assert "c" in content

    def test_empty_delta(self):
        p = _make_parser()
        results = self._feed(p, [""])
        assert results == []

    def test_start_token_in_delta_ids_enabled(self):
        p = _make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
        results = self._feed(p, [
            ("reasoning", []),
            (" at end</mm:think>", [END_ID]),
            (" content", []),
        ])
        reasoning = self._collect(results, "reasoning")
        content = self._collect(results, "content")
        assert "reasoning" in reasoning
        assert "content" in content


# ===========================================================================
# is_reasoning_end_streaming
# ===========================================================================

class TestIsReasoningEndStreaming:
    def test_end_id_in_delta_ids(self):
        p = _make_parser()
        assert p.is_reasoning_end_streaming([], [END_ID]) is True

    def test_end_id_in_input_ids(self):
        p = _make_parser()
        assert p.is_reasoning_end_streaming([END_ID], []) is True

    def test_initial_in_reasoning_returns_false(self):
        p = _make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
        # When in reasoning mode initially, without start token, return False
        assert p.is_reasoning_end_streaming([], []) is False

    def test_start_id_not_in_input(self):
        p = _make_parser()
        assert p.is_reasoning_end_streaming([999], [999]) is True  # arbitrary ids -> content mode

    def test_delta_without_end(self):
        p = _make_parser()
        assert p.is_reasoning_end_streaming([START_ID], [999]) is False

    def test_initial_in_reasoning_with_start_not_in_input(self):
        p = _make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
        # start_token_id not in input_ids, still in thinking -> False
        assert p.is_reasoning_end_streaming([999], [999]) is False

    def test_initial_in_reasoning_start_in_input(self):
        p = _make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
        # start_token_id IS in input_ids, now we check end
        assert p.is_reasoning_end_streaming([START_ID, END_ID], []) is True


# ===========================================================================
# extract_content_ids
# ===========================================================================

class TestExtractContentIds:
    def test_end_token_returns_suffix(self):
        p = _make_parser()
        result = p.extract_content_ids([START_ID, 1, 2, END_ID, 3, 4])
        assert result == [3, 4]

    def test_no_end_initial_in_reasoning(self):
        p = _make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
        result = p.extract_content_ids([1, 2, 3])
        assert result == []

    def test_no_start_not_initial_in_reasoning(self):
        p = _make_parser()
        result = p.extract_content_ids([1, 2, 3])
        assert result == [1, 2, 3]

    def test_start_present_no_end(self):
        p = _make_parser()
        result = p.extract_content_ids([START_ID, 1, 2])
        assert result == []

    def test_end_at_end_of_list(self):
        p = _make_parser()
        result = p.extract_content_ids([START_ID, 1, END_ID])
        assert result == []

    def test_multiple_ends_uses_last(self):
        p = _make_parser()
        result = p.extract_content_ids([START_ID, 1, END_ID, 2, END_ID, 3])
        assert result == [3]


# ===========================================================================
# count_reasoning_tokens
# ===========================================================================

class TestCountReasoningTokens:
    def test_without_initial_in_reasoning(self):
        p = _make_parser()
        result = p.count_reasoning_tokens([START_ID, 1, 2, END_ID, 3, 4])
        assert result == 2  # tokens between start and end

    def test_without_initial_in_reasoning_no_start(self):
        p = _make_parser()
        # No start token -> everything is content (0 reasoning tokens)
        result = p.count_reasoning_tokens([1, 2, 3])
        assert result == 0

    def test_initial_in_reasoning_depth_based(self):
        p = _make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
        result = p.count_reasoning_tokens([1, 2, END_ID, 3, 4])
        assert result == 2  # depth starts at 1, drops to 0 at END_ID

    def test_initial_in_reasoning_nested_thinking(self):
        p = _make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
        # nested: depth 1 -> START(2) -> END(1) -> END(0)
        result = p.count_reasoning_tokens([
            1, 2, START_ID, 3, 4, END_ID, 5, END_ID, 6, 7
        ])
        # depth=1: tokens 1,2 = 2
        # depth=2 (after START): tokens 3,4 = 2
        # depth=1 (after first END): token 5 = 1
        # depth=0 (after second END): tokens 6,7 = 0
        total = 2 + 2 + 1
        assert result == 5

    def test_initial_in_reasoning_only_reasoning(self):
        p = _make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
        result = p.count_reasoning_tokens([1, 2, 3, 4])
        assert result == 4


# ===========================================================================
# start_token / end_token properties
# ===========================================================================


def test_start_token_value():
    p = _make_parser()
    assert p.start_token == "<mm:think>"


def test_end_token_value():
    p = _make_parser()
    assert p.end_token == "</mm:think>"
