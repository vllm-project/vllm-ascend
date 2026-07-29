# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for the Kimi K3 (XTML) chat format.

Vendored from vLLM commit f5a7cce9b6a61f4d995629a7418c7ea822e34a64.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re
from transformers import PreTrainedTokenizerBase
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def _subseq_index(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """Return start index of the last occurrence of needle in haystack, or -1."""
    n = len(needle)
    if n == 0:
        return -1
    for i in range(len(haystack) - n, -1, -1):
        if list(haystack[i : i + n]) == list(needle):
            return i
    return -1


class KimiK3ReasoningParser(ReasoningParser):
    """Reasoning parser for the Kimi K3 (XTML) think channel."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser constructor during construction."
            )

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        if thinking is None:
            thinking = chat_kwargs.get("enable_thinking", True)
        self._thinking_enabled = bool(thinking)

        self._think_open = "<|open|>think<|sep|>"
        self._think_close = "<|close|>think<|sep|>"
        self._response_open = "<|open|>response<|sep|>"
        self._response_close = "<|close|>response<|sep|>"
        self._message_close = "<|close|>message<|sep|>"

        open_marker = r"<\|open\|>"
        close_marker = r"<\|close\|>"
        sep_marker = r"<\|sep\|>"
        self._think_open_re = re.compile(open_marker + r"\s*think\s*" + sep_marker)
        self._think_close_re = re.compile(close_marker + r"\s*think\s*" + sep_marker)
        self._response_open_re = re.compile(open_marker + r"\s*response\s*" + sep_marker)
        self._response_close_re = re.compile(close_marker + r"\s*response\s*" + sep_marker)
        self._message_close_re = re.compile(close_marker + r"\s*message\s*" + sep_marker)

        self._think_open_ids = tokenizer.encode(self._think_open, add_special_tokens=False)
        self._think_close_ids = tokenizer.encode(self._think_close, add_special_tokens=False)
        self._last_streaming_delta_token_ids: tuple[int, ...] | None = None
        self._last_streaming_content_token_ids: list[int] | None = None

    @property
    def reasoning_start_str(self) -> str | None:
        return self._think_open

    @property
    def reasoning_end_str(self) -> str | None:
        return self._think_close

    def adjust_request(
        self,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> "ChatCompletionRequest | ResponsesRequest":
        request.skip_special_tokens = False
        if hasattr(request, "spaces_between_special_tokens"):
            request.spaces_between_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if not self._thinking_enabled:
            return True
        last_close = _subseq_index(input_ids, self._think_close_ids)
        last_open = _subseq_index(input_ids, self._think_open_ids)
        if last_open == -1:
            return last_close != -1
        return last_close > last_open

    def _extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self._thinking_enabled:
            return input_ids
        idx = _subseq_index(input_ids, self._think_close_ids)
        if idx == -1:
            return []
        return input_ids[idx + len(self._think_close_ids) :]

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        cached_delta_ids = self._last_streaming_delta_token_ids
        cached_content_ids = self._last_streaming_content_token_ids
        self._last_streaming_delta_token_ids = None
        self._last_streaming_content_token_ids = None
        if cached_delta_ids == tuple(input_ids) and cached_content_ids is not None:
            return cached_content_ids
        return self._extract_content_ids(input_ids)

    def _strip_content_wrapper(self, text: str) -> str:
        m_ro = self._response_open_re.search(text)
        m_rc = self._response_close_re.search(text, m_ro.end() if m_ro else 0)
        if m_ro is not None and m_rc is not None:
            text = text[m_ro.end() : m_rc.start()]
        elif m_ro is not None:
            text = text[m_ro.end() :]
        else:
            text = self._response_open_re.sub("", text)
            text = self._response_close_re.sub("", text)
        text = self._message_close_re.sub("", text)
        return text

    @staticmethod
    def _should_preserve_tool_channels(
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> bool:
        return bool(getattr(request, "tools", None)) and (getattr(request, "tool_choice", None) != "none")

    def _content_after_reasoning(
        self,
        text: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> str | None:
        if self._should_preserve_tool_channels(request):
            return text or None
        return self._strip_content_wrapper(text) or None

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        if not self._thinking_enabled:
            return None, self._content_after_reasoning(model_output, request)

        m_open = self._think_open_re.search(model_output)
        content_start = m_open.end() if m_open is not None else 0
        if m_open is None and self._think_close_re.search(model_output) is None:
            return None, self._content_after_reasoning(model_output, request)

        m_close = self._think_close_re.search(model_output, content_start)
        if m_close is not None:
            reasoning = model_output[content_start : m_close.start()]
            rest = model_output[m_close.end() :]
            return (
                reasoning or None,
                self._content_after_reasoning(rest, request),
            )
        return (model_output[content_start:] or None, None)

    def _reasoning_text_ready_to_emit(self, text: str) -> str:
        m_open = self._think_open_re.search(text)
        if m_open is not None:
            text = text[m_open.end() :]
        overlap = 0
        for marker in (self._think_open, self._think_close):
            max_check = min(len(marker) - 1, len(text))
            for n in range(max_check, 0, -1):
                if text.endswith(marker[:n]):
                    overlap = max(overlap, n)
                    break
        return text[:-overlap] if overlap else text

    def _content_ready_to_emit(self, text: str) -> str:
        m_open = self._response_open_re.search(text)
        if m_open is not None:
            text = text[m_open.end() :]

        text = self._response_close_re.sub("", text)
        text = self._message_close_re.sub("", text)

        overlap = 0
        for marker in (
            self._response_open,
            self._response_close,
            self._message_close,
        ):
            max_check = min(len(marker) - 1, len(text))
            for n in range(max_check, 0, -1):
                if text.endswith(marker[:n]):
                    overlap = max(overlap, n)
                    break
        return text[:-overlap] if overlap else text

    def strip_content_streaming(
        self,
        previous_text: str,
        current_text: str,
    ) -> DeltaMessage | None:
        current_safe = self._content_ready_to_emit(current_text)
        previous_safe = self._content_ready_to_emit(previous_text)
        if current_safe.startswith(previous_safe):
            delta = current_safe[len(previous_safe) :]
        else:
            delta = current_safe
        return DeltaMessage(content=delta) if delta else None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        self._last_streaming_delta_token_ids = None
        self._last_streaming_content_token_ids = None
        if not self._thinking_enabled:
            return DeltaMessage(content=delta_text)

        if self._think_close_re.search(previous_text):
            return DeltaMessage(content=delta_text)

        m_close = self._think_close_re.search(current_text)
        if m_close is not None:
            self._last_streaming_delta_token_ids = tuple(delta_token_ids)
            self._last_streaming_content_token_ids = self._extract_content_ids(list(current_token_ids))
            m_open = self._think_open_re.search(current_text)
            r_start = m_open.end() if m_open is not None else 0
            reasoning = current_text[r_start : m_close.start()]
            already_sent = self._reasoning_text_ready_to_emit(previous_text)
            if reasoning.startswith(already_sent):
                reasoning_delta = reasoning[len(already_sent) :]
            else:
                reasoning_delta = reasoning
            content = current_text[m_close.end() :]
            return DeltaMessage(
                reasoning=reasoning_delta or None,
                content=content or None,
            )

        current_reasoning = self._reasoning_text_ready_to_emit(current_text)
        previous_reasoning = self._reasoning_text_ready_to_emit(previous_text)
        if current_reasoning.startswith(previous_reasoning):
            reasoning_delta = current_reasoning[len(previous_reasoning) :]
        else:
            reasoning_delta = current_reasoning
        if not reasoning_delta:
            return None
        return DeltaMessage(reasoning=reasoning_delta)

    extract_reasoning_content = extract_reasoning
    extract_reasoning_content_streaming = extract_reasoning_streaming
