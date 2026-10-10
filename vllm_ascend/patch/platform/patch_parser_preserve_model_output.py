#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Never let the engine-based tool-call parser silently drop model output.
#
# Background
# ----------
# Both failures below were reproduced on GLM-5.2 w4a8 (A3) with the pinned
# vLLM: a request whose model output is entirely a `<tool_call> ... </tool_call>`
# span comes back as `content=""` / `tool_calls=[]` while `usage` still counts
# every generated token, i.e. "HTTP 200 but empty", and an agent loop stalls.
#
# 1) `tool_choice="none"` (which is also the default of a request that
#    declares no `tools` at all) used to raise `_suppress_tool_calls`.  The
#    engine then consumes the whole tool-call span and the delegating layer
#    drops the parsed calls, so nothing at all reaches the client.
# 2) A span whose tool name is not part of `request.tools` produces no tool
#    call at all (`validate_tool_names` is on for GLM4.7/GLM-5) and the raw
#    text of that span is discarded instead of degrading back to content, so
#    the client only sees the half sentence that preceded the span.
#
# Fix
# ---
# 1) run the engine with `skip_tool_parsing` when `tool_choice="none"`: the
#    markup is then surfaced verbatim as content, exactly like a request
#    without a tool parser.  Tool parsing is untouched for `auto` / `required`
#    / named tool choices.
# 2) keep the raw text of every tool-call span and, when the span ends without
#    producing a tool call, append that text to `content` instead of dropping
#    it.  Spans that do become real tool calls are not modified.
#
# Both hooks only ever *add* text back; nothing is removed.
#

from __future__ import annotations

from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine

try:  # vLLM <= v0.26 keeps the delta protocol in the openai engine package.
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage  # type: ignore[import-not-found]
except ImportError:  # vLLM main moved it into the generate package.
    from vllm.entrypoints.generate.base.protocol import DeltaMessage  # type: ignore[import-not-found]


def _tool_call_emitted(parser: ParserEngine, tool_index: int) -> bool:
    """Whether the span at ``tool_index`` already produced a tool call."""
    if tool_index < 0:
        return True
    slots = getattr(parser, "_tool_slots", None)
    if not slots or tool_index >= len(slots):
        return False
    return bool(slots[tool_index].name_sent)


def _patch_skip_tool_parsing_for_tool_choice_none() -> None:
    if getattr(ParserEngine, "_vllm_ascend_keep_text_on_tool_choice_none", False):
        return

    original_check_skip_tool_parsing = ParserEngine._check_skip_tool_parsing

    def _check_skip_tool_parsing(self, request) -> None:
        original_check_skip_tool_parsing(self, request)
        if getattr(request, "tool_choice", None) != "none":
            return
        # The caller asked for no tool calls.  Swallowing the span here would
        # lose the whole answer, so let the engine run without tool parsing
        # instead: the markup is emitted as content, which is what a request
        # without an engine-based tool parser returns.
        self._suppress_tool_calls = False
        self.skip_tool_parsing = True

    ParserEngine._check_skip_tool_parsing = _check_skip_tool_parsing
    ParserEngine._vllm_ascend_keep_text_on_tool_choice_none = True


def _dropped_tool_span_text(
    parser: ParserEngine,
    spans: list[tuple[int, list[str]]],
) -> str:
    """Raw text of the spans that ended without producing a tool call."""
    dropped = ""
    for tool_index, parts in spans:
        if _tool_call_emitted(parser, tool_index):
            continue
        dropped += "".join(parts)
    return dropped


def _patch_tool_span_content_fallback() -> None:
    if getattr(ParserEngine, "_vllm_ascend_tool_span_text_fallback", False):
        return

    tool_text_event_types = (
        EventType.TOOL_CALL_START,
        EventType.TOOL_NAME,
        EventType.ARG_VALUE_CHUNK,
        EventType.TOOL_CALL_END,
    )
    original_events_to_delta = ParserEngine._events_to_delta
    original_reset = ParserEngine._reset

    def _reset(self, initial_state=None) -> None:
        original_reset(self, initial_state=initial_state)
        self._vllm_ascend_raw_tool_text = {}

    def _events_to_delta(self, events: list[SemanticEvent], finished: bool = False):
        raw_tool_text = getattr(self, "_vllm_ascend_raw_tool_text", None)
        if raw_tool_text is None:
            raw_tool_text = self._vllm_ascend_raw_tool_text = {}

        spans: list[tuple[int, list[str]]] = []
        for event in events:
            if event.type not in tool_text_event_types:
                continue
            if event.type is EventType.TOOL_CALL_START:
                # A new span starts: never mix in text left over from a span
                # that was truncated before its end marker.
                raw_tool_text.pop(event.tool_index, None)
            parts = raw_tool_text.setdefault(event.tool_index, [])
            parts.append(event.value)
            if event.type is EventType.TOOL_CALL_END:
                spans.append((event.tool_index, parts))

        result = original_events_to_delta(self, events, finished=finished)
        if not spans:
            return result

        for tool_index, _ in spans:
            raw_tool_text.pop(tool_index, None)
        fallback = _dropped_tool_span_text(self, spans)
        if not fallback:
            return result

        if result is None:
            result = DeltaMessage()
        result.content = (result.content or "") + fallback
        if fallback.strip():
            self._content_has_nonws = True
        return result

    ParserEngine._events_to_delta = _events_to_delta
    ParserEngine._reset = _reset
    ParserEngine._vllm_ascend_tool_span_text_fallback = True


_patch_skip_tool_parsing_for_tool_choice_none()
_patch_tool_span_content_fallback()
